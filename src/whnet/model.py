# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.utils.parametrizations import spectral_norm as _spectral_norm


SPATIAL_DIMS = (-3, -2, -1)


def _apply_spectral_norm(module, n_power_iterations=1):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv3d):
            _spectral_norm(child, n_power_iterations=n_power_iterations)
        else:
            _apply_spectral_norm(child, n_power_iterations)
    return module


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding="same"),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.CELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding="same"),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.CELU(),
        )

    def forward(self, x):
        return self.double_conv(x)


class UpsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding="same")

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="trilinear", align_corners=False)
        return F.celu(self.conv(x))


class ProximalNetwork(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        features=None,
        spectral_norm=True,
        normalized_conditioning=False,
    ):
        super().__init__()
        if features is None:
            features = [16, 32, 64]

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Retain the old state_dict key; the MLP controls the residual gain.
        self.alpha = nn.Parameter(torch.full((1,), 0.1), requires_grad=False)
        self.normalized_conditioning = normalized_conditioning

        in_c = in_channels
        for feature in features:
            self.downs.append(DoubleConv(in_c, feature))
            in_c = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        self.mlp = nn.Sequential(
            nn.Linear(2, 32), nn.GELU(), nn.Linear(32, 1), nn.Softplus()
        )

        nn.init.normal_(self.mlp[-2].weight, std=0.05)
        nn.init.constant_(self.mlp[-2].bias, -5.0 if normalized_conditioning else 0.5)

        for feature in reversed(features):
            self.ups.append(UpsampleConv(feature * 2, feature))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)
        nn.init.normal_(self.final_conv.weight, std=0.05)
        nn.init.zeros_(self.final_conv.bias)

        if spectral_norm:
            _apply_spectral_norm(self)

    def forward(self, x, t=0, updt=0):
        out = x
        skip_connections = []
        for down in self.downs:
            out = down(out)
            skip_connections.append(out)
            out = self.pool(out)

        out = self.bottleneck(out)

        skip_connections = skip_connections[::-1]
        for i in range(0, len(self.ups), 2):
            out = self.ups[i](out)
            skip = skip_connections[i // 2]
            if out.shape[2:] != skip.shape[2:]:
                out = F.interpolate(
                    out, size=skip.shape[2:], mode="trilinear", align_corners=False
                )
            out = self.ups[i + 1](torch.cat((skip, out), dim=1))

        t = torch.as_tensor(t, device=x.device, dtype=x.dtype).expand(x.shape[0], 1)
        updt = torch.as_tensor(updt, device=x.device, dtype=x.dtype).expand(
            x.shape[0], 1
        )
        if self.normalized_conditioning:
            t = t / 60.0
            updt = torch.log1p(updt.clamp_min(0))
        alpha = self.mlp(torch.cat([t, updt], dim=1))
        if self.normalized_conditioning:
            alpha = 0.25 * alpha / (1.0 + alpha)
        alpha = alpha.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        residual = self.final_conv(out)
        return x + alpha * residual


class ADMMUnrolledNet(nn.Module):
    def __init__(self, chinet, phinet, num_iters=5, eps=1e-6):
        super().__init__()
        self.chinet = chinet
        self.phinet = phinet
        self.num_iters = num_iters
        self.eps = eps

    def step_fn(
        self, chi, phi_h, y, u, v, eta_y, eta_u, eta_v, phi, mask, W, D, t, tol_res
    ):
        chi_fft = torch.fft.fftn(chi, dim=SPATIAL_DIMS)
        forward_physics = torch.fft.ifftn(D * chi_fft, dim=SPATIAL_DIMS).real
        total_field = forward_physics + phi_h

        data_resid = (total_field - phi) * mask
        resid_rms = (
            data_resid.pow(2).sum(dim=SPATIAL_DIMS)
            / mask.sum(dim=SPATIAL_DIMS).clamp(min=1.0)
        ).sqrt()

        y = (W * phi + (total_field + eta_y)) / (W + 1 + self.eps)

        u = self.chinet(chi + eta_u, t, tol_res)
        v = self.phinet(phi_h + eta_v, t, resid_rms)

        rhs_chi = torch.conj(D) * torch.fft.fftn(
            y - eta_y - phi_h, dim=SPATIAL_DIMS
        ) + torch.fft.fftn(u - eta_u, dim=SPATIAL_DIMS)
        denom_chi = torch.abs(D) ** 2 + 1 + self.eps
        chi_fft = rhs_chi / denom_chi
        chi = torch.fft.ifftn(chi_fft, dim=SPATIAL_DIMS).real * mask

        forward_physics = torch.fft.ifftn(D * chi_fft, dim=SPATIAL_DIMS).real * mask
        rhs_phi = torch.fft.fftn(
            y - eta_y - forward_physics, dim=SPATIAL_DIMS
        ) + torch.fft.fftn(v - eta_v, dim=SPATIAL_DIMS)
        denom_phi = 2
        phi_h = torch.fft.ifftn(rhs_phi / denom_phi, dim=SPATIAL_DIMS).real

        total_field = forward_physics + phi_h
        eta_y = eta_y + total_field - y
        eta_u = eta_u + chi - u
        eta_v = eta_v + phi_h - v

        return (chi, phi_h, y, u, v, eta_y, eta_u, eta_v)

    def forward(
        self,
        phi,
        mask,
        D,
        W=None,
        return_iterates=False,
        tol=None,
        max_iters=None,
        iteration_callback=None,
    ):
        phi = phi.float()
        mask = mask.float()
        W = mask if W is None else W.float()

        chi = torch.zeros_like(phi)
        phi_h = torch.zeros_like(phi)
        y = (W * phi) / (W + 1 + self.eps)
        u = torch.zeros_like(phi)
        v = torch.zeros_like(phi)
        eta_y = torch.zeros_like(phi)
        eta_u = torch.zeros_like(phi)
        eta_v = torch.zeros_like(phi)

        tol_res = torch.zeros(phi.shape[:2], device=phi.device)

        n_iters = self.num_iters if max_iters is None else max_iters
        use_ckpt = torch.is_grad_enabled()
        iterates = []

        for k in range(n_iters):
            chi_prev = chi
            if use_ckpt:
                (chi, phi_h, y, u, v, eta_y, eta_u, eta_v) = checkpoint(
                    self.step_fn,
                    chi,
                    phi_h,
                    y,
                    u,
                    v,
                    eta_y,
                    eta_u,
                    eta_v,
                    phi,
                    mask,
                    W,
                    D,
                    k,
                    tol_res,
                    use_reentrant=False,
                )
            else:
                (chi, phi_h, y, u, v, eta_y, eta_u, eta_v) = self.step_fn(
                    chi,
                    phi_h,
                    y,
                    u,
                    v,
                    eta_y,
                    eta_u,
                    eta_v,
                    phi,
                    mask,
                    W,
                    D,
                    k,
                    tol_res,
                )

            if return_iterates:
                iterates.append((chi, phi_h))
            if iteration_callback is not None:
                iteration_callback(k + 1, chi, phi_h)

            if tol is not None and k > 0:
                rel = torch.norm(chi - chi_prev) / (torch.norm(chi) + self.eps)
                if rel < tol:
                    break

            data_resid = (chi_prev - chi) * mask
            tol_res = (
                data_resid.pow(2).sum(dim=SPATIAL_DIMS)
                / mask.sum(dim=SPATIAL_DIMS).clamp(min=1.0)
            ).sqrt()

        if return_iterates:
            return iterates
        return chi, phi_h


class EfficientADMMUnrolledNet(ADMMUnrolledNet):
    """Projected learned ADMM with consistent fields and FP32 real FFTs.

    The support projection is an approximate constrained chi update, not an exact
    support-constrained quadratic solve. All subsequent fields use that projected
    chi. Legacy ADMMUnrolledNet remains available for old checkpoints.
    """

    def __init__(
        self,
        chinet,
        phinet,
        num_iters=20,
        eps=1e-6,
        normalize_input=True,
        checkpoint_steps=True,
        input_scale_floor=1e-6,
    ):
        super().__init__(chinet, phinet, num_iters, eps)
        self.register_buffer("_pipeline_version", torch.tensor(2, dtype=torch.int64))
        self.normalize_input = normalize_input
        self.input_scale_floor = input_scale_floor
        self.checkpoint_steps = checkpoint_steps
        self.last_num_iters = 0
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        # Buffer estimates must not change during checkpoint recomputation.
        for module in self.modules():
            if hasattr(module, "_u") and hasattr(module, "_v"):
                module.eval()
        return self

    @torch.no_grad()
    def refresh_spectral_norm(self, iterations=1):
        """Update estimates once per optimizer step, before forward/backward."""
        for module in self.modules():
            if not isinstance(module, nn.Conv3d) or not hasattr(
                module, "parametrizations"
            ):
                continue
            for parametrization in module.parametrizations.weight:
                if hasattr(parametrization, "_u") and hasattr(parametrization, "_v"):
                    parametrization.train()
                    try:
                        for _ in range(iterations):
                            _ = module.weight
                    finally:
                        parametrization.eval()

    def _rms(self, x, mask, count):
        return ((x.square() * mask).sum(dim=SPATIAL_DIMS) / count + self.eps**2).sqrt()

    def efficient_step(
        self,
        chi,
        phi_h,
        eta_y,
        eta_u,
        eta_v,
        phase,
        mask,
        weights,
        dipole,
        denominator,
        count,
        t,
        update_rms,
    ):
        # Convolutions inherit the surrounding BF16 autocast; physics stays FP32.
        shape = phase.shape[-3:]

        def fft(x):
            return torch.fft.rfftn(x.float(), dim=SPATIAL_DIMS)

        def inverse(x):
            return torch.fft.irfftn(x, s=shape, dim=SPATIAL_DIMS)

        field = inverse(dipole * fft(chi))
        total = field + phi_h
        residual_rms = self._rms(total - phase, mask, count)
        y = (weights * phase + total + eta_y) / (weights + 1.0)
        u = self.chinet(chi + eta_u, t, update_rms).float()
        v = self.phinet(phi_h + eta_v, t, residual_rms).float()
        chi_fft = (dipole * fft(y - eta_y - phi_h) + fft(u - eta_u)) / denominator
        chi = inverse(chi_fft) * mask
        # Recompute after the support projection; masking and A do not commute.
        field = inverse(dipole * fft(chi))
        phi_h = 0.5 * (y - eta_y - field + v - eta_v)
        return (
            chi,
            phi_h,
            eta_y + field + phi_h - y,
            eta_u + chi - u,
            eta_v + phi_h - v,
        )

    def forward(
        self,
        phi,
        mask,
        D,
        W=None,
        return_iterates=False,
        tol=None,
        max_iters=None,
        supervision_indices=None,
    ):
        phase, mask = phi.float(), mask.float()
        weights = mask if W is None else W.float()
        count = mask.sum(dim=SPATIAL_DIMS).clamp_min(1.0)
        scale = torch.ones_like(count)
        if self.normalize_input:
            scale = self._rms(phase, mask, count).clamp_min(self.input_scale_floor)
        scale = scale[..., None, None, None]
        phase = phase / scale
        # Accept full kernels for compatibility and half kernels from the new loader.
        dipole = D.float()[..., : phase.shape[-1] // 2 + 1]
        denominator = dipole.square() + 1.0 + self.eps
        chi, phi_h, eta_y, eta_u, eta_v = [torch.zeros_like(phase) for _ in range(5)]
        update_rms = torch.zeros_like(count)
        n_iters = self.num_iters if max_iters is None else max_iters
        if n_iters < 1:
            raise ValueError("max_iters must be positive")
        selected = None if supervision_indices is None else set(supervision_indices)
        if selected is not None and (
            not selected or min(selected) < 0 or max(selected) >= n_iters
        ):
            raise ValueError("Invalid supervision indices")
        iterates = []
        self.last_num_iters = 0
        for k in range(n_iters):
            previous_chi, previous_phi = chi, phi_h
            args = (
                chi,
                phi_h,
                eta_y,
                eta_u,
                eta_v,
                phase,
                mask,
                weights,
                dipole,
                denominator,
                count,
                phase.new_tensor(float(k)),
                update_rms,
            )
            if torch.is_grad_enabled() and self.checkpoint_steps:
                chi, phi_h, eta_y, eta_u, eta_v = checkpoint(
                    self.efficient_step,
                    *args,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                chi, phi_h, eta_y, eta_u, eta_v = self.efficient_step(*args)
            self.last_num_iters = k + 1
            if return_iterates and (selected is None or k in selected):
                iterates.append((chi * scale, phi_h * scale))
            update_rms = self._rms(chi - previous_chi, mask, count)
            if tol is not None and k >= 9:
                chi_rel = update_rms / self._rms(chi, mask, count).clamp_min(self.eps)
                phi_rel = self._rms(phi_h - previous_phi, mask, count) / self._rms(
                    phi_h, mask, count
                ).clamp_min(self.eps)
                if bool(((chi_rel < tol) & (phi_rel < tol)).all()):
                    break
        return iterates if return_iterates else (chi * scale, phi_h * scale)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prox = ADMMUnrolledNet(ProximalNetwork(), ProximalNetwork()).to(device)
    im = torch.ones([5, 1, 11, 140, 140]).float().to(device)
    out = prox(im, im, im, return_iterates=True)
    print(out[0][0].shape)
# %%

#%%
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
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding='same'),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.CELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding='same'),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.CELU(),
        )

    def forward(self, x):
        return self.double_conv(x)


class UpsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding='same')

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='trilinear', align_corners=False)
        return F.celu(self.conv(x))


class ProximalNetwork(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, features=None,
                 spectral_norm=True):
        super().__init__()
        if features is None:
            features = [16, 32, 64]

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # A proximal module should start close to the identity. The previous
        # alpha parameter was never used; with spectral normalization the
        # small final-convolution initialization does not make the residual
        # small by itself. This logit controls a bounded residual gate.
        self.alpha = nn.Parameter(torch.full((1,), -4.0))

        in_c = in_channels
        for feature in features:
            self.downs.append(DoubleConv(in_c, feature))
            in_c = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        self.mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # Initially the condition-dependent contribution is zero, so the
        # scalar gate is approximately 0.25 * sigmoid(-4) = 4.5e-3.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        for feature in reversed(features):
            self.ups.append(UpsampleConv(feature * 2, feature))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)
        nn.init.normal_(self.final_conv.weight, std=0.05)
        nn.init.zeros_(self.final_conv.bias)

        if spectral_norm:
            _apply_spectral_norm(self)

    @staticmethod
    def _batch_feature(value, x):
        """Convert a scalar or tensor to one floating feature per sample."""
        if not torch.is_tensor(value):
            return x.new_full((x.shape[0], 1), float(value))

        value = value.to(device=x.device, dtype=x.dtype)
        if value.ndim == 0 or value.numel() == 1:
            return value.reshape(1, 1).expand(x.shape[0], 1)
        if value.shape[0] != x.shape[0]:
            raise ValueError(
                f"Condition batch size {value.shape[0]} does not match input "
                f"batch size {x.shape[0]}."
            )
        return value.reshape(x.shape[0], -1).mean(dim=1, keepdim=True)

    def forward(self, x, t=0.0, updt=0.0):
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
                out = F.interpolate(out, size=skip.shape[2:],
                                    mode='trilinear', align_corners=False)
            out = self.ups[i + 1](torch.cat((skip, out), dim=1))

        # t is supplied by ADMM normalized to [0, 1]. Log-compressing the
        # residual feature avoids a large change of scale across noise levels.
        t_feature = self._batch_feature(t, x).clamp(0.0, 1.0)
        update_feature = torch.log1p(self._batch_feature(updt, x).abs())
        condition = torch.cat([t_feature, update_feature], dim=1)
        alpha = 0.25 * torch.sigmoid(self.alpha + self.mlp(condition))
        alpha = alpha.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        residual = self.final_conv(out)
        return x + alpha * residual


class ADMMUnrolledNet(nn.Module):

    def __init__(self, chinet, phinet, num_iters=5, eps=1e-6,
                 rho_y=0.1, rho_u=0.1, rho_v=0.1,
                 conditioning_iters=12, mask_chi=True,
                 normalize_input=True, input_scale_floor=1e-6,
                 gradient_checkpointing=True,
                 # Backward-compatible aliases for older experiment scripts.
                 mu1=None, mu2=None, mu3=None):
        super().__init__()
        self.chinet = chinet
        self.phinet = phinet
        self.num_iters = num_iters
        self.eps = eps
        self.rho_y = float(rho_y if mu1 is None else mu1)
        self.rho_u = float(rho_u if mu2 is None else mu2)
        self.rho_v = float(rho_v if mu3 is None else mu3)
        if min(self.rho_y, self.rho_u, self.rho_v) <= 0:
            raise ValueError("ADMM penalties rho_y, rho_u, and rho_v must be positive.")
        if conditioning_iters < 2:
            raise ValueError("conditioning_iters must be at least two.")
        if input_scale_floor <= 0:
            raise ValueError("input_scale_floor must be positive.")
        self.conditioning_iters = int(conditioning_iters)
        self.mask_chi = bool(mask_chi)
        self.normalize_input = bool(normalize_input)
        self.input_scale_floor = float(input_scale_floor)
        self.gradient_checkpointing = bool(gradient_checkpointing)

    @staticmethod
    def _expand_spatial(x):
        return x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    def _physics(self, chi, D):
        return torch.fft.ifftn(
            D * torch.fft.fftn(chi, dim=SPATIAL_DIMS), dim=SPATIAL_DIMS
        ).real

    def _roi_rms(self, x, mask):
        numerator = (x.square() * mask).sum(dim=SPATIAL_DIMS)
        denominator = mask.sum(dim=SPATIAL_DIMS).clamp_min(1.0)
        return (numerator / denominator).sqrt()

    def _closed_form_chi(self, q, r, mask, D):
        """Solve the unconstrained chi quadratic in the Fourier domain.

        When ``mask_chi`` is enabled, the exact full-FOV minimizer is projected
        into the ROI afterward. Since masking and dipole convolution do not
        commute, that projection is not the exact hard-support minimizer.
        """
        q_fft = torch.fft.fftn(q, dim=SPATIAL_DIMS)
        r_fft = torch.fft.fftn(r, dim=SPATIAL_DIMS)
        denominator = (
            self.rho_y * torch.abs(D).square() + self.rho_u
        )
        chi_fft = (
            self.rho_y * torch.conj(D) * q_fft + self.rho_u * r_fft
        ) / denominator
        chi = torch.fft.ifftn(chi_fft, dim=SPATIAL_DIMS).real
        return chi * mask if self.mask_chi else chi

    def step_fn(self, chi, phi_h, y, u, v, eta_y, eta_u, eta_v,
                phi, mask, W, D, t, tol_res):
        # Scaled ADMM objective:
        # 0.5 ||sqrt(W) (y - phi)||^2 + R_chi(u) + R_h(v),
        # with y = H chi + phi_h, u = chi, and v = phi_h. W is a
        # non-negative data-weight map, typically the mask or normalized
        # magnitude image.
        forward_physics = self._physics(chi, D)
        total_field = forward_physics + phi_h

        data_resid = total_field - phi
        resid_rms = (
            (W * data_resid.square()).sum(dim=SPATIAL_DIMS)
            / W.sum(dim=SPATIAL_DIMS).clamp_min(1.0)
        ).sqrt()

        y = (
            W * phi + self.rho_y * (total_field + eta_y)
        ) / (W + self.rho_y + self.eps)

        prior_chi = self.chinet(chi + eta_u, t, tol_res)
        # The susceptibility prior is explicitly restricted to the ROI in the
        # support-constrained formulation.
        u = mask * prior_chi if self.mask_chi else prior_chi
        v = self.phinet(phi_h + eta_v, t, resid_rms)

        q = y - eta_y - phi_h
        r = u - eta_u
        chi = self._closed_form_chi(q, r, mask, D)

        forward_physics = self._physics(chi, D)
        phi_h = (
            self.rho_y * (y - eta_y - forward_physics)
            + self.rho_v * (v - eta_v)
        ) / (self.rho_y + self.rho_v)

        total_field = forward_physics + phi_h
        eta_y = eta_y + total_field - y
        eta_u = eta_u + chi - u
        eta_v = eta_v + phi_h - v

        return chi, phi_h, y, u, v, eta_y, eta_u, eta_v

    def forward(self, phi, mask, D, W=None, return_iterates=False, tol=None, max_iters=None):
        if phi.ndim != 5:
            raise ValueError(f"phi must have shape [B, C, D, H, W], got {tuple(phi.shape)}.")
        if mask.shape != phi.shape:
            raise ValueError(f"mask shape {tuple(mask.shape)} must equal phi shape {tuple(phi.shape)}.")

        phi = phi.float()
        mask = mask.to(device=phi.device, dtype=phi.dtype).clamp(0.0, 1.0)

        if self.normalize_input:
            input_scale = self._roi_rms(phi, mask).clamp_min(
                self.input_scale_floor
            )
            phi = phi / self._expand_spatial(input_scale)
        else:
            input_scale = torch.ones(
                phi.shape[:2], device=phi.device, dtype=phi.dtype
            )

        if W is None:
            W = mask
        else:
            W = W.to(device=phi.device, dtype=phi.dtype)
            W = torch.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
            W = W.clamp_min(0.0)
            W = W * mask
        D = D.to(device=phi.device, dtype=phi.dtype)
        if D.ndim == 3:
            D = D.unsqueeze(0).unsqueeze(0)
        elif D.ndim == 4:
            D = D.unsqueeze(0)
        if D.ndim != 5 or D.shape[-3:] != phi.shape[-3:]:
            raise ValueError(
                "D must be broadcastable to [B, C, D, H, W] with the same spatial shape as phi."
            )

        chi = torch.zeros_like(phi)
        phi_h = torch.zeros_like(phi)
        y = (W * phi) / (W + self.rho_y + self.eps)
        u = torch.zeros_like(phi)
        v = torch.zeros_like(phi)
        eta_y = torch.zeros_like(phi)
        eta_u = torch.zeros_like(phi)
        eta_v = torch.zeros_like(phi)

        tol_res = torch.zeros(phi.shape[:2], device=phi.device, dtype=phi.dtype)

        n_iters = self.num_iters if max_iters is None else max_iters
        if n_iters < 1:
            raise ValueError("num_iters and max_iters must be at least one.")
        use_ckpt = torch.is_grad_enabled() and self.gradient_checkpointing
        iterates = []

        for k in range(n_iters):
            chi_prev = chi
            y_prev, u_prev, v_prev = y, u, v
            # Use an absolute training-depth coordinate. Normalizing by the
            # requested inference depth would change every earlier update when
            # max_iters changes, preventing consistent anytime inference.
            t = min(float(k) / (self.conditioning_iters - 1), 1.0)
            if use_ckpt:
                (chi, phi_h, y, u, v, eta_y, eta_u, eta_v) = checkpoint(
                    self.step_fn, chi, phi_h, y, u, v, eta_y, eta_u, eta_v,
                    phi, mask, W, D, t, tol_res, use_reentrant=False
                )
            else:
                (chi, phi_h, y, u, v, eta_y, eta_u, eta_v) = self.step_fn(
                    chi, phi_h, y, u, v, eta_y, eta_u, eta_v,
                    phi, mask, W, D, t, tol_res
                )

            if return_iterates:
                iterates.append((chi, phi_h))

            tol_res = self._roi_rms(chi - chi_prev, mask)

            if tol is not None and k > 0:
                primal = torch.stack((
                    self._roi_rms(self._physics(chi, D) + phi_h - y, mask),
                    self._roi_rms(chi - u, mask),
                    self._roi_rms(phi_h - v, mask),
                )).amax()
                dual = torch.stack((
                    self.rho_y * self._roi_rms(y - y_prev, mask),
                    self.rho_u * self._roi_rms(u - u_prev, mask),
                    self.rho_v * self._roi_rms(v - v_prev, mask),
                )).amax()
                if primal < tol and dual < tol:
                    break

        if return_iterates:
            scale = self._expand_spatial(input_scale)
            return [
                (chi_iter * scale, phi_iter * scale)
                for chi_iter, phi_iter in iterates
            ]
        scale = self._expand_spatial(input_scale)
        return chi * scale, phi_h * scale
    
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prox = ADMMUnrolledNet(ProximalNetwork(), ProximalNetwork()).to(device)
    im = torch.ones([5, 1, 11, 140, 140]).float().to(device)
    out = prox(im, im, im, return_iterates=True)
    print(out[0][0].shape)
# %%

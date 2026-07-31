#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.utils.parametrizations import spectral_norm as _spectral_norm


SPATIAL_DIMS = (-3, -2, -1)


def masked_rms(x, mask):
    """Return the mask-aware RMS of each sample/channel."""
    work_dtype = (
        torch.float16
        if x.device.type == "cuda" and x.dtype == torch.float16
        else torch.float32
    )
    x_work = x.to(dtype=work_dtype)
    mask_work = mask.to(dtype=work_dtype)
    numerator_norm = torch.linalg.vector_norm(
        x_work * mask_work.sqrt(), dim=SPATIAL_DIMS, keepdim=True
    )
    denominator = mask_work.sum(
        dim=SPATIAL_DIMS, keepdim=True, dtype=torch.float32
    ).clamp_min(1.0).sqrt()
    # vector_norm defines a zero subgradient at the origin, unlike a manually
    # expanded sqrt(sum(x^2)), while preserving exact positive homogeneity.
    rms = numerator_norm / denominator
    return rms.to(dtype=x.dtype)


def masked_rms_scale(x, mask):
    """Return one positive, padding-independent RMS scale per sample/channel."""
    rms = masked_rms(x, mask)
    return torch.where(rms > 0, rms, torch.ones_like(rms)).to(dtype=x.dtype)


def weighted_rms(x, weight, fallback_mask=None):
    """Return an RMS driven by reliable voxels, with an ROI fallback."""
    x_float = x.float()
    weight_float = torch.nan_to_num(
        weight.float(), nan=0.0, posinf=0.0, neginf=0.0
    ).clamp_min(0.0)
    denominator = weight_float.sum(dim=SPATIAL_DIMS, keepdim=True)
    weighted = (
        (x_float.square() * weight_float).sum(dim=SPATIAL_DIMS, keepdim=True)
        / denominator.clamp_min(1e-6)
    ).sqrt()
    if fallback_mask is None:
        fallback_mask = (weight_float > 0).to(dtype=x_float.dtype)
    fallback = masked_rms(x_float, fallback_mask.float())
    return torch.where(denominator > 0, weighted, fallback).to(dtype=x.dtype)


def weighted_rms_scale(x, weight, fallback_mask=None):
    """Return a positive, division-safe reliability-weighted RMS scale."""
    scale = weighted_rms(x, weight, fallback_mask=fallback_mask)
    return torch.where(scale > 0, scale, torch.ones_like(scale)).to(dtype=x.dtype)


def normalize_confidence(confidence, mask):
    """Normalize a non-negative confidence map to unit ROI mean."""
    confidence = torch.nan_to_num(
        confidence.float(), nan=0.0, posinf=0.0, neginf=0.0
    ).clamp_min(0.0)
    mask_float = mask.float()
    confidence = confidence * mask_float
    count = mask_float.sum(dim=SPATIAL_DIMS, keepdim=True).clamp_min(1.0)
    roi_mean = confidence.sum(dim=SPATIAL_DIMS, keepdim=True) / count
    normalized = confidence / roi_mean.clamp_min(1e-6)
    normalized = normalized.clamp_max(10.0) * mask_float
    clipped_mean = normalized.sum(dim=SPATIAL_DIMS, keepdim=True) / count
    normalized = torch.where(
        clipped_mean > 0,
        normalized / clipped_mean.clamp_min(1e-6),
        normalized,
    )
    return normalized.to(dtype=mask.dtype) * mask


def magnitude_to_data_weight(magnitude, mask):
    """Convert magnitude to unit-ROI-mean phase precision proportional to W²."""
    magnitude = torch.nan_to_num(
        magnitude.float(), nan=0.0, posinf=0.0, neginf=0.0
    ).clamp_min(0.0) * mask.float()
    precision = magnitude.square()
    count = mask.float().sum(dim=SPATIAL_DIMS, keepdim=True).clamp_min(1.0)
    roi_mean = precision.sum(dim=SPATIAL_DIMS, keepdim=True) / count
    normalized = precision / roi_mean.clamp_min(1e-6)
    normalized = normalized.clamp_max(20.0) * mask.float()
    clipped_mean = normalized.sum(dim=SPATIAL_DIMS, keepdim=True) / count
    return torch.where(
        clipped_mean > 0,
        normalized / clipped_mean.clamp_min(1e-6),
        normalized,
    ) * mask.float()


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

        self.alpha = nn.Parameter(torch.full((1,), 0.1))

        in_c = in_channels
        for feature in features:
            self.downs.append(DoubleConv(in_c, feature))
            in_c = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        self.mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )

        nn.init.normal_(self.mlp[-2].weight, std=0.05)
        nn.init.constant_(self.mlp[-2].bias, 0.5)

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
                out = F.interpolate(out, size=skip.shape[2:],
                                    mode='trilinear', align_corners=False)
            out = self.ups[i + 1](torch.cat((skip, out), dim=1))

        t = torch.tensor(t, device=updt.device).unsqueeze(0).repeat(x.shape[0]).unsqueeze(1)
        alpha = self.mlp(torch.cat([t, updt], dim=1)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        residual = self.final_conv(out)
        self.last_alpha = alpha.detach()
        return x + alpha * residual


class ADMMUnrolledNet(nn.Module):

    def __init__(self, chinet, phinet, num_iters=5, eps=1e-6):
        super().__init__()
        self.chinet = chinet
        self.phinet = phinet
        self.num_iters = num_iters
        self.eps = eps

    def step_fn(self, chi, phi_h, y, u, v, eta_y, eta_u, eta_v,
                phi, mask, W, D, t, tol_res):

        # Recurrent state may be FP16 under autocast, but CUDA half FFTs only
        # support restricted shapes. Promote FFT inputs and physics to FP32,
        # then store the updated ADMM state back in its compact dtype.
        state_dtype = chi.dtype
        chi_fft = torch.fft.fftn(chi.float(), dim=SPATIAL_DIMS)
        forward_physics = torch.fft.ifftn(
            D.float() * chi_fft, dim=SPATIAL_DIMS
        ).real
        total_field = forward_physics + phi_h.float()

        data_resid = (total_field - phi.float()) * mask.float()
        resid_rms = (
            data_resid.pow(2).sum(dim=SPATIAL_DIMS)
            / mask.sum(dim=SPATIAL_DIMS).clamp(min=1.0)
        ).sqrt()

        y = (
            (W.float() * phi.float() + total_field + eta_y.float())
            / (W.float() + 1 + self.eps)
        ).to(dtype=state_dtype)

        u = self.chinet(chi + eta_u, t, tol_res).to(dtype=state_dtype)
        v = self.phinet(phi_h + eta_v, t, resid_rms).to(dtype=state_dtype)

        rhs_chi = (
            torch.conj(D.float())
            * torch.fft.fftn(
                (y - eta_y - phi_h).float(), dim=SPATIAL_DIMS
            )
            + torch.fft.fftn((u - eta_u).float(), dim=SPATIAL_DIMS)
        )
        denom_chi = torch.abs(D.float()) ** 2 + 1 + self.eps
        chi_fft = rhs_chi / denom_chi
        chi = (
            torch.fft.ifftn(chi_fft, dim=SPATIAL_DIMS).real
            * mask.float()
        ).to(dtype=state_dtype)

        forward_physics = (
            torch.fft.ifftn(D.float() * chi_fft, dim=SPATIAL_DIMS).real
            * mask.float()
        )
        rhs_phi = (
            torch.fft.fftn(
                y.float() - eta_y.float() - forward_physics,
                dim=SPATIAL_DIMS,
            )
            + torch.fft.fftn((v - eta_v).float(), dim=SPATIAL_DIMS)
        )
        denom_phi = 2
        phi_h = torch.fft.ifftn(
            rhs_phi / denom_phi, dim=SPATIAL_DIMS
        ).real.to(dtype=state_dtype)

        total_field = forward_physics.to(dtype=state_dtype) + phi_h
        eta_y = eta_y + total_field - y
        eta_u = eta_u + chi - u
        eta_v = eta_v + phi_h - v

        return (chi, phi_h, y, u, v, eta_y, eta_u, eta_v)

    def forward(self, phi, mask, D, W=None, return_iterates=False,
                return_diagnostics=False, tol=None, max_iters=None):

        phi = phi.float()
        mask = mask.float()
        W = mask if W is None else W.float()
        use_fp16_state = (
            phi.device.type == "cuda"
            and torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") == torch.float16
        )
        state_dtype = torch.float16 if use_fp16_state else phi.dtype
        state_mask = mask.to(dtype=state_dtype)

        chi = torch.zeros_like(phi, dtype=state_dtype)
        phi_h = torch.zeros_like(phi, dtype=state_dtype)
        y = ((W * phi) / (W + 1 + self.eps)).to(dtype=state_dtype)
        u = torch.zeros_like(phi, dtype=state_dtype)
        v = torch.zeros_like(phi, dtype=state_dtype)
        eta_y = torch.zeros_like(phi, dtype=state_dtype)
        eta_u = torch.zeros_like(phi, dtype=state_dtype)
        eta_v = torch.zeros_like(phi, dtype=state_dtype)

        tol_res = torch.zeros(
            phi.shape[:2], device=phi.device, dtype=state_dtype
        )

        n_iters = self.num_iters if max_iters is None else max_iters
        use_ckpt = torch.is_grad_enabled()
        iterates = []
        diagnostic_lists = {
            "chi_step": [],
            "phi_step": [],
            "primal": [],
            "dual": [],
        }

        for k in range(n_iters):
            chi_prev = chi
            phi_h_prev = phi_h
            y_prev = y
            u_prev = u
            v_prev = v
            eta_y_prev = eta_y
            eta_u_prev = eta_u
            eta_v_prev = eta_v
            if use_ckpt:
                (chi, phi_h, y, u, v, eta_y, eta_u, eta_v) = checkpoint(
                    self.step_fn, chi, phi_h, y, u, v, eta_y, eta_u, eta_v,
                    phi, mask, W, D, k, tol_res,
                    use_reentrant=False
                )
            else:
                (chi, phi_h, y, u, v, eta_y, eta_u, eta_v) = self.step_fn(
                    chi, phi_h, y, u, v, eta_y, eta_u, eta_v,
                    phi, mask, W, D, k, tol_res
                )

            chi_step = masked_rms(
                (chi - chi_prev) * state_mask, state_mask
            ).flatten(1).float()
            phi_step = masked_rms(
                (phi_h - phi_h_prev) * state_mask, state_mask
            ).flatten(1).float()
            primal = torch.stack((
                masked_rms(
                    (eta_y - eta_y_prev) * state_mask, state_mask
                ).flatten(1),
                masked_rms(
                    (eta_u - eta_u_prev) * state_mask, state_mask
                ).flatten(1),
                masked_rms(
                    (eta_v - eta_v_prev) * state_mask, state_mask
                ).flatten(1),
            )).amax(dim=0).float()
            dual = torch.stack((
                masked_rms((y - y_prev) * state_mask, state_mask).flatten(1),
                masked_rms((u - u_prev) * state_mask, state_mask).flatten(1),
                masked_rms((v - v_prev) * state_mask, state_mask).flatten(1),
            )).amax(dim=0).float()
            diagnostic_lists["chi_step"].append(chi_step)
            diagnostic_lists["phi_step"].append(phi_step)
            diagnostic_lists["primal"].append(primal)
            diagnostic_lists["dual"].append(dual)

            if return_iterates:
                iterates.append((chi, phi_h))

            if tol is not None and k > 0:
                relative_change = (
                    torch.linalg.vector_norm(chi - chi_prev)
                    / torch.linalg.vector_norm(chi).clamp_min(self.eps)
                )
                if relative_change < tol:
                    break

            data_resid = (chi_prev - chi) * state_mask
            tol_res = (
                data_resid.pow(2).sum(dim=SPATIAL_DIMS)
                / mask.sum(dim=SPATIAL_DIMS).clamp(min=1.0)
            ).sqrt()

        diagnostics = {
            name: torch.stack(values, dim=0)
            for name, values in diagnostic_lists.items()
        }

        if return_iterates and return_diagnostics:
            return iterates, diagnostics
        if return_iterates:
            return iterates
        outputs = (chi.float(), phi_h.float())
        if return_diagnostics:
            return (*outputs, diagnostics)
        return outputs
    
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prox = ADMMUnrolledNet(ProximalNetwork(), ProximalNetwork()).to(device)
    im = torch.ones([5, 1, 11, 140, 140]).float().to(device)
    out = prox(im, im, im, return_iterates=True)
    print(out[0][0].shape)
# %%

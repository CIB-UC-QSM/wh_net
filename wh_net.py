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
        return x + alpha * residual


class ADMMUnrolledNet(nn.Module):

    """Unrolled ADMM for the masked susceptibility regularizer.

    The susceptibility term is ``R_chi(M * chi)``. Its scaled ADMM split is
    therefore ``u = M * chi`` (rather than ``u = chi``), while ``v = phi_h``
    and ``y = D * chi + phi_h`` retain their original meanings. The mask does
    not commute with dipole convolution, so the resulting chi normal equation
    is solved with a fixed number of conjugate-gradient iterations instead of
    the previous element-wise Fourier-domain division.
    """

    def __init__(self, chinet, phinet, num_iters=5, eps=1e-6, cg_iters=4):
        super().__init__()
        self.chinet = chinet
        self.phinet = phinet
        self.num_iters = num_iters
        self.eps = eps
        if cg_iters < 1:
            raise ValueError("cg_iters must be at least one.")
        self.cg_iters = int(cg_iters)

    def _apply_chi_normal(self, chi, mask, D):
        """Apply ``DᴴD + MᴴM + eps I`` to ``chi``."""
        chi_fft = torch.fft.fftn(chi, dim=SPATIAL_DIMS)
        data_term = torch.fft.ifftn(
            torch.abs(D) ** 2 * chi_fft, dim=SPATIAL_DIMS
        ).real
        return data_term + mask.square() * chi + self.eps * chi

    def _solve_chi(self, rhs, mask, D, initial):
        """Solve the masked chi normal equation with batched CG."""
        chi = initial
        residual = rhs - self._apply_chi_normal(chi, mask, D)
        direction = residual
        residual_norm = residual.square().sum(
            dim=SPATIAL_DIMS, keepdim=True
        )

        for _ in range(self.cg_iters):
            normal_direction = self._apply_chi_normal(direction, mask, D)
            denominator = (direction * normal_direction).sum(
                dim=SPATIAL_DIMS, keepdim=True
            ).clamp_min(self.eps)
            step = residual_norm / denominator
            chi = chi + step * direction
            residual = residual - step * normal_direction
            next_residual_norm = residual.square().sum(
                dim=SPATIAL_DIMS, keepdim=True
            )
            direction = residual + (
                next_residual_norm / residual_norm.clamp_min(self.eps)
            ) * direction
            residual_norm = next_residual_norm

        return chi

    def step_fn(self, chi, phi_h, y, u, v, eta_y, eta_u, eta_v, phi, mask, W, D, t, tol_res):

        chi_fft = torch.fft.fftn(chi, dim=SPATIAL_DIMS)
        forward_physics = torch.fft.ifftn(D * chi_fft, dim=SPATIAL_DIMS).real
        total_field = forward_physics + phi_h

        data_resid = (total_field - phi) * mask
        resid_rms = (data_resid.pow(2).sum(dim=SPATIAL_DIMS) / mask.sum(dim=SPATIAL_DIMS).clamp(min=1.0)).sqrt()

        y = (W * phi +  (total_field + eta_y)) / (W + 1 + self.eps)

        # R_chi acts on M * chi, so u is the auxiliary variable for that
        # masked quantity. eta_u is already in the same auxiliary space.
        u = self.chinet(mask * chi + eta_u, t, tol_res)
        v = self.phinet(phi_h + eta_v, t, resid_rms)

        # (DᴴD + MᴴM + eps I) chi = Dᴴ(y - eta_y - phi_h)
        #                              + Mᴴ(u - eta_u)
        chi_rhs = y - eta_y - phi_h
        rhs_chi = (
            torch.fft.ifftn(
                torch.conj(D) * torch.fft.fftn(chi_rhs, dim=SPATIAL_DIMS),
                dim=SPATIAL_DIMS,
            ).real
            + mask * (u - eta_u)
        )
        chi = self._solve_chi(rhs_chi, mask, D, initial=chi)
        chi_fft = torch.fft.fftn(chi, dim=SPATIAL_DIMS)

        forward_physics = torch.fft.ifftn(D * chi_fft, dim=SPATIAL_DIMS).real
        rhs_phi = (torch.fft.fftn(y - eta_y - forward_physics, dim=SPATIAL_DIMS)
                   + torch.fft.fftn(v - eta_v, dim=SPATIAL_DIMS))
        denom_phi = 2
        phi_h = torch.fft.ifftn(rhs_phi / denom_phi, dim=SPATIAL_DIMS).real

        total_field = forward_physics + phi_h
        eta_y = eta_y + total_field - y
        eta_u = eta_u + mask * chi - u
        eta_v = eta_v + phi_h - v

        return (chi, phi_h, y, u, v, eta_y, eta_u, eta_v)

    def forward(self, phi, mask, D, W=None, return_iterates=False, tol=None,
                max_iters=None, iterate_callback=None):
        """Reconstruye ``chi`` y ``phi_h`` mediante ADMM desenrollado.

        ``iterate_callback``, cuando se proporciona, se llama después de cada
        iteración como ``callback(iteration, chi, phi_h)``. Permite evaluar las
        reconstrucciones intermedias sin conservar todos los volúmenes en GPU
        con ``return_iterates=True``; este último conserva su comportamiento
        original para la supervisión profunda durante el entrenamiento.
        """

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
                    self.step_fn, chi, phi_h, y, u, v, eta_y, eta_u, eta_v,
                    phi, mask, W, D, k, tol_res, use_reentrant=False
                )
            else:
                (chi, phi_h, y, u, v, eta_y, eta_u, eta_v) = self.step_fn(
                    chi, phi_h, y, u, v, eta_y, eta_u, eta_v,
                    phi, mask, W, D, k, tol_res
                )

            if return_iterates:
                iterates.append((chi, phi_h))

            if iterate_callback is not None:
                iterate_callback(k + 1, chi, phi_h)

            if tol is not None and k > 0:
                rel = torch.norm(chi - chi_prev) / (torch.norm(chi) + self.eps)
                if rel < tol:
                    break

            data_resid = (chi_prev - chi) * mask
            tol_res = (data_resid.pow(2).sum(dim=SPATIAL_DIMS) / mask.sum(dim=SPATIAL_DIMS).clamp(min=1.0)).sqrt()

        if return_iterates:
            return iterates
        return chi, phi_h
    
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prox = ADMMUnrolledNet(ProximalNetwork(), ProximalNetwork()).to(device)
    im = torch.ones([5, 1, 11, 140, 140]).float().to(device)
    out = prox(im, im, im, return_iterates=True)
    print(out[0][0].shape)
# %%

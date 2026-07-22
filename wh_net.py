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

    def __init__(self, chinet, phinet, num_iters=5, eps=1e-6, mu1=0.1, mu2=0.1, mu3=0.1):
        super().__init__()
        self.chinet = chinet
        self.phinet = phinet
        self.num_iters = num_iters
        self.eps = eps
        self.mu1 = mu1
        self.mu2 = mu2
        self.mu3 = mu3

    def step_fn(self, chi, phi_h, z1, z2, z3, s1, s2, s3, phi, mask, W, D, t, tol_res):

        chi_fft = torch.fft.fftn(chi, dim=SPATIAL_DIMS)
        forward_physics = torch.fft.ifftn(D * chi_fft, dim=SPATIAL_DIMS).real
        total_field = forward_physics + phi_h

        data_resid = (total_field - phi) * mask
        resid_rms = (data_resid.pow(2).sum(dim=SPATIAL_DIMS) / mask.sum(dim=SPATIAL_DIMS).clamp(min=1.0)).sqrt()

        z1 = (W * phi +  self.mu1 * (total_field + s1)) / (W + self.mu1 + self.eps)

        z2 = self.chinet(chi + s2, t, tol_res)
        z3 = self.phinet(phi_h + s3, t, resid_rms)

        rhs_chi = (self.mu1*torch.conj(D) * torch.fft.fftn(z1 - s1 - phi_h, dim=SPATIAL_DIMS)
                   + self.mu2*torch.fft.fftn(z2 - s2, dim=SPATIAL_DIMS))
        denom_chi = self.mu1 * torch.abs(D) ** 2 + self.mu2 + self.eps
        chi_fft = rhs_chi / denom_chi
        chi = torch.fft.ifftn(chi_fft, dim=SPATIAL_DIMS).real * mask

        forward_physics = torch.fft.ifftn(D * chi_fft, dim=SPATIAL_DIMS).real * mask
        rhs_phi = (self.mu1*torch.fft.fftn(z1 - s1 - forward_physics, dim=SPATIAL_DIMS)
                   + self.mu3*torch.fft.fftn(z3 - s3, dim=SPATIAL_DIMS))
        denom_phi = self.mu1 + self.mu3
        phi_h = torch.fft.ifftn(rhs_phi / denom_phi, dim=SPATIAL_DIMS).real

        total_field = forward_physics + phi_h
        s1 = s1 + total_field - z1
        s2 = s2 + chi - z2
        s3 = s3 + phi_h - z3

        return (chi, phi_h, z1, z2, z3, s1, s2, s3)

    def forward(self, phi, mask, D, W=None, return_iterates=False, tol=None, max_iters=None):

        phi = phi.float()
        mask = mask.float()
        W = mask if W is None else W.float()**2

        chi = torch.zeros_like(phi)
        phi_h = torch.zeros_like(phi)
        z1 = (W * phi) / (W + self.mu1 + self.eps)
        z2 = torch.zeros_like(phi)
        z3 = torch.zeros_like(phi)
        s1 = torch.zeros_like(phi)
        s2 = torch.zeros_like(phi)
        s3 = torch.zeros_like(phi)

        tol_res = torch.zeros(phi.shape[:2], device=phi.device)

        n_iters = self.num_iters if max_iters is None else max_iters
        use_ckpt = torch.is_grad_enabled()
        iterates = []

        for k in range(n_iters):
            chi_prev = chi
            if use_ckpt:
                (chi, phi_h, z1, z2, z3, s1, s2, s3) = checkpoint(
                    self.step_fn, chi, phi_h, z1, z2, z3, s1, s2, s3,
                    phi, mask, W, D, k, tol_res, use_reentrant=False
                )
            else:
                (chi, phi_h, z1, z2, z3, s1, s2, s3) = self.step_fn(
                    chi, phi_h, z1, z2, z3, s1, s2, s3,
                    phi, mask, W, D, k, tol_res
                )

            if return_iterates:
                iterates.append((chi, phi_h))

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

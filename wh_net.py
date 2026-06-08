import torch
import torch.nn as nn

class ProximalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv3d(16, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.conv_blocks(x)

class SharedADMMStage(nn.Module):
    def __init__(self, net_chi, net_phi, step_size=0.1, rho_chi=1.0, rho_phi=1.0):
        super().__init__()
        self.step_size = step_size
        self.rho_chi = rho_chi
        self.rho_phi = rho_phi
        self.net_chi = net_chi
        self.net_phi = net_phi

    def data_consistency(self, chi, phi, z_chi, z_phi, u_chi, u_phi, phase, W, D):
        total_field = torch.fft.irfftn(D * torch.fft.rfftn(chi), s=chi.shape[-3:]) + phi
        grad_fidelity = (W ** 2) * torch.sin(total_field - phase)
        grad_chi = torch.fft.irfftn(D.conj() * torch.fft.rfftn(grad_fidelity), s=chi.shape[-3:]) + self.rho_chi * (chi - z_chi + u_chi)
        grad_phi = grad_fidelity + self.rho_phi * (phi - z_phi + u_phi)
        chi_next = chi - self.step_size * grad_chi
        phi_next = phi - self.step_size * grad_phi
        return chi_next, phi_next

    def forward(self, chi, phi, z_chi, z_phi, u_chi, u_phi, phase, W, D):
        chi_next, phi_next = self.data_consistency(chi, phi, z_chi, z_phi, u_chi, u_phi, phase, W, D)
        z_chi_next = self.net_chi(chi_next + u_chi)
        z_phi_next = self.net_phi(phi_next + u_phi)
        u_chi_next = u_chi + chi_next - z_chi_next
        u_phi_next = u_phi + phi_next - z_phi_next
        return chi_next, phi_next, z_chi_next, z_phi_next, u_chi_next, u_phi_next

class RecurrentHybridQSMModel(nn.Module):
    def __init__(self, pretrained_chi, pretrained_phi, train_stages=5):
        super().__init__()
        self.train_stages = train_stages
        self.stage = SharedADMMStage(pretrained_chi, pretrained_phi)

    def forward(self, phase, W, D, inference_stages=None):
        num_stages = inference_stages if inference_stages is not None else self.train_stages
        chi = torch.zeros_like(phase)
        phi = torch.zeros_like(phase)
        z_chi = torch.zeros_like(phase)
        z_phi = torch.zeros_like(phase)
        u_chi = torch.zeros_like(phase)
        u_phi = torch.zeros_like(phase)
        predictions = []
        for _ in range(num_stages):
            chi, phi, z_chi, z_phi, u_chi, u_phi = self.stage(chi, phi, z_chi, z_phi, u_chi, u_phi, phase, W, D)
            predictions.append((z_chi, z_phi))
        return predictions
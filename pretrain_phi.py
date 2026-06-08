#%%
import torch
import torch.optim as optim
from wh_net import ProximalNetwork
from utils import  imshow_3d
from dataset import QSMDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

def laplacian_3d(tensor):
    kernel = torch.tensor([[[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                            [[0.0, 1.0, 0.0], [1.0, -6.0, 1.0], [0.0, 1.0, 0.0]],
                            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]],
                          device=tensor.device)
    kernel = kernel.repeat(tensor.shape[1], 1, 1, 1, 1)
    return F.conv3d(tensor, kernel, padding=1)

def decoupled_physics_loss(phi_pred, phi_gt, mask, lam_phi=1, lam_harm=10):
    loss_phi = F.l1_loss(phi_pred*mask, phi_gt*mask)
    laplacian_phi = laplacian_3d(phi_pred)
    loss_harmonic = torch.sum((mask * laplacian_phi) ** 2) / (torch.sum(mask) + 1e-8)
    print(loss_phi.item()*lam_phi, loss_harmonic.item()*lam_harm)
    return lam_phi * loss_phi + lam_harm * loss_harmonic


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProximalNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    dataset = QSMDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for epoch in range(50):
        for phase, mask, W, D, chi_gt, phi_gt in dataloader:
            mask, phi_gt = mask.to(device), phi_gt.to(device)
            optimizer.zero_grad()
            noise = torch.randn_like(phi_gt) * 0.05
            phi_corrupted = phi_gt + noise
            phi_pred = model(phi_corrupted)
            loss = decoupled_physics_loss(phi_pred, phi_gt, mask)
            loss.backward()
            optimizer.step()
    imshow_3d(phi_gt.numpy()[0, 0], 'phi_gt', rango=(-0.1, 0.1), angles=(-90, -90, 90))
    imshow_3d(phi_corrupted.numpy()[0, 0], 'phi_corrupted', rango=(-0.1, 0.1), angles=(-90, -90, 90))
    imshow_3d(phi_pred.numpy()[0, 0], 'phi_pred', rango=(-0.1, 0.1), angles=(-90, -90, 90))
    # torch.save(model.state_dict(), "weights/pretrained_phi.pth")

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProximalNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    dataset = QSMDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for epoch in range(200):
        for phase, mask, W, D, chi_gt, phi_gt in dataloader:
            mask, phi_gt = mask.to(device), phi_gt.to(device)
            optimizer.zero_grad()
            local_phase = torch.real(torch.fft.ifftn(D * torch.fft.fftn(chi_gt), s=chi_gt.shape[-3:]))
            structural_noise = local_phase * mask * (0.1 + 0.9 * torch.rand(1).to(device))
            gaussian_noise = torch.randn_like(phi_gt) / torch.randint(25, 150,(1,))
            
            phi_corrupted = phi_gt + structural_noise + gaussian_noise
            phi_pred = model(phi_corrupted)
            loss = decoupled_physics_loss(phi_pred, phi_gt, mask)
            loss.backward()
            optimizer.step()
    imshow_3d(phi_gt.numpy()[0, 0], 'phi_gt', rango=(-0.1, 0.1), angles=(-90, -90, 90))
    imshow_3d(phi_corrupted.numpy()[0, 0], 'phi_corrupted', rango=(-0.1, 0.1), angles=(-90, -90, 90))
    imshow_3d(structural_noise.numpy()[0, 0], 'structural_noise', rango=(-0.1, 0.1), angles=(-90, -90, 90))
    imshow_3d(phi_pred.detach().numpy()[0, 0], 'phi_pred', rango=(-0.1, 0.1), angles=(-90, -90, 90))
    imshow_3d(np.abs(phi_corrupted.numpy()[0, 0]-phi_pred.detach().numpy()[0, 0]), 'diff', rango=(0, 0.1), angles=(-90, -90, 90))
    # torch.save(model.state_dict(), "weights/pretrained_phi.pth")
# %%

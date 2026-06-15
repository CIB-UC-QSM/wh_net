#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from wh_net import ProximalNetwork, ADMMUnrolledNet
from utils import imshow_3d
from dataset import QSMDataset
import os
import numpy as np

def calculate_nrmse(pred, target):
    rmse = torch.norm(pred.flatten() - target.flatten())
    denom = torch.norm(target.flatten())
    return rmse / denom

def spatial_gradient_3d(x):
    dx = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
    dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
    dz = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
    return dx, dy, dz

def gradient_loss(pred, target):
    px, py, pz = spatial_gradient_3d(pred)
    tx, ty, tz = spatial_gradient_3d(target)
    return F.l1_loss(px, tx) + F.l1_loss(py, ty) + F.l1_loss(pz, tz)

def weak_harmonic_loss(phi, mask):
    laplacian_kernel = torch.tensor([[[[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                                       [[0.0, 1.0, 0.0], [1.0, -6.0, 1.0], [0.0, 1.0, 0.0]],
                                       [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]]], device=phi.device)
    laplacian_phi = F.conv3d(phi, laplacian_kernel, padding=1)
    return torch.mean((mask * laplacian_phi) ** 2)

def hybrid_qsm_loss(chi_pred, chi_gt, phi_pred, phi_gt, mask, lam_chi=10.0, lam_phi=0.5, lam_grad=0.2, lam_wh=100):
    loss_chi = F.l1_loss(chi_pred*mask, chi_gt*mask)
    loss_phi = F.l1_loss(phi_pred*mask, phi_gt*mask)
    loss_grad = gradient_loss(chi_pred*mask, chi_gt*mask)
    loss_wh = weak_harmonic_loss(phi_pred, mask)
    return lam_chi * loss_chi + lam_phi * loss_phi + lam_grad * loss_grad + lam_wh * loss_wh

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    net_chi = ProximalNetwork().to(device)
    net_phi = ProximalNetwork().to(device)
    model = ADMMUnrolledNet(net_chi, net_phi, num_iters=5).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler("cuda:0")
    
    dataset = QSMDataset()
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    #%%
    epochs = 2
    model.train()
    
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for phase, mask, W, D, chi_gt, phi_gt in pbar:
            model.num_iters = np.clip(epoch-2, 3, 5)
            phase = phase.to(device)
            mask = mask.to(device)
            D = D.to(device)
            chi_gt = chi_gt.to(device)
            phi_gt = phi_gt.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast("cuda", dtype=torch.bfloat16):
                chi_pred, phi_pred = model(phase, mask, D)
                loss = hybrid_qsm_loss(chi_pred, chi_gt, phi_pred, phi_gt, mask)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            with torch.no_grad():
                nrmse_chi = calculate_nrmse(chi_pred*mask, chi_gt*mask)
                nrmse_phi = calculate_nrmse(phi_pred*mask, phi_gt*mask)
                
            pbar.set_postfix(
                loss=f"{loss.item():.4f}", 
                nrmse_chi=f"{nrmse_chi.item():.4f}", 
                nrmse_phi=f"{nrmse_phi.item():.4f}"
            )
        torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pth")

    model.eval()
    with torch.no_grad():
        chi_pred, phi_pred = model(phase, mask, D)
        final_nrmse_chi = calculate_nrmse(chi_pred*mask, chi_gt*mask)
        final_nrmse_phi = calculate_nrmse(phi_pred*mask, phi_gt*mask)
        print(f"Final NRMSE Chi: {final_nrmse_chi.item():.4f} | Final NRMSE Phi: {final_nrmse_phi.item():.4f}")
    
    imshow_3d(phase.detach().cpu().numpy()[0, 0], 'phase in', rango=(-0.1, 0.1), angles=(-90, -90, 90))
    imshow_3d(chi_pred.detach().cpu().numpy()[0, 0], 'chi_pred', rango=(-0.1, 0.1), angles=(-90, -90, 90))
    imshow_3d(phi_pred.detach().cpu().numpy()[0, 0], 'phi_pred', rango=(-0.1, 0.1), angles=(-90, -90, 90))
    imshow_3d(chi_gt.detach().cpu().numpy()[0, 0], 'chi_gt', rango=(-0.1, 0.1), angles=(-90, -90, 90))
    imshow_3d(phi_gt.detach().cpu().numpy()[0, 0], 'phi_gt', rango=(-0.1, 0.1), angles=(-90, -90, 90))
# %%

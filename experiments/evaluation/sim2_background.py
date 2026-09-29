#%%
1+1
#%%
import matplotlib.pyplot as plt
from whnet.model import ProximalNetwork, ADMMUnrolledNet
from whnet.physics import continuous_dipole_kernel, imshow_3d, rmse
from whnet.paths import (
    CHECKPOINTS_DIR,
    EVALUATION_DATA_DIR,
    EVALUATION_VOLUMES_DIR,
    EXPORTS_DIR,
)
from scipy.io import loadmat
import torch.nn.functional as F
import torch
from torch.amp import autocast
import numpy as np
import nibabel as nib


# %%
chi_gt = nib.load(EVALUATION_VOLUMES_DIR / "Sim2ChiGT.nii.gz").get_fdata()
mask = nib.load(EVALUATION_VOLUMES_DIR / "mask4.nii").get_fdata()
phase = nib.load(EVALUATION_VOLUMES_DIR / "unwrapped_seguetotalphase.nii").get_fdata()*mask / (7 * 8 * 2 * np.pi * 42.58 * 1e-3)

imshow_3d(chi_gt, f'chi_gt', rango=(-0.1, 0.1), angles=(-90, -90, 90))
imshow_3d(phase, f'phase', rango=(-0.1, 0.1), angles=(-90, -90, 90))
imshow_3d(mask, f'mask', rango=(-0.1, 0.1), angles=(-90, -90, 90))

#%%
def pad_to_sqr_shape(volume, zero_pad):
    x, y, z = volume.shape
    max_size = (max_o:=np.max([x, y, z])) + 2*zero_pad
    new_volume = np.zeros(3 * (max_size,))
    dx = (max_o - x)//2 + zero_pad
    dy = (max_o - y)//2 + zero_pad
    dz = (max_o - z)//2 + zero_pad
    new_volume[dx:x+dx, dy:y+dy, dz:z+dz] = volume
    return new_volume

#%%
factor = 0.325
pad = 20

coords = np.argwhere(mask)
z_min, y_min, x_min = coords.min(axis=0)
z_max, y_max, x_max = coords.max(axis=0) + 1

w = pad_to_sqr_shape(mask[z_min:z_max, y_min:y_max, x_min:x_max], pad)
mask = pad_to_sqr_shape(mask[z_min:z_max, y_min:y_max, x_min:x_max], pad)
phase = pad_to_sqr_shape(phase[z_min:z_max, y_min:y_max, x_min:x_max], pad)
chi_gt = pad_to_sqr_shape(chi_gt[z_min:z_max, y_min:y_max, x_min:x_max], pad)
imshow_3d(chi_gt, 'chi_gt', rango=(-0.1, 0.1), angles=(-90, -90, 90))
imshow_3d(w, 'w', rango=(0, 1), angles=(-90, -90, 90))


#%%

SPATIAL = (-3, -2, -1)

phase = torch.from_numpy(phase).unsqueeze(0).float()/factor
mask = torch.from_numpy(mask).unsqueeze(0).float()

# W = mask.clone()
W = torch.from_numpy(w).unsqueeze(0).float()
      
D = torch.from_numpy(continuous_dipole_kernel(phase.shape[-3:])).unsqueeze(0).float()


phase = phase.unsqueeze(0)
mask = mask.unsqueeze(0)
D = D.unsqueeze(0)
W = W.unsqueeze(0)

imshow_3d(phase.squeeze().numpy(), f'Local', rango=(-0.1, 0.1), angles=(-90, -90, 90))


#%%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


net_chi = ProximalNetwork().to(device)
net_phi = ProximalNetwork().to(device)
model = ADMMUnrolledNet(net_chi, net_phi, num_iters=100).to(device)

model.load_state_dict(torch.load(CHECKPOINTS_DIR / "checkpoints_scratch6" / "model_best.pth", map_location=device), strict=True)

@torch.no_grad()
def evaluate(model, phase_in, mask, D, W):
    model.eval()

    with autocast("cuda", dtype=torch.bfloat16):
        preds = model(phase_in.to(device), mask.to(device), D.to(device), W.to(device), return_iterates=True)
    chi_pred = [x[0].cpu() for x in preds]
    phi_pred = [x[1].cpu() for x in preds]
    return chi_pred, phi_pred

chi_preds, phi_preds = evaluate(model, phase, mask, D, W)
print(chi_preds[-1].shape)


# %%
chi_pred = chi_preds[-1].squeeze().numpy() * factor
phi_pred = phi_preds[-1].squeeze().numpy()
mask = mask.squeeze().numpy()
imshow_3d(chi_pred, f'chi_cosmos rmse={rmse(chi_pred, chi_gt).item():.2f}', rango=(-0.1, 0.1), angles=(-90, -90, 90))
imshow_3d(phi_pred, 'phi_pred', rango=(-0.1, 0.1), angles=(-90, -90, 90))


# %%
errores = []
for i, chi in enumerate(chi_preds):
    chi_pred = chi.squeeze().numpy() * factor
    print(i, val:=rmse(chi_pred, chi_gt))
    errores.append(val)

chi_pred = chi_preds[np.argmin(errores)].squeeze().numpy()* factor
imshow_3d(chi_pred, f'rmse={rmse(chi_pred, chi_gt).item():.2f}', rango=(-0.1, 0.1), angles=(-90, -90, 90))

# %%
imshow_3d(chi_pred, f'rmse={rmse(chi_pred, chi_gt).item():.2f}', rango=(-0.1, 0.1), angles=(-90, -90, 90))
imshow_3d(chi_gt, 'Ground Truth', rango=(-0.1, 0.1), angles=(-90, -90, 90))
imshow_3d(chi_pred-chi_gt, 'chi_pred-chi_gt', rango=(-0.1, 0.1), angles=(-90, -90, 90))

#%%
grid_search = []
for factor in np.linspace(0.1, 1, 9):
    for pad in [10, 20, 30]:

        
        mask = loadmat(EVALUATION_DATA_DIR / "mask_final.mat")['mask_final']
        phase = loadmat(EVALUATION_DATA_DIR / "Sim2.mat")['phase']
        chi_gt = nib.load(EVALUATION_VOLUMES_DIR / "Sim2ChiGT.nii.gz").get_fdata()

        coords = np.argwhere(mask)
        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0) + 1

        mask = pad_to_sqr_shape(mask[z_min:z_max, y_min:y_max, x_min:x_max], pad)
        phase = pad_to_sqr_shape(phase[z_min:z_max, y_min:y_max, x_min:x_max], pad)
        chi_gt = pad_to_sqr_shape(chi_gt[z_min:z_max, y_min:y_max, x_min:x_max], pad)

        phase = torch.from_numpy(phase).unsqueeze(0).float()/factor
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        W = mask.clone()
            
        D = torch.from_numpy(continuous_dipole_kernel(phase.shape[-3:])).unsqueeze(0).float()


        phase = phase.unsqueeze(0)
        mask = mask.unsqueeze(0)
        D = D.unsqueeze(0)
        W = W.unsqueeze(0)

        chi_preds, phi_preds = evaluate(model, phase, mask, D, W)

        errores = []
        for i, chi in enumerate(chi_preds):
            chi_pred = chi.squeeze().numpy() * factor
            val=rmse(chi_pred, chi_gt)
            errores.append(val)

        grid_search.append((np.min(errores), factor, pad))

errs = [x[0] for x in grid_search]
idx = np.argmin(errs)
print(grid_search[idx])

#%%



# %%

def save_nii(name, img):
    TE = 25e-3 
    B0 = 7     
    gyro = 2 * np.pi * 42.58e6 
    phs_scale = TE * gyro * B0 * 1e-6 
    true_phase = img.squeeze().numpy() * phs_scale
    phase_wrapped = np.angle(np.exp(1j * true_phase))
    affine = np.eye(4)

    nifti_img = nib.Nifti1Image(phase_wrapped, affine)
    nib.save(nifti_img, name)
save_nii(EXPORTS_DIR / "input_phase_sim2.nii", phase*factor)

# %%


affine = np.eye(4)

nifti_img = nib.Nifti1Image(mask, affine)
nib.save(nifti_img, EXPORTS_DIR / "input_mask_sim2.nii")
# %%

pred_iqsm = nib.load(EVALUATION_VOLUMES_DIR / "iQSM.nii.gz").get_fdata()
imshow_3d(pred_iqsm*mask, f'pred_iqsm rmse={rmse(pred_iqsm*mask, chi_gt*mask).item():.2f}', rango=(-0.1, 0.1), angles=(-90, -90, -90))
imshow_3d(chi_pred*mask, f'pred_whnet rmse={rmse(chi_pred*mask, chi_gt*mask).item():.2f}', rango=(-0.1, 0.1), angles=(-90, -90, -90))

# %%
imshow_3d(pred_iqsm*mask-chi_gt, f'pred_iqsm rmse={rmse(pred_iqsm*mask, chi_gt*mask).item():.2f}', rango=(-0.1, 0.1), angles=(-90, -90, -90))
imshow_3d(chi_pred*mask-chi_gt, f'pred_whnet rmse={rmse(chi_pred*mask, chi_gt*mask).item():.2f}', rango=(-0.1, 0.1), angles=(-90, -90, -90))

# %%

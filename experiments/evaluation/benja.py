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
#%%
factor = 0.264

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

msk = loadmat(EVALUATION_DATA_DIR / "data_wh.mat")['mask']
phase = loadmat(EVALUATION_DATA_DIR / "data_wh.mat")['phase_use']
w = loadmat(EVALUATION_DATA_DIR / "data_wh.mat")['magn_use']

coords = np.argwhere(msk)
z_min, y_min, x_min = coords.min(axis=0)
z_max, y_max, x_max = coords.max(axis=0) + 1


imshow_3d(phase, 'chi_gt', rango=(-0.1, 0.1), angles=(-90, -90, 90))
imshow_3d(w, 'w', rango=(0, 1), angles=(-90, -90, 90))


#%%

SPATIAL = (-3, -2, -1)

phase = torch.from_numpy(phase).unsqueeze(0).float()/factor
msk = torch.from_numpy(msk).unsqueeze(0).float()

W = msk.clone()
# W = torch.from_numpy(w).unsqueeze(0).float()
      
D = torch.from_numpy(continuous_dipole_kernel(phase.shape[-3:])).unsqueeze(0).float()


phase = phase.unsqueeze(0)
msk = msk.unsqueeze(0)
D = D.unsqueeze(0)
W = W.unsqueeze(0)

imshow_3d(phase.squeeze().numpy(), f'Local', rango=(-0.1, 0.1), angles=(-90, -90, 90))


#%%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


net_chi = ProximalNetwork().to(device)
net_phi = ProximalNetwork().to(device)
model = ADMMUnrolledNet(net_chi, net_phi, num_iters=50).to(device)

model.load_state_dict(torch.load(CHECKPOINTS_DIR / "checkpoints_scratch7" / "model_best.pth", map_location=device), strict=True)

@torch.no_grad()
def evaluate(model, phase_in, mask, D, W):
    model.eval()

    with autocast("cuda", dtype=torch.bfloat16):
        preds = model(phase_in.to(device), mask.to(device), D.to(device), W.to(device), return_iterates=True)
    chi_pred = [x[0].cpu() for x in preds]
    phi_pred = [x[1].cpu() for x in preds]
    return chi_pred, phi_pred

chi_preds, phi_preds = evaluate(model, phase, msk, D, W)
print(chi_preds[-1].shape)


# %%
chi_pred = chi_preds[-1].squeeze().numpy() * factor
phi_pred = phi_preds[-1].squeeze().numpy()
mask = msk.squeeze().numpy()
imshow_3d(chi_pred, f'chi_pred', rango=(-0.1, 0.1), angles=(-90, -90, 90))
imshow_3d(phi_pred, 'phi_pred', rango=(-0.1, 0.1), angles=(-90, -90, 90))



#%%



#%%






























# %%

def save_nii(name, img):
    TE = 25e-3      # Tiempo de eco en segundos (25 ms)
    B0 = 7     # Campo magnético en Teslas
    gyro = 2 * np.pi * 42.58e6 
    phs_scale = TE * gyro * B0 * 1e-6 
    true_phase = img.squeeze().numpy() * phs_scale
    phase_wrapped = np.angle(np.exp(1j * true_phase))
    affine = np.eye(4)

    nifti_img = nib.Nifti1Image(phase_wrapped, affine)
    nib.save(nifti_img, name)
save_nii(EXPORTS_DIR / "input_phase_benja.nii", phase*factor)

# %%

msk_s = msk.squeeze().numpy()

affine = np.eye(4)

nifti_img = nib.Nifti1Image(msk_s, affine)
nib.save(nifti_img, EXPORTS_DIR / "input_mask_benja.nii")

# %%

pred_iqsm = nib.load(EVALUATION_VOLUMES_DIR / "iQSM.nii.gz").get_fdata()
chi_pred = nib.load(EVALUATION_VOLUMES_DIR / "whnet.nii.gz").get_fdata()

imshow_3d(pred_iqsm, f'pred_iqsm', rango=(-0.1, 0.1), angles=(-90, -90, -90))
imshow_3d(chi_pred, f'pred_whnet', rango=(-0.1, 0.1), angles=(-90, -90, -90))



# %%

affine = np.eye(4)

nifti_img = nib.Nifti1Image(chi_pred, affine)
nib.save(nifti_img, EXPORTS_DIR / "whnet.nii.gz")
# %%

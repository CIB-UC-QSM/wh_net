#%%
import matplotlib.pyplot as plt
from wh_net import ProximalNetwork, ADMMUnrolledNet
from utils import continuous_dipole_kernel, imshow_3d, rmse
from scipy.io import loadmat
import torch.nn.functional as F
import torch
from torch.amp import autocast
import numpy as np
import nibabel as nib
#%%
factor = 0.325
pad = 20

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

msk = loadmat('mask_final.mat')['mask_final']
phase = loadmat('Sim2.mat')['phase']
w = loadmat('sim2_w.mat')['w']
chi_gt = nib.load('Sim2ChiGT.nii.gz').get_fdata()

coords = np.argwhere(msk)
z_min, y_min, x_min = coords.min(axis=0)
z_max, y_max, x_max = coords.max(axis=0) + 1

msk = pad_to_sqr_shape(msk[z_min:z_max, y_min:y_max, x_min:x_max], pad)
phase = pad_to_sqr_shape(phase[z_min:z_max, y_min:y_max, x_min:x_max], pad)
chi_gt = pad_to_sqr_shape(chi_gt[z_min:z_max, y_min:y_max, x_min:x_max], pad)
w = pad_to_sqr_shape(w[z_min:z_max, y_min:y_max, x_min:x_max], pad)
imshow_3d(chi_gt, 'chi_gt', rango=(-0.1, 0.1), angles=(-90, -90, 90))
imshow_3d(w, 'w', rango=(0, 1), angles=(-90, -90, 90))


#%%

SPATIAL = (-3, -2, -1)

phase = torch.from_numpy(phase).unsqueeze(0).float()/factor
msk = torch.from_numpy(msk).unsqueeze(0).float()

# W = msk.clone()
W = torch.from_numpy(w).unsqueeze(0).float()
      
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
model = ADMMUnrolledNet(net_chi, net_phi, num_iters=100).to(device)

model.load_state_dict(torch.load("checkpoints_scratch5/model_best.pth" , map_location=device), strict=True)

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

        
        msk = loadmat('mask_final.mat')['mask_final']
        phase = loadmat('Sim2.mat')['phase']
        chi_gt = nib.load('Sim2ChiGT.nii.gz').get_fdata()

        coords = np.argwhere(msk)
        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0) + 1

        msk = pad_to_sqr_shape(msk[z_min:z_max, y_min:y_max, x_min:x_max], pad)
        phase = pad_to_sqr_shape(phase[z_min:z_max, y_min:y_max, x_min:x_max], pad)
        chi_gt = pad_to_sqr_shape(chi_gt[z_min:z_max, y_min:y_max, x_min:x_max], pad)

        phase = torch.from_numpy(phase).unsqueeze(0).float()/factor
        msk = torch.from_numpy(msk).unsqueeze(0).float()

        W = msk.clone()
            
        D = torch.from_numpy(continuous_dipole_kernel(phase.shape[-3:])).unsqueeze(0).float()


        phase = phase.unsqueeze(0)
        msk = msk.unsqueeze(0)
        D = D.unsqueeze(0)
        W = W.unsqueeze(0)

        chi_preds, phi_preds = evaluate(model, phase, msk, D, W)

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
    TE = 25e-3      # Tiempo de eco en segundos (25 ms)
    B0 = 7     # Campo magnético en Teslas
    gyro = 2 * np.pi * 42.58e6 
    phs_scale = TE * gyro * B0 * 1e-6 
    true_phase = img.squeeze().numpy() * phs_scale
    phase_wrapped = np.angle(np.exp(1j * true_phase))
    affine = np.eye(4)

    nifti_img = nib.Nifti1Image(phase_wrapped, affine)
    nib.save(nifti_img, name)
save_nii('input_phase_sim2.nii', phase*factor)

# %%

msk_s = msk.squeeze().numpy()

affine = np.eye(4)

nifti_img = nib.Nifti1Image(msk_s, affine)
nib.save(nifti_img, 'input_mask_sim2.nii')

# %%

pred_iqsm = nib.load('iQSM.nii.gz').get_fdata()
imshow_3d(pred_iqsm, f'pred_iqsm rmse={rmse(pred_iqsm, chi_gt).item():.2f}', rango=(-0.1, 0.1), angles=(-90, -90, -90))
imshow_3d(chi_pred, f'pred_whnet rmse={rmse(chi_pred, chi_gt).item():.2f}', rango=(-0.1, 0.1), angles=(-90, -90, -90))

# %%
imshow_3d(pred_iqsm-chi_gt, 'pred_iqsm-chi_gt', rango=(-0.1, 0.1), angles=(-90, -90, -90))
imshow_3d(chi_pred-chi_gt, 'chi_pred-chi_gt', rango=(-0.1, 0.1), angles=(-90, -90, -90))
imshow_3d(chi_pred-pred_iqsm, 'chi_pred-pred_iqsm', rango=(-0.1, 0.1), angles=(-90, -90, -90))

# %%




#%%

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

net_chi = ProximalNetwork().to(device)
net_phi = ProximalNetwork().to(device)
model = ADMMUnrolledNet(net_chi, net_phi, num_iters=50, mask_chi=True).to(device)

model.load_state_dict(torch.load("checkpoints_scratch3/model_best.pth" , map_location=device), strict=True)

# grid_search = []
for snr in np.linspace(10, 100, 7):
    for back in range(1, 6):

        chi = torch.from_numpy(pad_to_sqr_shape(chi_gt:=loadmat('chi_cosmos.mat')['chi_cosmos'], pad)).unsqueeze(0).float() /factor
        msk = torch.from_numpy(pad_to_sqr_shape(loadmat('msk.mat')['msk'], pad)).unsqueeze(0).float()
        W = torch.from_numpy(pad_to_sqr_shape(loadmat('msk.mat')['msk'], pad)).unsqueeze(0).float()
        D = torch.from_numpy(continuous_dipole_kernel(chi.shape[-3:])).unsqueeze(0).float()
        chi_k = torch.fft.fftn(chi * msk, dim=SPATIAL)
        local = torch.real(torch.fft.ifftn(D * chi_k, dim=SPATIAL)) * msk

        msk_dil = F.max_pool3d(msk.unsqueeze(0), kernel_size=7, stride=1, padding=3).squeeze(0)
        ext_k = torch.fft.fftn((1.0 - msk_dil) * back, dim=SPATIAL)
        phi = torch.real(torch.fft.ifftn(D * ext_k, dim=SPATIAL)) * msk

        inside = msk > 0
        if inside.sum() > 1:
            field_std = torch.std(local[inside])
        else:
            field_std = torch.tensor(0.0)
        noise_std = field_std / snr
        phase = msk * (local + torch.randn_like(local) * noise_std + phi)

        phase = phase.unsqueeze(0)
        msk = msk.unsqueeze(0)
        D = D.unsqueeze(0)
        W = W.unsqueeze(0)


        # chi_preds, phi_preds = evaluate(model, phase, msk, D, W)

        # errores = []
        # for i, chi in enumerate(chi_preds):
        #     chi_pred = chi.squeeze().numpy()[pad:160+pad, pad:160+pad, pad:160+pad]* factor
        #     val=rmse(chi_pred, chi_gt)
        #     errores.append(val)

        # grid_search.append((np.min(errores), snr, back))

        save_nii(f'./inputs/phase_{int(snr)}_{back}.nii', phase*factor)
#%%
errs = [x[0] for x in grid_search]
idx = np.argmin(errs)
print(grid_search[idx])

# %%


snr_vals  = np.linspace(10, 100, 7)   # eje X
back_vals = np.arange(1, 6)           # eje Y

# Matriz: filas = back (Y), columnas = snr (X)
err = np.full((len(back_vals), len(snr_vals)), np.nan)

col = {round(float(s), 6): j for j, s in enumerate(snr_vals)}
row = {int(b): i for i, b in enumerate(back_vals)}

for e, s, b in grid_search:
    err[row[int(b)], col[round(float(s), 6)]] = e

# --- Heatmap ---
fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(err, origin='lower', aspect='auto', cmap='viridis')

ax.set_xticks(range(len(snr_vals)), [f'{s:.0f}' for s in snr_vals])
ax.set_yticks(range(len(back_vals)), [f'{b:d}'  for b in back_vals])
ax.set_xlabel('SNR')
ax.set_ylabel('background susceptibility (ppm)')
ax.set_title('RMSE WH-NET')

# Anotaciones con 1 decimal (color según fondo para legibilidad)
mid = (np.nanmin(err) + np.nanmax(err)) / 2
for i in range(err.shape[0]):
    for j in range(err.shape[1]):
        if not np.isnan(err[i, j]):
            c = 'black' if err[i, j] > mid else 'white'
            ax.text(j, i, f'{err[i, j]:.1f}', ha='center', va='center', color=c)

fig.colorbar(im, ax=ax, label='RMSE')
plt.tight_layout()
plt.show()
# %%

grid_search_iqsm = []
for snr in np.linspace(10, 100, 7):
    for back in range(1, 6):
        pred_iqsm = nib.load(f'iQSM_{int(snr)}_{back}.nii.gz').get_fdata()
        val=rmse(pred_iqsm, chi_gt, mask==1)
        grid_search_iqsm.append((val, snr, back))

# %%
err = np.full((len(back_vals), len(snr_vals)), np.nan)

col = {round(float(s), 6): j for j, s in enumerate(snr_vals)}
row = {int(b): i for i, b in enumerate(back_vals)}

for e, s, b in grid_search_iqsm:
    err[row[int(b)], col[round(float(s), 6)]] = e

# --- Heatmap ---
fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(err, origin='lower', aspect='auto', cmap='viridis')

ax.set_xticks(range(len(snr_vals)), [f'{s:.0f}' for s in snr_vals])
ax.set_yticks(range(len(back_vals)), [f'{b:d}'  for b in back_vals])
ax.set_xlabel('SNR')
ax.set_ylabel('back')
ax.set_title('RMSE iQSM')

# Anotaciones con 1 decimal (color según fondo para legibilidad)
mid = (np.nanmin(err) + np.nanmax(err)) / 2
for i in range(err.shape[0]):
    for j in range(err.shape[1]):
        if not np.isnan(err[i, j]):
            c = 'black' if err[i, j] > mid else 'white'
            ax.text(j, i, f'{err[i, j]:.1f}', ha='center', va='center', color=c)

fig.colorbar(im, ax=ax, label='RMSE')
plt.tight_layout()
plt.show()

# %%

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

snr_vals  = np.linspace(10, 100, 7)   # X axis
back_vals = np.arange(1, 6)           # Y axis

def to_matrix(gs):
    """grid_search [(err, snr, back), ...] -> matrix (back x snr)."""
    M = np.full((len(back_vals), len(snr_vals)), np.nan)
    col = {round(float(s), 6): j for j, s in enumerate(snr_vals)}
    row = {int(b): i for i, b in enumerate(back_vals)}
    for e, s, b in gs:
        M[row[int(b)], col[round(float(s), 6)]] = e
    return M

def annotate(ax, M, norm, cmap):
    """Write each value with 1 decimal; text color chosen by background luminance."""
    cmap = plt.get_cmap(cmap)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                continue
            r, g, b, _ = cmap(norm(v))
            txt = 'black' if (0.299*r + 0.587*g + 0.114*b) > 0.5 else 'white'
            ax.text(j, i, f'{v:.1f}', ha='center', va='center', color=txt, fontsize=8)

E1   = to_matrix(grid_search)        # ADMM-NET
E2   = to_matrix(grid_search_iqsm)   # iQSM
diff = E1 - E2                       # RMSE(ADMM-NET) - RMSE(iQSM)
labels = ('ADMM-NET', 'iQSM')

# Shared scale so both methods are directly comparable
vmin, vmax = np.nanmin([E1, E2]), np.nanmax([E1, E2])
norm  = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
cmap  = 'viridis'

# Symmetric scale centered at 0 for the difference
dmax  = max(np.nanmax(np.abs(diff)), 1e-9)
normd = mpl.colors.TwoSlopeNorm(vmin=-dmax, vcenter=0, vmax=dmax)
cmapd = 'coolwarm'

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

for ax, M, title in zip(axes[:2], (E1, E2), labels):
    im = ax.imshow(M, origin='lower', aspect='auto', cmap=cmap, norm=norm)
    annotate(ax, M, norm, cmap)
    ax.set_title(f'RMSE — {title}')

imd = axes[2].imshow(diff, origin='lower', aspect='auto', cmap=cmapd, norm=normd)
annotate(axes[2], diff, normd, cmapd)
axes[2].set_title(f'Difference: {labels[0]} − {labels[1]}')

for ax in axes:
    ax.set_xticks(range(len(snr_vals)), [f'{s:.0f}' for s in snr_vals])
    ax.set_yticks(range(len(back_vals)), [f'{b:d}'  for b in back_vals])
    ax.set_xlabel('SNR')
axes[0].set_ylabel('background susceptibility (ppm)')

fig.colorbar(im,  ax=axes[:2], label='RMSE', shrink=0.85)
fig.colorbar(imd, ax=axes[2],  label='ΔRMSE  (<0 ⇒ ADMM-NET better)', shrink=0.85)

plt.show()
# %%

#%%
import torch
from torch.utils.data import Dataset, DataLoader
from utils import continuous_dipole_kernel, imshow_3d
import torch.nn.functional as F
import os

SPATIAL = (-3, -2, -1)  # FFT solo sobre dimensiones espaciales


class QSMDataset(Dataset):
    def __init__(self, data_dir='train_data'):
        self.chi_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if 'image' in f])
        self.msk_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if 'mask' in f])
        assert len(self.chi_files) == len(self.msk_files), \
            f"Numero de volumenes 'image' ({len(self.chi_files)}) != 'mask' ({len(self.msk_files)})"

        # IMPORTANTE: la convencion de continuous_dipole_kernel debe coincidir con
        # torch.fft (DC en la esquina, indice [0,0,0]). Si el kernel viene CENTRADO,
        # usar:  D = torch.fft.ifftshift(D, dim=(-3,-2,-1))
        # El bloque __main__ incluye un test de harmonicidad para verificarlo.
        self.D = torch.from_numpy(continuous_dipole_kernel((160, 160, 160))).unsqueeze(0).float()

    def __len__(self):
        return len(self.chi_files)

    def __getitem__(self, idx):
        chi = torch.load(self.chi_files[idx]).float()
        msk = torch.load(self.msk_files[idx]).float()          # se guardo como long -> a float

        # Peso de fidelidad. Actualmente es la mascara binaria: NO aporta ponderacion
        # por SNR/magnitud (a diferencia de W=magnitud en WH_wTV). Para fidelidad
        # espacialmente variable habria que simular/cargar una magnitud aqui.
        W = msk.clone()

        # --- Campo local de tejido: D * (chi dentro del cerebro) ---
        chi_k = torch.fft.fftn(chi * msk, dim=SPATIAL)
        local = torch.real(torch.fft.ifftn(self.D * chi_k, dim=SPATIAL)) * msk

        # --- Campo de fondo (armonico dentro de la mascara): dipolo de fuentes externas ---
        # Se dilata la mascara 3 voxeles para dejar un margen sin fuentes, de modo que
        # el campo resultante sea armonico (Laplace=0) dentro del cerebro.
        msk_dil = F.max_pool3d(msk.unsqueeze(0), kernel_size=7, stride=1, padding=3).squeeze(0)
        scale = torch.empty(1).uniform_(1, 2).item()
        ext_k = torch.fft.fftn((1.0 - msk_dil) * scale, dim=SPATIAL)
        phi = torch.real(torch.fft.ifftn(self.D * ext_k, dim=SPATIAL)) * msk

        # --- Ruido (relativo al campo local), referido al modelo phase = local + phi + ruido ---
        inside = msk > 0
        if inside.sum() > 1:
            field_std = torch.std(local[inside])
        else:
            field_std = torch.tensor(0.0)
        snr = torch.empty(1).uniform_(90, 100).item()
        noise_std = field_std / snr
        phase = msk * (local + torch.randn_like(local) * noise_std + phi)

        return phase, msk, W, self.D, chi, phi


if __name__ == '__main__':
    dataset = QSMDataset()

    # --- Auto-diagnostico de convencion del kernel / harmonicidad del fondo ---
    # Si el fondo phi es realmente armonico, |lap(phi)|/|phi| en el interior ~ 0.
    # Un valor grande (> ~0.3) sugiere que el kernel necesita ifftshift.
    phase, msk, W, D, chi, phi = dataset[0]
    lap_k = torch.zeros(1, 1, 3, 3, 3)
    lap_k[0, 0, 1, 1, 1] = -6
    for d in [(0, 1, 1), (2, 1, 1), (1, 0, 1), (1, 2, 1), (1, 1, 0), (1, 1, 2)]:
        lap_k[0, 0, d[0], d[1], d[2]] = 1
    interior = (F.max_pool3d((1 - msk).unsqueeze(0), 9, 1, 4).squeeze(0) < 0.5)
    lap = F.conv3d(phi.unsqueeze(0), lap_k, padding=1).squeeze(0)
    if interior.sum() > 0:
        rel = (lap[interior].abs().mean() / (phi[interior].abs().mean() + 1e-9)).item()
        print(f"[diagnostico] |lap(phi)|/|phi| interior = {rel:.3e}  "
              f"({'OK, armonico' if rel < 0.3 else 'ALTO -> revisar convencion del kernel (ifftshift?)'})")
    print(f"[escala] std(phase_total)={torch.std(phase[msk>0]).item():.4f} | "
          f"std(phi_fondo)={torch.std(phi[msk>0]).item():.4f}  (verificar que el fondo no domine excesivamente)")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for phase, mask, W, D, chi_gt, phi_gt in dataloader:
        imshow_3d(phase.numpy()[0, 0], 'phase', rango=(-0.1, 0.1), angles=(-90, -90, 90))
        imshow_3d(mask.numpy()[0, 0], 'mask', rango=(0, 1), angles=(-90, -90, 90))
        imshow_3d(W.numpy()[0, 0], 'W', rango=(0, 1), angles=(-90, -90, 90))
        imshow_3d(chi_gt.numpy()[0, 0], 'chi_gt', rango=(-0.1, 0.1), angles=(-90, -90, 90))
        imshow_3d(phi_gt.numpy()[0, 0], 'phi_gt', rango=(-0.1, 0.1), angles=(-90, -90, 90))
        break
# %%
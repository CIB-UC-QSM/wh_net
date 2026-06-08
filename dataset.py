#%%
import torch
from torch.utils.data import Dataset, DataLoader
from utils import continuous_dipole_kernel, forward_simulation, imshow_3d
from scipy.io import loadmat
import numpy as np

class QSMDataset(Dataset):
    def __init__(self):
        self.chi = loadmat('chi_cosmos.mat')['chi_cosmos']
        self.msk = loadmat('msk.mat')['msk']
        self.D = continuous_dipole_kernel(self.chi.shape)
        self.W = loadmat('magn.mat')['magn']
        self.phase = np.real(np.fft.ifftn(self.D * np.fft.fftn(self.chi*self.msk))) * self.msk
        self.phi = np.real(np.fft.ifftn(self.D * np.fft.fftn(~self.msk))) * self.msk

        self.chi = torch.from_numpy(self.chi).unsqueeze(0).float()
        self.msk = torch.from_numpy(self.msk).unsqueeze(0).float()
        self.D = torch.from_numpy(self.D).unsqueeze(0).float()
        self.W = torch.from_numpy(self.W).unsqueeze(0).float()
        self.phase = torch.from_numpy(self.phase).unsqueeze(0).float()
        self.phi = torch.from_numpy(self.phi).unsqueeze(0).float()
        

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.phase, self.msk, self.W, self.D, self.chi, self.phi
        

if __name__ == '__main__':
    dataset = QSMDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for phase, mask, W, D, chi_gt, phi_gt in dataloader:
        imshow_3d(phase.numpy()[0, 0], 'phase', rango=(-0.1, 0.1), angles=(-90, -90, 90))
        imshow_3d(mask.numpy()[0, 0], 'mask', rango=(0, 1), angles=(-90, -90, 90))
        imshow_3d(W.numpy()[0, 0], 'W', rango=(0, 1), angles=(-90, -90, 90))
        imshow_3d(chi_gt.numpy()[0, 0], 'chi_gt', rango=(-0.1, 0.1), angles=(-90, -90, 90))
        imshow_3d(phi_gt.numpy()[0, 0], 'phi_gt', rango=(-0.1, 0.1), angles=(-90, -90, 90))
# %%

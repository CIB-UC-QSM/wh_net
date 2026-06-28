#%%
from utils import imshow_3d
import os
import nibabel as nii
from scipy.ndimage import binary_dilation, binary_erosion, zoom
import torch
import numpy as np
from tqdm import tqdm

def process_3d_volume(image):
    mask = image != 0
    
    if np.random.rand() > 0.5:
        mask = binary_dilation(mask)
        mask = binary_erosion(mask)
    
    coords = np.argwhere(mask)
    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0) + 1
    
    cropped_image = image[z_min:z_max, y_min:y_max, x_min:x_max]
    cropped_mask = mask[z_min:z_max, y_min:y_max, x_min:x_max]

    target_size = np.random.randint(80, 160, 3)
    
    img_zoom = tuple(target_size[i] / s for i, s in enumerate(cropped_image.shape))
    resized_image = zoom(cropped_image, img_zoom, order=3)
    
    mask_zoom = tuple(target_size[i] / s for i, s in enumerate(cropped_mask.shape))
    resized_mask = zoom(cropped_mask, mask_zoom, order=0)
    
    delta_a = 160 - target_size[0]
    delta_b = 160 - target_size[1]
    delta_c = 160 - target_size[2]
    n = np.random.randint(1, 5, 3)
    pad_width = ((delta_a//n[0], delta_a-delta_a//n[0]), (delta_b//n[1], delta_b-delta_b//n[1]), (delta_c//n[2], delta_c-delta_c//n[2]))
    final_image = np.pad(resized_image, pad_width, mode='constant', constant_values=0)
    final_mask = np.pad(resized_mask, pad_width, mode='constant', constant_values=0)
    
    return final_image, final_mask

def save_volumes_as_tensors(image: np.ndarray, mask: np.ndarray, image_path: str, mask_path: str):
    image_tensor = torch.from_numpy(image).float().unsqueeze(0)
    mask_tensor = torch.from_numpy(mask).long().unsqueeze(0)
    
    torch.save(image_tensor, image_path)
    torch.save(mask_tensor, mask_path)

os.makedirs('train_data', exist_ok=True)


for i in tqdm(range(105)):
    img = nii.load(f'qsm_data/sub-{i:04d}_ses-1_acq-wb_mod-qsm_orient-std_brain.nii.gz').get_fdata()
    
    img, mask = process_3d_volume(img)
    image_file = f'train_data/image_{2*105+i:04d}.pt'
    mask_file = f'train_data/mask_{2*105+i:04d}.pt'
    print(mask.shape)
    imshow_3d(img*mask, '', rango=(-0.1, 0.1))
    break
    # save_volumes_as_tensors(img, mask, image_file, mask_file)
# %%

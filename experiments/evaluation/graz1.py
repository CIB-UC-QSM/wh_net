# %%
1 + 1
# %%
import matplotlib.pyplot as plt
from whnet.model import ProximalNetwork, ADMMUnrolledNet
from whnet.physics import continuous_dipole_kernel, imshow_3d, rmse
from whnet.paths import (
    CHECKPOINTS_DIR,
    EVALUATION_DATA_DIR,
    EVALUATION_VOLUMES_DIR,
    EXPORTS_DIR,
    JULIA_ENV_DIR,
)
from scipy.io import loadmat
import torch.nn.functional as F
import torch
from torch.amp import autocast
import numpy as np
from typing import Optional, Union, Sequence
import shutil
import nibabel as nib
import tempfile
from pathlib import Path
import subprocess

# %%

data = loadmat(EVALUATION_DATA_DIR / "data_blend_1_graz.mat")
print(data.keys())


# %%
def romeo_unwrap(
    input_array: np.ndarray,
    affine: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    TEs: Optional[Union[Sequence[float], np.ndarray]] = None,
    julia_exe: str = "/home/mathias/.juliaup/bin/julia",
) -> np.ndarray:
    """
    Python wrapper for ROMEO (Phase Unwrapping Tool) that executes natively via the Julia CLI.
    Uses temporary files for I/O and deletes them automatically.

    Parameters
    ----------
    input_array : np.ndarray
        3D/4D NumPy array of the wrapped phase volume (4th dimension is echo).
    affine : np.ndarray, optional
        4x4 affine matrix for the NIfTI image. Defaults to an identity matrix.
    mask : np.ndarray, optional
        Binary mask array to restrict the unwrapping region.
    TEs : Sequence[float] or np.ndarray, optional
        List or array of echo times. Passing this triggers ROMEO's 4D multi-echo
        temporal unwrapping and phase offset correction.
    julia_exe : str, default='julia'
        Path to the Julia executable (assumes 'julia' is in your system PATH).

    Returns
    -------
    np.ndarray
        The unwrapped phase array.
    """
    # 1. Verify Julia is installed and accessible
    if shutil.which(julia_exe) is None:
        raise RuntimeError(
            f"Julia executable '{julia_exe}' was not found. "
            "Ensure Julia is installed and added to your system PATH."
        )

    if affine is None:
        affine = np.eye(4)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Using uncompressed .nii to avoid Julia Zlib/GZip dependency issues
        in_file = tmp_path / "input.nii"
        out_file = tmp_path / "output.nii"
        script_file = tmp_path / "run_romeo.jl"

        # 2. Save wrapped phase input to the temporary directory
        nib.save(nib.Nifti1Image(input_array, affine), in_file)

        cmd = [
            julia_exe,
            f"--project={JULIA_ENV_DIR}",
            str(script_file),
            str(in_file),
            str(out_file),
        ]

        if mask is not None:
            mask_file = tmp_path / "mask.nii"
            nib.save(nib.Nifti1Image(mask, affine), mask_file)
            cmd.append(str(mask_file))

        # 3. Format the TEs list for direct Julia injection
        if TEs is not None:
            # Convert any sequence/array to a comma-separated string of floats
            te_str = ", ".join(str(float(te)) for te in TEs)
            te_julia_code = f"TEs = Float64[{te_str}]"

            # Construct the appropriate Julia kwargs for the unwrap function
            unwrap_cmd_mask = "unwrapped = unwrap(phase; mask=mask_data, TEs=TEs)"
            unwrap_cmd_nomask = "unwrapped = unwrap(phase; TEs=TEs)"
        else:
            te_julia_code = ""
            unwrap_cmd_mask = "unwrapped = unwrap(phase; mask=mask_data)"
            unwrap_cmd_nomask = "unwrapped = unwrap(phase)"

        # 4. Generate the Julia processing script on the fly
        # (F-strings are used to inject the TE code and unwrap commands)
        julia_script = f"""
        try
            using NIfTI
            using ROMEO
        catch
            println("Required packages missing. Auto-installing NIfTI and ROMEO...")
            using Pkg
            Pkg.add(["NIfTI", "ROMEO"])
            using NIfTI
            using ROMEO
        end

        phase_path = ARGS[1]
        out_path = ARGS[2]
        has_mask = length(ARGS) > 2

        # Read phase data
        phase_img = niread(phase_path)
        phase = phase_img.raw

        # Inject Echo Times (if provided)
        {te_julia_code}

        # Apply Unwrapping
        if has_mask
            mask_path = ARGS[3]
            mask_img = niread(mask_path)
            mask_data = mask_img.raw .> 0
            {unwrap_cmd_mask}
        else
            {unwrap_cmd_nomask}
        end

        # Write unwrapped output
        out_img = NIVolume(phase_img.header, phase_img.extensions, unwrapped)
        niwrite(out_path, out_img)
        """

        script_file.write_text(julia_script)

        # 5. Execute the Julia command
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Julia ROMEO execution failed with exit code {e.returncode}.\n"
                f"Error details:\n{e.stderr}\n{e.stdout}"
            ) from e

        # 6. Load result back into a NumPy array
        unwrap_img = nib.load(out_file).get_fdata().copy()

    return unwrap_img


# %%

# unwraped_phase = romeo_unwrap(data['Phase'], mask=data['mask'], TEs=data['TE'][0])


unwraped_phase = np.zeros_like(data["Phase"])
for i in range(data["ER"].item()):
    aux = romeo_unwrap(data["Phase"][..., i], mask=data["mask"])
    unwraped_phase[..., i] = aux


# %%

magn = data["Magn"]
magn /= magn.max()

tes = data["TE"][0]

w = np.zeros(magn.shape[:3])
phase = np.zeros(magn.shape[:3])
den = np.zeros(magn.shape[:3])


for i in range(6):
    den += tes[i] * magn[..., i]
    phase += unwraped_phase[..., i] * magn[..., i] * tes[i]
    w += magn[..., i] * magn[..., i] * tes[i]

w /= den + 1e-5
phase /= den + 1e-5

imshow_3d(phase / data["phs_scale"], f"phase", rango=(-0.1, 0.1), angles=(-90, -90, 90))
imshow_3d(w, f"phase", rango=(0, 1), angles=(-90, -90, 90))


# %%
from scipy.ndimage import zoom


def resample_to_isotropic(
    volume: np.ndarray, voxel_size: tuple[float, float, float]
) -> np.ndarray:
    """Remuestra un volumen MRI 3D a un tamaño de voxel isotrópico de [1, 1, 1].

    Parameters
    ----------
    volume : np.ndarray
        Array 3D que representa el volumen de la MRI.
    voxel_size : tuple[float, float, float]
        Tamaño de voxel original en mm, ej: (0.5, 0.5, 2.0).

    Returns
    -------
    np.ndarray
        Volumen interpolado con voxel_size = [1, 1, 1].
    """
    # 1. Definir el voxel size objetivo
    target_voxel_size = (1.0, 1.0, 1.0)

    # 2. Calcular los factores de escala (zoom) para cada eje
    # Factor = voxel_size_actual / voxel_size_objetivo
    zoom_factors = [
        orig / target for orig, target in zip(voxel_size, target_voxel_size)
    ]

    # 3. Aplicar la interpolación spline de orden 3 (cúbica)
    # order=3 es el estándar para intensidad médica (trilinear es order=1)
    resampled_volume = zoom(volume, zoom=zoom_factors, order=3, mode="nearest")

    return resampled_volume


# %%

phase = resample_to_isotropic(phase, data["voxel_size"][0]) / (
    0.00492 * 2 * np.pi * 42.58 * 7
)
w = resample_to_isotropic(w, data["voxel_size"][0])
mask = resample_to_isotropic(data["mask"], data["voxel_size"][0])

# %%
w /= w.max()
mask = mask == 1

# %%
imshow_3d(mask * phase, f"phase", rango=(-0.1, 0.1), angles=(-90, -90, 90))
imshow_3d(mask * w, f"phase", rango=(0, 1), angles=(-90, -90, 90))


# %%

# %%
factor = 1
pad = 20


def pad_to_sqr_shape(volume, zero_pad):
    x, y, z = volume.shape
    max_size = (max_o := np.max([x, y, z])) + 2 * zero_pad
    new_volume = np.zeros(3 * (max_size,))
    dx = (max_o - x) // 2 + zero_pad
    dy = (max_o - y) // 2 + zero_pad
    dz = (max_o - z) // 2 + zero_pad
    new_volume[dx : x + dx, dy : y + dy, dz : z + dz] = volume
    return new_volume


coords = np.argwhere(mask)
z_min, y_min, x_min = coords.min(axis=0)
z_max, y_max, x_max = coords.max(axis=0) + 1

mask = pad_to_sqr_shape(mask[z_min:z_max, y_min:y_max, x_min:x_max], pad)
phase = pad_to_sqr_shape(phase[z_min:z_max, y_min:y_max, x_min:x_max], pad)
w = pad_to_sqr_shape(w[z_min:z_max, y_min:y_max, x_min:x_max], pad)
imshow_3d(w, "w", rango=(0, 1), angles=(-90, -90, 90))


SPATIAL = (-3, -2, -1)

phase = torch.from_numpy(phase * mask).unsqueeze(0).float() / factor
msk = torch.from_numpy(mask).unsqueeze(0).float()

# W = msk.clone()
W = torch.from_numpy(w * mask).unsqueeze(0).float()

D = torch.from_numpy(continuous_dipole_kernel(phase.shape[-3:])).unsqueeze(0).float()


phase = phase.unsqueeze(0)
msk = msk.unsqueeze(0)
D = D.unsqueeze(0)
W = W.unsqueeze(0)

imshow_3d(phase.squeeze().numpy(), f"Total", rango=(-0.1, 0.1), angles=(-90, -90, 90))


# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


net_chi = ProximalNetwork().to(device)
net_phi = ProximalNetwork().to(device)
model = ADMMUnrolledNet(net_chi, net_phi, num_iters=48).to(device)

model.load_state_dict(
    torch.load(
        CHECKPOINTS_DIR / "checkpoints_scratch7" / "model_best.pth", map_location=device
    ),
    strict=True,
)


@torch.no_grad()
def evaluate(model, phase_in, mask, D, W):
    model.eval()

    with autocast("cuda", dtype=torch.bfloat16):
        preds = model(
            phase_in.to(device),
            mask.to(device),
            D.to(device),
            mask.to(device),
            return_iterates=True,
        )
    chi_pred = [x[0].cpu() for x in preds]
    phi_pred = [x[1].cpu() for x in preds]
    return chi_pred, phi_pred

#%%
chi_preds, phi_preds = evaluate(model, phase, msk, D, W)
print(chi_preds[-1].shape)


# %%
chi_pred = chi_preds[-1].squeeze().numpy() * factor
phi_pred = phi_preds[-1].squeeze().numpy()* factor
mask = msk.squeeze().numpy()
imshow_3d(chi_pred, f"chi_pred", rango=(-0.3, 0.3), angles=(-90, -90, -90))
imshow_3d(phi_pred, "phi_pred", rango=(-0.1, 0.1), angles=(-90, -90, -90))

# %%


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


save_nii(EXPORTS_DIR / "input_phase_graz1.nii", phase * factor)

# %%


affine = np.eye(4)

nifti_img = nib.Nifti1Image(mask, affine)
nib.save(nifti_img, EXPORTS_DIR / "input_mask_graz1.nii")
# %%

pred_iqsm = nib.load(EVALUATION_VOLUMES_DIR / "iQSM.nii.gz").get_fdata()
imshow_3d(pred_iqsm * mask-(~(mask==1))*1, f"pred_iqsm", rango=(-0.3, 0.3), angles=(-90, -90, -90))
imshow_3d(chi_pred * mask-(~(mask==1))*1, f"pred_whnet", rango=(-0.3, 0.3), angles=(-90, -90, -90))

# %%

"""Compare the training harmonic loss with a boundary-safe candidate."""

# %% Imports
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from whnet.data import QSMDataset
from whnet.paths import TRAINING_DATA_DIR
from whnet.training import (
    LAM_WH,
    _LAP_KERNEL,
    complete_stencil_mask,
    weak_harmonic_loss,
)


# %% Configuration
DATA_DIR = TRAINING_DATA_DIR
SAMPLE_INDEX = 0
SEED = 0
LAMBDA_WH = LAM_WH

# Set to a path such as Path("harmonic_analysis.png") to save the figure.
SAVE_PATH: Path | None = None


# %% Analysis and visualization helpers
def symmetric_limit(volume: np.ndarray, mask: np.ndarray) -> float:
    values = np.abs(volume[mask])
    if values.size == 0:
        raise ValueError("The selected region is empty")
    limit = float(np.quantile(values, 0.995))
    return limit if limit > 0.0 else 1.0


def orthogonal_slices(volume: np.ndarray) -> tuple[np.ndarray, ...]:
    depth, height, width = volume.shape
    return (
        volume[depth // 2, :, :],
        volume[:, height // 2, :],
        volume[:, :, width // 2],
    )


# %% Load a deterministic synthetic sample
if LAMBDA_WH < 0:
    raise ValueError("LAMBDA_WH must be non-negative")

torch.manual_seed(SEED)
dataset = QSMDataset(DATA_DIR)
if not 0 <= SAMPLE_INDEX < len(dataset):
    raise IndexError(
        f"SAMPLE_INDEX {SAMPLE_INDEX} is outside a dataset of length {len(dataset)}"
    )

_, mask, _, _, _, phi = dataset[SAMPLE_INDEX]
phi = phi.unsqueeze(0)  # [C, D, H, W] -> [N, C, D, H, W]
mask = mask.unsqueeze(0)


# %% Calculate the legacy and boundary-safe harmonic losses
with torch.no_grad():
    kernel = _LAP_KERNEL.to(device=phi.device, dtype=phi.dtype)
    laplacian = F.conv3d(phi, kernel, padding=1)

    # Legacy implementation: includes centers whose stencil crosses the ROI edge.
    masked_laplacian = mask * laplacian
    legacy_loss = F.smooth_l1_loss(
        masked_laplacian,
        torch.zeros_like(masked_laplacian),
        beta=0.1,
    )

    # Candidate: accept a voxel only if its complete 3x3x3 stencil is in the ROI.
    interior_mask = complete_stencil_mask(mask)
    if interior_mask.sum().item() == 0:
        raise ValueError("No valid 3x3x3 stencil centers remain after mask erosion")
    interior_laplacian = interior_mask * laplacian

    # Drop-in replacement: it preserves the current full-volume mean reduction.
    candidate_loss_full_volume = F.smooth_l1_loss(
        interior_laplacian,
        torch.zeros_like(interior_laplacian),
        beta=0.1,
    )

    # Training replacement: normalize by the number of valid centers so that
    # mask occupancy does not change the effective strength of the loss.
    candidate_loss_valid_mean = weak_harmonic_loss(phi, mask)


# %% Report numerical results
inside = mask.bool()
valid_inside = interior_mask.bool()

print(f"sample: {DATA_DIR}[{SAMPLE_INDEX}] (seed={SEED})")
print(f"ROI fraction: {mask.float().mean().item():.6f}")
print(f"complete-stencil fraction: {interior_mask.mean().item():.6f}")
print(
    "retained ROI centers: "
    f"{(interior_mask.sum() / mask.sum()).item():.2%}"
)
print(f"phi RMS inside ROI: {phi[inside].square().mean().sqrt().item():.6e}")
print(
    "Laplacian RMS inside ROI (legacy): "
    f"{laplacian[inside].square().mean().sqrt().item():.6e}"
)
print(
    "Laplacian RMS at complete-stencil centers: "
    f"{laplacian[valid_inside].square().mean().sqrt().item():.6e}"
)
print()
print(f"legacy training loss: {legacy_loss.item():.6e}")
print(
    f"legacy weighted loss (lambda={LAMBDA_WH:g}): "
    f"{(LAMBDA_WH * legacy_loss).item():.6e}"
)
print(
    "candidate loss, legacy full-volume reduction: "
    f"{candidate_loss_full_volume.item():.6e}"
)
print(
    "candidate weighted loss, legacy full-volume reduction: "
    f"{(LAMBDA_WH * candidate_loss_full_volume).item():.6e}"
)
print(
    "candidate loss, valid-center mean (recommended): "
    f"{candidate_loss_valid_mean.item():.6e}"
)
print(
    "candidate weighted loss, valid-center mean: "
    f"{(LAMBDA_WH * candidate_loss_valid_mean).item():.6e}"
)


# %% Display phi, the legacy Laplacian, and the candidate Laplacian
phi_np = phi[0, 0].cpu().numpy()
legacy_laplacian_np = masked_laplacian[0, 0].cpu().numpy()
candidate_laplacian_np = interior_laplacian[0, 0].cpu().numpy()
mask_np = mask[0, 0].bool().cpu().numpy()
interior_mask_np = interior_mask[0, 0].bool().cpu().numpy()

row_data = (
    (phi_np, symmetric_limit(phi_np, mask_np), r"$\phi$"),
    (
        legacy_laplacian_np,
        symmetric_limit(legacy_laplacian_np, mask_np),
        r"legacy: mask $\times\ \nabla^2\phi$",
    ),
    (
        candidate_laplacian_np,
        symmetric_limit(candidate_laplacian_np, interior_mask_np),
        r"candidate: interior mask $\times\ \nabla^2\phi$",
    ),
)

fig, axes = plt.subplots(3, 3, figsize=(12, 11))
plane_names = ("axial", "coronal", "sagittal")
for row, (volume, limit, label) in enumerate(row_data):
    for column, (image, plane) in enumerate(
        zip(orthogonal_slices(volume), plane_names, strict=True)
    ):
        axis = axes[row, column]
        shown = axis.imshow(
            image,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            origin="lower",
        )
        axis.set_title(f"{label} — {plane}")
        axis.axis("off")
        fig.colorbar(shown, ax=axis, fraction=0.046, pad=0.04)

fig.suptitle(
    f"Legacy loss = {legacy_loss.item():.4e} | "
    f"interior valid-center loss = {candidate_loss_valid_mean.item():.4e}"
)
fig.tight_layout()

if SAVE_PATH is not None:
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAVE_PATH, dpi=160, bbox_inches="tight")
    print(f"figure saved to: {SAVE_PATH}")

plt.show()

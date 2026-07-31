from comet_ml import Experiment
import math
import os
import random
from collections import defaultdict

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy.io import loadmat
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import QSMDataset, complex_gaussian_noisy_phase
from utils import continuous_dipole_kernel
from wh_net import (
    ADMMUnrolledNet,
    ProximalNetwork,
    magnitude_to_data_weight,
    weighted_rms_scale,
)


# ----------------------------- Configuration -----------------------------
# Fine-tune the proven scratch5 residual proximal instead of initializing a
# different architecture. Keep this 50-iteration run in its own directory.
INIT_CKPT = "checkpoints_scratch5_k50/model_best.pth"
RESUME_CKPT = None
TRAINING_VERSION = 3
CKPT_DIR = "checkpoints_scratch5_k50"

# The checkpoint was trained near K=20. Increase depth gradually while keeping
# each stage fixed, then spend the longest stage at the requested K=50.
TRAINING_STAGES = (
    # (20, 10),
    # (30, 10),
    # (40, 15),
    (50, 25),
)
EPOCHS = sum(stage_epochs for _, stage_epochs in TRAINING_STAGES)
BATCH_SIZE = 5
NUM_WORKERS = 10
USE_FP16 = True
FP16_INIT_SCALE = 1.0
PEAK_LR = 2e-5
MIN_LR_FACTOR = 0.1
WARMUP_EPOCHS = 3
WH_WARMUP_EPOCHS = 20
CONVERGENCE_WARMUP_EPOCHS = 5

# These retain the original 100:50:10:100:25:5:2:1 tradeoff without the
# redundant common factor of 100 that caused nearly every step to be clipped.
LAM_CHI = 1.0
LAM_PHI = 0.5
LAM_GRAD = 0.1
LAM_WH = 2.0
LAM_DATA = 0.25
LAM_CONTRACTION = 0.2
LAM_PRIMAL = 0.02
LAM_DUAL = 0.02

TARGET_ITERS = 50
DEEP_SUP_K = 5
CONVERGENCE_TAIL_K = 8
CONVERGENCE_BURN_IN = 8
CONTRACTION_TARGET = 0.98
CONTRACTION_EPS = 1e-3
PROBE_DEPTHS = (10, 20, 30, 40, 50)

EVAL_TOL = 1e-3
EVAL_MAX_ITERS = TRAINING_STAGES[-1][0]
GRAD_CLIP = 10.0
EMA_DECAY = 0.999
VAL_FRAC = 0.15
SCORE_DATA_WEIGHT = 0.05
SCORE_CONVERGENCE_WEIGHT = 0.1
EARLY_STOPPING_PATIENCE = 8
SEED = 0
SOURCE_ID_MODULUS = 105
BACKGROUND_SCALE = (0.1, 5.0)
# The checkpoint was not trained for arbitrary changes of numerical units.
GLOBAL_SCALE = (0.3, 3)
TV_REGULARIZATION = (0.0, 0.01)
TV_PROBABILITY = 0.5
# scratch5 was pretrained with W equal to the ROI mask.
MASK_WEIGHT_PROBABILITY = 1.0
# Establish a stable baseline first. Re-enable the dataset's 10% sparse
# outlier augmentation only after the staged model converges reliably.
NOISE_OUTLIER_PROBABILITY = 0.1
NOISE_OUTLIER_VOXEL_COUNT = (1, 3)
NOISE_OUTLIER_FACTOR = (10.0, 100.0)

# Fixed out-of-distribution simulation evaluations, prepared once and evaluated
# with EMA parameters at the end of every epoch.
EXTERNAL_EVAL_ENABLED = True
COSMOS_FACTOR = 1
COSMOS_PAD = 12
COSMOS_SNR = 100.0
COSMOS_BACKGROUND_SCALE = 5
COSMOS_NOISE_SEED = 12_345
SIM2_FACTOR = 1
SIM2_PAD = 20
# --------------------------------------------------------------------------

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

_LAP_KERNEL = torch.tensor(
    [[[
        [[1 / 13, 3 / 26, 1 / 13], [3 / 26, 3 / 13, 3 / 26], [1 / 13, 3 / 26, 1 / 13]],
        [[3 / 26, 3 / 13, 3 / 26], [3 / 13, -44 / 13, 3 / 13], [3 / 26, 3 / 13, 3 / 26]],
        [[1 / 13, 3 / 26, 1 / 13], [3 / 26, 3 / 13, 3 / 26], [1 / 13, 3 / 26, 1 / 13]],
    ]]]
)


def masked_mean_per_sample(values, mask):
    expanded_mask = mask.expand_as(values)
    numerator = (values * expanded_mask).flatten(1).sum(dim=1)
    denominator = expanded_mask.flatten(1).sum(dim=1).clamp_min(1.0)
    return numerator / denominator


def masked_mean(values, mask):
    return masked_mean_per_sample(values, mask).mean()


def masked_l1(pred, target, mask):
    return masked_mean((pred - target).abs(), mask)


def calculate_nrmse(pred, target):
    return calculate_nrmse_per_sample(pred, target).mean()


def calculate_nrmse_per_sample(pred, target):
    error_norm = torch.linalg.vector_norm((pred - target).flatten(1), dim=1)
    target_norm = torch.linalg.vector_norm(target.flatten(1), dim=1).clamp_min(1e-8)
    return error_norm / target_norm


def spatial_gradient_3d(x):
    return (
        x[:, :, 1:, :, :] - x[:, :, :-1, :, :],
        x[:, :, :, 1:, :] - x[:, :, :, :-1, :],
        x[:, :, :, :, 1:] - x[:, :, :, :, :-1],
    )


def gradient_loss(pred, target, mask):
    px, py, pz = spatial_gradient_3d(pred)
    tx, ty, tz = spatial_gradient_3d(target)
    mask_x = mask[:, :, 1:, :, :] * mask[:, :, :-1, :, :]
    mask_y = mask[:, :, :, 1:, :] * mask[:, :, :, :-1, :]
    mask_z = mask[:, :, :, :, 1:] * mask[:, :, :, :, :-1]
    return (
        masked_mean((px - tx).abs(), mask_x)
        + masked_mean((py - ty).abs(), mask_y)
        + masked_mean((pz - tz).abs(), mask_z)
    )


def erode_mask(mask, radius=2):
    kernel = 2 * radius + 1
    outside = 1.0 - mask
    return (1.0 - F.max_pool3d(outside, kernel_size=kernel, stride=1, padding=radius)).clamp(0.0, 1.0)


def weak_harmonic_loss(phi_h, mask):
    kernel = _LAP_KERNEL.to(device=phi_h.device, dtype=phi_h.dtype)
    laplacian = F.conv3d(phi_h, kernel, padding=1)
    interior = erode_mask(mask, radius=2)
    pointwise = F.smooth_l1_loss(
        laplacian, torch.zeros_like(laplacian), beta=0.1, reduction="none"
    )
    return masked_mean(pointwise, interior)


def dipole_forward(chi, D):
    return torch.fft.ifftn(
        D * torch.fft.fftn(chi, dim=(-3, -2, -1)), dim=(-3, -2, -1)
    ).real


def data_consistency_loss(chi, phi_h, phase, mask, W, D):
    residual = dipole_forward(chi, D) + phi_h - phase
    weight = magnitude_to_data_weight(W, mask)
    return masked_mean(residual.square(), weight)


def hybrid_qsm_loss(chi_pred, chi_gt, phi_pred, phi_gt, phase, mask, W, D,
                    lam_chi=LAM_CHI, lam_phi=LAM_PHI, lam_grad=LAM_GRAD,
                    lam_wh=LAM_WH, lam_data=LAM_DATA):
    # Every objective term is evaluated in the same per-sample dimensionless
    # units used internally by the unrolled solver. This prevents large-scale
    # samples and quadratic terms from dominating the batch objective.
    data_weight = magnitude_to_data_weight(W, mask)
    scale = weighted_rms_scale(
        phase, data_weight, fallback_mask=mask
    ).detach()
    chi_pred_norm = chi_pred / scale
    chi_gt_norm = chi_gt / scale
    phi_pred_norm = phi_pred / scale
    phi_gt_norm = phi_gt / scale
    phase_norm = phase / scale

    loss_chi = masked_l1(chi_pred_norm, chi_gt_norm, mask)
    loss_phi = masked_l1(phi_pred_norm, phi_gt_norm, mask)
    loss_grad = gradient_loss(chi_pred_norm, chi_gt_norm, mask)
    loss_wh = weak_harmonic_loss(phi_pred_norm, mask)
    loss_data = data_consistency_loss(
        chi_pred_norm, phi_pred_norm, phase_norm, mask, W, D
    )
    total = (
        lam_chi * loss_chi
        + lam_phi * loss_phi
        + lam_grad * loss_grad
        + lam_wh * loss_wh
        + lam_data * loss_data
    )
    return total, {
        "loss_chi": loss_chi.item(),
        "loss_phi": loss_phi.item(),
        "loss_grad": loss_grad.item(),
        "loss_wh": loss_wh.item(),
        "loss_data": loss_data.item(),
    }


def deep_supervised_loss(iterates, chi_gt, phi_gt, phase, mask, W, D, lam_wh):
    """Weighted deep supervision over a small set of late-biased iterates."""
    count = len(iterates)
    if count == 0:
        raise RuntimeError("The model returned no iterates for deep supervision.")
    if count <= DEEP_SUP_K:
        indices = list(range(2, count))
    else:
        indices = sorted(set(torch.linspace(2, count - 1, DEEP_SUP_K).round().int().tolist()))

    weights = torch.arange(1, len(indices) + 1, device=chi_gt.device, dtype=chi_gt.dtype)
    weights = weights / weights.sum()
    total = chi_gt.new_zeros(())
    parts_weighted = {
        "loss_chi": 0.0,
        "loss_phi": 0.0,
        "loss_grad": 0.0,
        "loss_wh": 0.0,
        "loss_data": 0.0,
    }
    for weight, index in zip(weights, indices):
        chi_k, phi_k = iterates[index]
        loss_k, parts = hybrid_qsm_loss(
            chi_k, chi_gt, phi_k, phi_gt, phase, mask, W, D, lam_wh=lam_wh
        )
        total = total + weight * loss_k
        weight_value = weight.detach().item()
        for name, value in parts.items():
            parts_weighted[name] += weight_value * value
    return total, parts_weighted


def convergence_loss(
    diagnostics,
    tail_k=CONVERGENCE_TAIL_K,
    burn_in=CONVERGENCE_BURN_IN,
):
    """Penalize post-burn-in update growth and late ADMM residuals."""
    available = diagnostics["chi_step"].shape[0]
    zero = diagnostics["chi_step"].new_zeros(())
    combined_step = torch.linalg.vector_norm(
        torch.stack(
            (diagnostics["chi_step"], diagnostics["phi_step"]), dim=0
        ),
        dim=0,
    )

    # Do not penalize productive early iterations. After burn-in, require each
    # update to contract relative to the preceding (detached) update.
    if available > burn_in + 1:
        previous = combined_step[burn_in:-1].detach()
        current = combined_step[burn_in + 1:]
        relative = current / previous.clamp_min(CONTRACTION_EPS)
        log_ratio = torch.log(current + CONTRACTION_EPS) - torch.log(
            previous + CONTRACTION_EPS
        )
        contraction_excess = F.relu(
            log_ratio - math.log(CONTRACTION_TARGET)
        )
        loss_contraction = contraction_excess.square().mean()
        contraction_ratio = relative.detach().mean()
    else:
        loss_contraction = zero
        contraction_ratio = zero.detach()

    tail_start = max(min(burn_in, available - 1), available - tail_k)
    tail = slice(tail_start, available)
    loss_primal = diagnostics["primal"][tail].square().mean()
    loss_dual = diagnostics["dual"][tail].square().mean()
    total = (
        LAM_CONTRACTION * loss_contraction
        + LAM_PRIMAL * loss_primal
        + LAM_DUAL * loss_dual
    )
    return total, {
        "loss_contraction": loss_contraction.detach().item(),
        "loss_primal": loss_primal.detach().item(),
        "loss_dual": loss_dual.detach().item(),
        "contraction_ratio": contraction_ratio.item(),
    }


class EMA:
    """EMA of parameters while copying normalization buffers from the latest model."""

    def __init__(self, model, decay):
        self.decay = float(decay)
        self.num_updates = 0
        self.current_decay = 0.0
        self.param_names = {name for name, _ in model.named_parameters()}
        self.shadow = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        # Warm up the average so a short run does not retain a large fraction
        # of its randomly initialized model. The target decay is approached as
        # the number of optimization steps grows.
        self.current_decay = min(
            self.decay,
            (1.0 + self.num_updates) / (10.0 + self.num_updates),
        )
        for key, value in model.state_dict().items():
            if key in self.param_names and value.dtype.is_floating_point:
                self.shadow[key].mul_(self.current_decay).add_(
                    value.detach(), alpha=1.0 - self.current_decay
                )
            else:
                self.shadow[key].copy_(value)


def log_ortho_slices(experiment, volume, name, epoch, value_range=(-0.1, 0.1)):
    depth, height, width = volume.shape
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, image in zip(
        axes,
        [volume[depth // 2], volume[:, height // 2], volume[:, :, width // 2]],
    ):
        axis.imshow(image, cmap="gray", vmin=value_range[0], vmax=value_range[1])
        axis.axis("off")
    plt.tight_layout()
    experiment.log_figure(figure_name=f"{name}_epoch_{epoch}", figure=fig)
    plt.close(fig)


def pad_to_cube(volume, padding, center=False):
    """Pad a 3-D volume to a cube with an explicit physical zero border."""
    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3-D volume, got shape {volume.shape}.")
    cube_size = int(max(volume.shape) + 2 * padding)
    padded = np.zeros((cube_size, cube_size, cube_size), dtype=np.float32)
    if center:
        starts = [(cube_size - size) // 2 for size in volume.shape]
    else:
        starts = [padding, padding, padding]
    slices = tuple(slice(start, start + size) for start, size in zip(starts, volume.shape))
    padded[slices] = volume.astype(np.float32, copy=False)
    return padded


def mask_weight(W, mask):
    """Sanitize a non-negative mask/magnitude weight without rescaling it."""
    W = torch.nan_to_num(W.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    return W * mask


def _external_case(name, phase, mask, W, D, chi_target, display_factor):
    return {
        "name": name,
        "phase": phase.contiguous(),
        "mask": mask.contiguous(),
        "W": mask_weight(W, mask).contiguous(),
        "D": D.contiguous(),
        "chi_target": chi_target.contiguous(),
        "display_factor": float(display_factor),
    }


def build_cosmos_evaluation():
    """Reproduce the deterministic COSMOS simulation used in test_cosmos.py."""
    chi_gt = pad_to_cube(loadmat("chi_cosmos.mat")["chi_cosmos"], COSMOS_PAD)
    mask_np = pad_to_cube(loadmat("msk.mat")["msk"], COSMOS_PAD)

    mask = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)
    chi_target = torch.from_numpy(chi_gt).float().unsqueeze(0).unsqueeze(0) / COSMOS_FACTOR
    D = torch.from_numpy(continuous_dipole_kernel(chi_gt.shape)).float().unsqueeze(0).unsqueeze(0)

    local = dipole_forward(chi_target * mask, D) * mask
    dilated_mask = F.max_pool3d(mask, kernel_size=7, stride=1, padding=3)
    exterior_chi = (1.0 - dilated_mask) * COSMOS_BACKGROUND_SCALE
    phi_h = dipole_forward(exterior_chi, D) * mask

    field_peak = local[mask > 0].abs().max().clamp_min(0.0)
    generator = torch.Generator().manual_seed(COSMOS_NOISE_SEED)
    clean_phase = mask * (local + phi_h)
    phase = mask * complex_gaussian_noisy_phase(
        clean_phase,
        mask,
        field_peak / COSMOS_SNR,
        generator=generator,
    )
    return _external_case("cosmos", phase, mask, mask, D, chi_target, COSMOS_FACTOR)


def build_sim2_evaluation():
    """Reproduce the crop, centered padding, and weighting in test_sim2.py."""
    mask_raw = loadmat("mask_final.mat")["mask_final"]
    phase_raw = loadmat("Sim2.mat")["phase"]
    # W_raw = loadmat("sim2_w.mat")["w"]
    W_raw = loadmat("mask_final.mat")["mask_final"]
    chi_raw = nib.load("Sim2ChiGT.nii.gz").get_fdata()

    coordinates = np.argwhere(mask_raw > 0)
    if coordinates.size == 0:
        raise RuntimeError("mask_final.mat does not contain any non-zero ROI voxel.")
    lower = coordinates.min(axis=0)
    upper = coordinates.max(axis=0) + 1
    crop = tuple(slice(start, stop) for start, stop in zip(lower, upper))

    mask_np = pad_to_cube(mask_raw[crop], SIM2_PAD, center=True)
    phase_np = pad_to_cube(phase_raw[crop], SIM2_PAD, center=True)
    W_np = pad_to_cube(W_raw[crop], SIM2_PAD, center=True)
    chi_np = pad_to_cube(chi_raw[crop], SIM2_PAD, center=True)

    mask = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)
    phase = torch.from_numpy(phase_np).float().unsqueeze(0).unsqueeze(0) / SIM2_FACTOR
    W = torch.from_numpy(W_np).float().unsqueeze(0).unsqueeze(0)
    chi_target = torch.from_numpy(chi_np).float().unsqueeze(0).unsqueeze(0) / SIM2_FACTOR
    D = torch.from_numpy(continuous_dipole_kernel(phase_np.shape)).float().unsqueeze(0).unsqueeze(0)
    return _external_case("sim2", phase, mask, W, D, chi_target, SIM2_FACTOR)


def build_external_evaluations():
    if not EXTERNAL_EVAL_ENABLED:
        return []
    required = [
        "chi_cosmos.mat", "msk.mat", "mask_final.mat", "Sim2.mat",
        "Sim2ChiGT.nii.gz",
    ]
    missing = [path for path in required if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            "External epoch-end evaluation requires: " + ", ".join(missing)
        )
    return [build_cosmos_evaluation(), build_sim2_evaluation()]


@torch.no_grad()
def evaluate_external_case(model, case, device, max_iters):
    """Run one fixed simulation and return a CPU result suitable for Comet."""
    was_training = model.training
    model.eval()
    phase = case["phase"].to(device, non_blocking=True)
    mask = case["mask"].to(device, non_blocking=True)
    W = case["W"].to(device, non_blocking=True)
    D = case["D"].to(device, non_blocking=True)
    chi_target = case["chi_target"].to(device, non_blocking=True)
    with amp_context(device):
        chi_pred, _ = model(
            phase, mask, D, W, tol=EVAL_TOL, max_iters=max_iters
        )
    nrmse = calculate_nrmse(chi_pred * mask, chi_target * mask).item()
    model.train(was_training)
    return {
        "name": case["name"],
        "nrmse": nrmse,
        "phase": case["phase"][0, 0].cpu().numpy() * case["display_factor"],
        "mask": case["mask"][0, 0].cpu().numpy(),
        "prediction": chi_pred[0, 0].float().cpu().numpy() * case["display_factor"],
        "target": case["chi_target"][0, 0].cpu().numpy() * case["display_factor"],
    }


def _orthogonal_views(volume):
    depth, height, width = volume.shape
    return [volume[depth // 2], volume[:, height // 2], volume[:, :, width // 2]]


def log_external_result(experiment, result, epoch):
    """Log prediction, target, error, and input field as one Comet figure."""
    prediction = result["prediction"]
    target = result["target"]
    error = (prediction - target) * result["mask"]
    phase = result["phase"] * result["mask"]
    figure, axes = plt.subplots(3, 4, figsize=(14, 10))
    columns = [
        (prediction, "prediction", (-0.1, 0.1)),
        (target, "target", (-0.1, 0.1)),
        (error, "error", (-0.05, 0.05)),
        (phase, "input field", (-0.1, 0.1)),
    ]
    for row, view_index in enumerate(range(3)):
        for column, (volume, title, value_range) in enumerate(columns):
            axes[row, column].imshow(
                _orthogonal_views(volume)[view_index],
                cmap="gray", vmin=value_range[0], vmax=value_range[1],
            )
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(title)
    figure.suptitle(f"{result['name']} | NRMSE = {100.0 * result['nrmse']:.2f}%")
    figure.tight_layout()
    experiment.log_figure(
        figure_name=f"{result['name']}_reconstruction_epoch_{epoch}", figure=figure
    )
    plt.close(figure)


def evaluate_and_log_external_cases(
    experiment, model, cases, device, epoch, step, max_iters
):
    metrics = {}
    for case in cases:
        result = evaluate_external_case(model, case, device, max_iters)
        metric_name = f"external_{result['name']}_nrmse_chi"
        experiment.log_metric(metric_name, result["nrmse"], step=step)
        log_external_result(experiment, result, epoch)
        metrics[result["name"]] = result["nrmse"]
    return metrics


def training_stage_for_epoch(epoch):
    """Return stage index, fixed depth, local epoch, and stage boundaries."""
    start = 0
    for stage_index, (depth, stage_epochs) in enumerate(TRAINING_STAGES):
        end = start + stage_epochs
        if start <= epoch < end:
            return stage_index, depth, epoch - start, start, end
        start = end
    raise IndexError(f"Epoch {epoch} is outside the configured training stages.")


def split_by_subject(group_ids, val_fraction, seed):
    groups = defaultdict(list)
    for index, group in enumerate(group_ids):
        groups[group].append(index)
    unique_groups = sorted(groups)
    if len(unique_groups) < 2:
        raise RuntimeError("At least two source-subject groups are required for a train/validation split.")

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(unique_groups), generator=generator).tolist()
    n_val = min(
        len(unique_groups) - 1,
        max(1, round(len(unique_groups) * val_fraction)),
    )
    val_groups = {unique_groups[index] for index in order[:n_val]}
    train_indices = [index for group in unique_groups if group not in val_groups for index in groups[group]]
    val_indices = [index for group in unique_groups if group in val_groups for index in groups[group]]
    return train_indices, val_indices, val_groups


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def amp_context(device):
    """Use FP16 for CUDA learned kernels and recurrent state storage."""
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=USE_FP16 and device.type == "cuda",
    )


def atomic_torch_save(payload, path):
    """Write a checkpoint completely before replacing its previous version."""
    temporary_path = f"{path}.tmp"
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


@torch.no_grad()
def refresh_spectral_norm(model, power_iterations=5):
    """Align spectral-normalization vectors after loading EMA parameters."""
    was_training = model.training
    model.train()
    for _ in range(power_iterations):
        for module in model.modules():
            if isinstance(module, torch.nn.Conv3d) and hasattr(
                module, "parametrizations"
            ) and "weight" in module.parametrizations:
                _ = module.weight
    model.train(was_training)


@torch.no_grad()
def evaluate(model, loader, device, max_iters):
    """Deterministic stage-matched validation with convergence diagnostics."""
    was_training = model.training
    model.eval()
    totals = {
        "nrmse_chi": 0.0,
        "phi_l1": 0.0,
        "data_rms": 0.0,
        "tail_step": 0.0,
        "tail_growth": 0.0,
        "tail_primal": 0.0,
        "tail_dual": 0.0,
    }
    count = 0
    for phase, mask, W, D, chi_gt, phi_gt in loader:
        phase = phase.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        W = W.to(device, non_blocking=True)
        D = D.to(device, non_blocking=True)
        chi_gt = chi_gt.to(device, non_blocking=True)
        phi_gt = phi_gt.to(device, non_blocking=True)
        with amp_context(device):
            chi_pred, phi_pred, diagnostics = model(
                phase,
                mask,
                D,
                W,
                return_diagnostics=True,
                tol=None,
                max_iters=max_iters,
            )
        data_weight = magnitude_to_data_weight(W, mask)
        scale = weighted_rms_scale(
            phase, data_weight, fallback_mask=mask
        )
        residual = (
            dipole_forward(chi_pred, D) + phi_pred - phase
        ) / scale
        data_rms = (
            (data_weight * residual.square()).flatten(1).sum(dim=1)
            / data_weight.flatten(1).sum(dim=1).clamp_min(1.0)
        ).sqrt()
        phi_l1 = masked_mean_per_sample(
            (phi_pred - phi_gt).abs() / scale, mask
        )
        tail = slice(
            max(0, diagnostics["chi_step"].shape[0] - CONVERGENCE_TAIL_K),
            None,
        )
        combined_step = torch.linalg.vector_norm(
            torch.stack(
                (diagnostics["chi_step"], diagnostics["phi_step"]), dim=0
            ),
            dim=0,
        )
        tail_step = combined_step[tail].mean(dim=(0, 2))
        if combined_step.shape[0] > CONVERGENCE_BURN_IN + 1:
            previous = combined_step[CONVERGENCE_BURN_IN:-1]
            current = combined_step[CONVERGENCE_BURN_IN + 1:]
            tail_growth = (
                current / previous.clamp_min(CONTRACTION_EPS)
            )[-CONVERGENCE_TAIL_K:].mean(dim=(0, 2))
        else:
            tail_growth = torch.zeros_like(tail_step)
        tail_primal = diagnostics["primal"][tail].mean(dim=(0, 2))
        tail_dual = diagnostics["dual"][tail].mean(dim=(0, 2))
        batch_values = {
            "nrmse_chi": calculate_nrmse_per_sample(
                chi_pred * mask, chi_gt * mask
            ),
            "phi_l1": phi_l1,
            "data_rms": data_rms,
            "tail_step": tail_step,
            "tail_growth": tail_growth,
            "tail_primal": tail_primal,
            "tail_dual": tail_dual,
        }
        batch_size = phase.shape[0]
        for name, values in batch_values.items():
            totals[name] += values.sum().item()
        count += batch_size
    model.train(was_training)
    return {name: value / max(1, count) for name, value in totals.items()}


@torch.no_grad()
def evaluate_probe_depths(model, batch, device, depths):
    """Evaluate one fixed validation batch at all trained probe depths."""
    depths = tuple(sorted(set(int(depth) for depth in depths)))
    if not depths:
        return {}
    was_training = model.training
    model.eval()
    phase, mask, W, D, chi_gt, _ = (
        value.to(device, non_blocking=True) for value in batch
    )
    with amp_context(device):
        iterates = model(
            phase,
            mask,
            D,
            W,
            return_iterates=True,
            max_iters=max(depths),
        )
    metrics = {}
    for depth in depths:
        chi_pred, _ = iterates[depth - 1]
        metrics[f"probe_nrmse_chi_k{depth}"] = calculate_nrmse_per_sample(
            chi_pred * mask, chi_gt * mask
        ).mean().item()
    model.train(was_training)
    return metrics


if __name__ == "__main__":
    if INIT_CKPT is not None and RESUME_CKPT is not None:
        raise ValueError("Set only one of INIT_CKPT and RESUME_CKPT.")
    stage_depths = [depth for depth, _ in TRAINING_STAGES]
    stage_epochs = [epochs for _, epochs in TRAINING_STAGES]
    if (
        not TRAINING_STAGES
        or any(depth < 3 for depth in stage_depths)
        or any(epochs < 1 for epochs in stage_epochs)
        or stage_depths != sorted(set(stage_depths))
        or stage_depths[-1] != TARGET_ITERS
    ):
        raise ValueError(
            "Training stages must have unique increasing depths >= 3, positive "
            "epoch counts, and finish at TARGET_ITERS."
        )
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_device = device
    loader_workers = 0 if data_device.type == "cuda" else NUM_WORKERS

    experiment = Experiment(project_name="wh-net-qsm")
    experiment.log_parameters({
        "mode": "scratch5_pretrained_residual_admm_k50",
        "training_version": TRAINING_VERSION,
        "init_ckpt": str(INIT_CKPT),
        "resume_ckpt": str(RESUME_CKPT),
        "ckpt_dir": CKPT_DIR,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "data_device": str(data_device),
        "loader_workers": loader_workers,
        "use_fp16": USE_FP16,
        "fp16_init_scale": FP16_INIT_SCALE,
        "peak_lr": PEAK_LR,
        "min_lr_factor": MIN_LR_FACTOR,
        "warmup_epochs": WARMUP_EPOCHS,
        "wh_warmup_epochs": WH_WARMUP_EPOCHS,
        "convergence_warmup_epochs": CONVERGENCE_WARMUP_EPOCHS,
        "lam_chi": LAM_CHI,
        "lam_phi": LAM_PHI,
        "lam_grad": LAM_GRAD,
        "lam_wh": LAM_WH,
        "lam_data": LAM_DATA,
        "lam_contraction": LAM_CONTRACTION,
        "lam_primal": LAM_PRIMAL,
        "lam_dual": LAM_DUAL,
        "training_stages": TRAINING_STAGES,
        "target_iters": TARGET_ITERS,
        "deep_sup_k": DEEP_SUP_K,
        "convergence_tail_k": CONVERGENCE_TAIL_K,
        "convergence_burn_in": CONVERGENCE_BURN_IN,
        "contraction_target": CONTRACTION_TARGET,
        "probe_depths": PROBE_DEPTHS,
        "eval_tol": EVAL_TOL,
        "eval_max_iters": EVAL_MAX_ITERS,
        "ema_decay": EMA_DECAY,
        "score_data_weight": SCORE_DATA_WEIGHT,
        "score_convergence_weight": SCORE_CONVERGENCE_WEIGHT,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "val_frac": VAL_FRAC,
        "source_id_modulus": SOURCE_ID_MODULUS,
        "background_scale": BACKGROUND_SCALE,
        "global_scale": GLOBAL_SCALE,
        "tv_regularization": TV_REGULARIZATION,
        "tv_probability": TV_PROBABILITY,
        "mask_weight_probability": MASK_WEIGHT_PROBABILITY,
        "noise_outlier_probability": NOISE_OUTLIER_PROBABILITY,
        "noise_outlier_voxel_count": NOISE_OUTLIER_VOXEL_COUNT,
        "noise_outlier_factor": NOISE_OUTLIER_FACTOR,
        "external_eval_enabled": EXTERNAL_EVAL_ENABLED,
        "cosmos_factor": COSMOS_FACTOR,
        "cosmos_pad": COSMOS_PAD,
        "cosmos_snr": COSMOS_SNR,
        "cosmos_background_scale": COSMOS_BACKGROUND_SCALE,
        "sim2_factor": SIM2_FACTOR,
        "sim2_pad": SIM2_PAD,
    })

    train_full = QSMDataset(
        training=True,
        seed=SEED,
        source_id_modulus=SOURCE_ID_MODULUS,
        background_scale=BACKGROUND_SCALE,
        global_scale=GLOBAL_SCALE,
        tv_regularization=TV_REGULARIZATION,
        tv_probability=TV_PROBABILITY,
        mask_weight_probability=MASK_WEIGHT_PROBABILITY,
        noise_outlier_probability=NOISE_OUTLIER_PROBABILITY,
        noise_outlier_voxel_count=NOISE_OUTLIER_VOXEL_COUNT,
        noise_outlier_factor=NOISE_OUTLIER_FACTOR,
        device=data_device,
    )
    val_full = QSMDataset(
        training=False,
        seed=SEED + 10_000,
        source_id_modulus=SOURCE_ID_MODULUS,
        background_scale=BACKGROUND_SCALE,
        global_scale=GLOBAL_SCALE,
        tv_regularization=TV_REGULARIZATION,
        tv_probability=TV_PROBABILITY,
        mask_weight_probability=MASK_WEIGHT_PROBABILITY,
        noise_outlier_probability=NOISE_OUTLIER_PROBABILITY,
        noise_outlier_voxel_count=NOISE_OUTLIER_VOXEL_COUNT,
        noise_outlier_factor=NOISE_OUTLIER_FACTOR,
        device=data_device,
    )
    train_indices, val_indices, val_groups = split_by_subject(
        train_full.group_ids, VAL_FRAC, SEED
    )
    if set(train_full.group_ids[index] for index in train_indices) & val_groups:
        raise AssertionError("Subject leakage between training and validation sets.")

    train_set = Subset(train_full, train_indices)
    val_set = Subset(val_full, val_indices)
    # CUDA tensors must be created in the main process: worker processes
    # would each initialize a separate CUDA context. CPU fallback retains
    # parallel loading, persistent workers, and worker-side prefetching.
    pin_memory = device.type == "cuda" and data_device.type == "cpu"
    loader_generator = torch.Generator().manual_seed(SEED)
    loader_kwargs = {
        "num_workers": loader_workers,
        "pin_memory": pin_memory,
    }
    if loader_workers > 0:
        loader_kwargs.update({
            "persistent_workers": True,
            "prefetch_factor": 2,
            "worker_init_fn": seed_worker,
        })
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        generator=loader_generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        generator=torch.Generator().manual_seed(SEED + 1),
        **loader_kwargs,
    )
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise RuntimeError(
            "Training and validation loaders must both contain at least one batch."
        )
    probe_batch = next(iter(val_loader))

    net_chi = ProximalNetwork().to(device)
    net_phi = ProximalNetwork().to(device)
    model = ADMMUnrolledNet(
        net_chi,
        net_phi,
        num_iters=TRAINING_STAGES[0][0],
    ).to(device)
    external_cases = build_external_evaluations()
    if external_cases:
        print("Prepared external epoch-end evaluations: " + ", ".join(
            case["name"] for case in external_cases
        ))

    if INIT_CKPT is not None:
        if not os.path.isfile(INIT_CKPT):
            raise FileNotFoundError(f"Initial checkpoint not found: {INIT_CKPT}")
        model.load_state_dict(torch.load(INIT_CKPT, map_location=device, weights_only=True), strict=True)
        print(f"Loaded initial weights from {INIT_CKPT}")
    else:
        print("Training checkpoint-compatible residual ADMM model from scratch")

    atomic_torch_save(
        {
            "training_version": TRAINING_VERSION,
            "solver": "scratch5-compatible residual ADMM",
            "eps": model.eps,
            "initial_num_iters": model.num_iters,
            "penalties": {"y": 1.0, "u": 1.0, "v": 1.0},
            "conditioning": "raw iteration index and update RMS",
            "normalization": "none inside the solver (checkpoint-compatible units)",
            "chi_update": "scratch5 unit-penalty Fourier update and ROI projection",
            "weight_semantics": "W is the direct non-negative data-fidelity weight",
            "proximal_update": "residual x + softplus(gate)*network(x)",
            "precision": "CUDA FP16 learned kernels/state with FP32 FFT physics and gradient scaling",
            "training_stages": TRAINING_STAGES,
            "convergence_supervision": "post-burn-in contraction plus late primal and dual residuals",
        },
        os.path.join(CKPT_DIR, "solver_config.pth"),
    )

    decay_parameters = []
    no_decay_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or "parametrizations.weight.original" in name:
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)
    optimizer = optim.AdamW(
        [
            {"params": decay_parameters, "weight_decay": 1e-5},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ],
        lr=PEAK_LR,
    )

    
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = steps_per_epoch * WARMUP_EPOCHS

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return MIN_LR_FACTOR + (1.0 - MIN_LR_FACTOR) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=FP16_INIT_SCALE,
        enabled=USE_FP16 and device.type == "cuda",
    )
    ema = EMA(model, EMA_DECAY)
    global_step = 0
    active_stage_index = -1
    stage_best_score = float("inf")
    start_epoch = 0
    epochs_without_improvement = 0

    if RESUME_CKPT is not None:
        if not os.path.isfile(RESUME_CKPT):
            raise FileNotFoundError(f"Resume checkpoint not found: {RESUME_CKPT}")
        resume = torch.load(RESUME_CKPT, map_location=device, weights_only=False)
        if resume.get("training_version") != TRAINING_VERSION:
            raise RuntimeError(
                "Resume checkpoint uses incompatible training/proximal semantics. "
                "Start from the scratch5 pretrained weights instead."
            )
        model.load_state_dict(resume["model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        if resume.get("scaler") is not None:
            scaler.load_state_dict(resume["scaler"])
        ema.shadow = {
            key: value.to(device) for key, value in resume["ema_shadow"].items()
        }
        ema.num_updates = int(resume["ema_num_updates"])
        ema.current_decay = float(resume["ema_current_decay"])
        global_step = int(resume["global_step"])
        active_stage_index = int(resume["stage_index"])
        stage_best_score = float(resume["stage_best_score"])
        start_epoch = int(resume["epoch"])
        epochs_without_improvement = int(
            resume.get("epochs_without_improvement", 0)
        )
        torch.set_rng_state(resume["torch_rng_state"].cpu())
        if torch.cuda.is_available() and resume.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(resume["cuda_rng_state"])
        np.random.set_state(resume["numpy_rng_state"])
        random.setstate(resume["python_rng_state"])
        loader_generator.set_state(resume["loader_generator_state"].cpu())
        print(f"Resumed training from epoch {start_epoch}: {RESUME_CKPT}")
    model.train()

    for epoch in range(start_epoch, EPOCHS):
        stage_index, stage_depth, stage_epoch, stage_start, stage_end = (
            training_stage_for_epoch(epoch)
        )
        if stage_index != active_stage_index:
            if active_stage_index >= 0:
                previous_depth = TRAINING_STAGES[active_stage_index][0]
                previous_best = os.path.join(
                    CKPT_DIR, f"model_best_k{previous_depth}.pth"
                )
                model.load_state_dict(
                    torch.load(
                        previous_best,
                        map_location=device,
                        weights_only=True,
                    ),
                    strict=True,
                )
                # Stage promotion starts from the best EMA model. Optimizer
                # moments and the EMA itself are reset because they belong to
                # the previous depth's objective.
                optimizer.state.clear()
                ema = EMA(model, EMA_DECAY)
                print(
                    f"Promoted best K={previous_depth} EMA weights into the "
                    f"next stage."
                )
            active_stage_index = stage_index
            stage_best_score = float("inf")
            epochs_without_improvement = 0
            print(
                f"Starting stage {stage_index + 1}/{len(TRAINING_STAGES)}: "
                f"fixed depth {stage_depth}, epochs {stage_start + 1}-{stage_end}."
            )
        model.num_iters = stage_depth
        wh_factor = min(1.0, (epoch + 1) / max(1, WH_WARMUP_EPOCHS))
        convergence_factor = min(
            1.0, (epoch + 1) / max(1, CONVERGENCE_WARMUP_EPOCHS)
        )
        skipped = 0
        clipped = 0
        successful_steps = 0
        epoch_totals = defaultdict(float)
        pbar = tqdm(
            train_loader,
            desc=(
                f"Epoch {epoch + 1}/{EPOCHS} "
                f"(stage {stage_index + 1}, K={stage_depth})"
            ),
        )

        for phase, mask, W, D, chi_gt, phi_gt in pbar:
            phase = phase.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            W = W.to(device, non_blocking=True)
            D = D.to(device, non_blocking=True)
            chi_gt = chi_gt.to(device, non_blocking=True)
            phi_gt = phi_gt.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with amp_context(device):
                iterates, diagnostics = model(
                    phase,
                    mask,
                    D,
                    W,
                    return_iterates=True,
                    return_diagnostics=True,
                )
                reconstruction_loss, parts = deep_supervised_loss(
                    iterates, chi_gt, phi_gt, phase, mask, W, D,
                    lam_wh=LAM_WH * wh_factor,
                )
                residual_loss, residual_parts = convergence_loss(diagnostics)
                loss = reconstruction_loss + convergence_factor * residual_loss
            parts.update(residual_parts)
            chi_pred, phi_pred = iterates[-1]
            alpha_chi = model.chinet.last_alpha.mean().item()
            alpha_phi = model.phinet.last_alpha.mean().item()

            if not torch.isfinite(loss):
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradients_finite = all(
                parameter.grad is None
                or torch.isfinite(parameter.grad).all().item()
                for parameter in model.parameters()
            )
            if not gradients_finite:
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=GRAD_CLIP,
                error_if_nonfinite=True,
            )
            clipped += int(grad_norm.item() > GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)

            with torch.no_grad():
                nrmse_chi = calculate_nrmse(chi_pred * mask, chi_gt * mask)
                data_weight = magnitude_to_data_weight(W, mask)
                phase_scale = weighted_rms_scale(
                    phase, data_weight, fallback_mask=mask
                )
                phi_l1 = masked_l1(
                    phi_pred / phase_scale, phi_gt / phase_scale, mask
                )

            experiment.log_metric("loss", loss.item(), step=global_step)
            experiment.log_metric("nrmse_chi", nrmse_chi.item(), step=global_step)
            experiment.log_metric("phi_l1", phi_l1.item(), step=global_step)
            experiment.log_metric("lr", scheduler.get_last_lr()[0], step=global_step)
            experiment.log_metric("grad_norm", grad_norm.item(), step=global_step)
            experiment.log_metric("num_iters", model.num_iters, step=global_step)
            experiment.log_metric("stage_index", stage_index + 1, step=global_step)
            experiment.log_metric("ema_decay", ema.current_decay, step=global_step)
            experiment.log_metric("grad_scale", scaler.get_scale(), step=global_step)
            experiment.log_metric("convergence_factor", convergence_factor, step=global_step)
            experiment.log_metric("alpha_chi", alpha_chi, step=global_step)
            experiment.log_metric("alpha_phi", alpha_phi, step=global_step)
            for name, value in parts.items():
                experiment.log_metric(name, value, step=global_step)
            weighted_parts = {
                "weighted_chi": LAM_CHI * parts["loss_chi"],
                "weighted_phi": LAM_PHI * parts["loss_phi"],
                "weighted_grad": LAM_GRAD * parts["loss_grad"],
                "weighted_wh": LAM_WH * wh_factor * parts["loss_wh"],
                "weighted_data": LAM_DATA * parts["loss_data"],
                "weighted_contraction": (
                    convergence_factor
                    * LAM_CONTRACTION
                    * parts["loss_contraction"]
                ),
                "weighted_primal": convergence_factor * LAM_PRIMAL * parts["loss_primal"],
                "weighted_dual": convergence_factor * LAM_DUAL * parts["loss_dual"],
            }
            for name, value in weighted_parts.items():
                experiment.log_metric(name, value, step=global_step)

            successful_steps += 1
            epoch_totals["loss"] += loss.item()
            epoch_totals["nrmse_chi"] += nrmse_chi.item()
            epoch_totals["phi_l1"] += phi_l1.item()
            epoch_totals["grad_norm"] += grad_norm.item()
            epoch_totals["contraction_ratio"] += parts["contraction_ratio"]

            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                chi=f"{nrmse_chi.item():.3f}",
                phi=f"{phi_l1.item():.3f}",
                it=model.num_iters,
                lr=f"{scheduler.get_last_lr()[0]:.1e}",
            )
            global_step += 1

        if successful_steps == 0:
            raise RuntimeError(
                f"Every optimization step was skipped in epoch {epoch + 1}."
            )
        experiment.log_metric("skipped_frac", skipped / max(1, len(train_loader)), step=global_step)
        experiment.log_metric("clipped_frac", clipped / max(1, len(train_loader) - skipped), step=global_step)
        for name, total in epoch_totals.items():
            experiment.log_metric(
                f"epoch_train_{name}",
                total / max(1, successful_steps),
                step=global_step,
            )

        backup = {key: value.detach().clone() for key, value in model.state_dict().items()}
        model.load_state_dict(ema.shadow, strict=True)
        refresh_spectral_norm(model)
        ema_eval_state = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }
        model.num_iters = stage_depth
        val_metrics = evaluate(model, val_loader, device, max_iters=stage_depth)
        trained_probe_depths = tuple(
            depth for depth in PROBE_DEPTHS if depth <= stage_depth
        )
        probe_metrics = evaluate_probe_depths(
            model, probe_batch, device, trained_probe_depths
        )
        external_metrics = evaluate_and_log_external_cases(
            experiment,
            model,
            external_cases,
            device,
            epoch + 1,
            global_step,
            max_iters=stage_depth,
        )
        model.load_state_dict(backup, strict=True)
        model.train()

        for name, value in val_metrics.items():
            experiment.log_metric(f"val_{name}", value, step=global_step)
        for name, value in probe_metrics.items():
            experiment.log_metric(name, value, step=global_step)
        experiment.log_metric("val_stage_depth", stage_depth, step=global_step)
        experiment.log_metric("val_stage_epoch", stage_epoch + 1, step=global_step)
        experiment.log_metric("wh_factor", wh_factor, step=global_step)
        for name, value in external_metrics.items():
            print(f"[epoch {epoch + 1}] external {name} NRMSE: {value:.4f}")

        score = (
            val_metrics["nrmse_chi"]
            + SCORE_DATA_WEIGHT * val_metrics["data_rms"]
            + SCORE_CONVERGENCE_WEIGHT
            * (
                max(
                    0.0,
                    val_metrics["tail_growth"] - CONTRACTION_TARGET,
                )
                + val_metrics["tail_primal"]
                + val_metrics["tail_dual"]
            )
        )
        if not math.isfinite(score):
            raise RuntimeError(
                "Validation produced a non-finite selection score: "
                + ", ".join(
                    f"{name}={value}" for name, value in val_metrics.items()
                )
        )
        experiment.log_metric("val_score", score, step=global_step)
        if score < stage_best_score:
            stage_best_score = score
            epochs_without_improvement = 0
            atomic_torch_save(
                ema_eval_state,
                os.path.join(CKPT_DIR, f"model_best_k{stage_depth}.pth"),
            )
            # This alias always points to the best model in the most recently
            # entered stage; after the final stage it is deployment-ready.
            atomic_torch_save(
                ema_eval_state, os.path.join(CKPT_DIR, "model_best.pth")
            )
        else:
            epochs_without_improvement += 1
        experiment.log_metric(
            "stage_best_val_score", stage_best_score, step=global_step
        )
        atomic_torch_save(
            ema_eval_state, os.path.join(CKPT_DIR, "model_last.pth")
        )
        atomic_torch_save(
            {
                "training_version": TRAINING_VERSION,
                "epoch": epoch + 1,
                "global_step": global_step,
                "stage_index": active_stage_index,
                "stage_depth": stage_depth,
                "stage_best_score": stage_best_score,
                "epochs_without_improvement": epochs_without_improvement,
                "model": model.state_dict(),
                "ema_shadow": ema.shadow,
                "ema_num_updates": ema.num_updates,
                "ema_current_decay": ema.current_decay,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                ),
                "numpy_rng_state": np.random.get_state(),
                "python_rng_state": random.getstate(),
                "loader_generator_state": loader_generator.get_state(),
            },
            os.path.join(CKPT_DIR, "training_last.pth"),
        )

        if (epoch + 1) % 2 == 0:
            model.load_state_dict(ema_eval_state, strict=True)
            model.eval()
            with torch.no_grad(), amp_context(device):
                chi_pred, phi_pred = model(
                    phase, mask, D, W, tol=EVAL_TOL, max_iters=stage_depth
                )
            for volume, name in [
                (phase, "phase_in"),
                (chi_pred, "chi_pred"),
                (phi_pred, "phih_pred"),
                (chi_gt, "chi_gt"),
                (phi_gt, "phih_gt"),
            ]:
                log_ortho_slices(
                    experiment,
                    (volume * mask).detach().cpu().numpy()[0, 0],
                    name,
                    epoch + 1,
                )
            model.load_state_dict(backup, strict=True)
            model.train()

        is_final_stage = stage_index == len(TRAINING_STAGES) - 1
        if (
            is_final_stage
            and epochs_without_improvement >= EARLY_STOPPING_PATIENCE
        ):
            print(
                f"Final-stage early stopping at epoch {epoch + 1}; no "
                f"K={stage_depth} score improvement for "
                f"{EARLY_STOPPING_PATIENCE} epochs."
            )
            break

    model.load_state_dict(
        torch.load(os.path.join(CKPT_DIR, "model_best.pth"), map_location=device, weights_only=True),
        strict=True,
    )
    refresh_spectral_norm(model)
    val_metrics = evaluate(
        model, val_loader, device, max_iters=EVAL_MAX_ITERS
    )
    print("[BEST EMA] " + " | ".join(
        f"{name}: {value:.4f}" for name, value in val_metrics.items()
    ))
    experiment.end()

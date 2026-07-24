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
from torch.amp import autocast
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import QSMDataset
from utils import continuous_dipole_kernel
from wh_net import ADMMUnrolledNet, ProximalNetwork


# ----------------------------- Configuration -----------------------------
# Continue from the best checkpoint trained with the restored unit-penalty
# ADMM formulation. Keep this run separate from the newer solver experiments.
INIT_CKPT = "checkpoints_scratch5/model_best.pth"
CKPT_DIR = "checkpoints_restored_admm"

EPOCHS = 50
BATCH_SIZE = 5
NUM_WORKERS = 10
PEAK_LR = 1e-4
MIN_LR_FACTOR = 0.1
WARMUP_EPOCHS = 5
WH_WARMUP_EPOCHS = 10

LAM_CHI = 1000.0
LAM_PHI = 10.0
LAM_GRAD = 10.0
LAM_WH = 1200.0
LAM_DATA = 10.0

ITER_START = 30
ITER_MIN = 28
ITER_SAMPLE_MAX = 32
ITER_RAMP_EPOCHS = 5
DEEP_SUP_K = 4

EVAL_TOL = 1e-3
EVAL_MAX_ITERS = 15
GRAD_CLIP = 1.0
EMA_DECAY = 0.999
VAL_FRAC = 0.15
VAL_PHI_WEIGHT = 0.25
SEED = 0
SOURCE_ID_MODULUS = 105
BACKGROUND_SCALE = (0.001, 20.0)
TV_REGULARIZATION = (0.0, 0.01)
TV_PROBABILITY = 0.5
WEIGHT_AUGMENTATION_PROBABILITY = 0.5

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

torch.backends.cudnn.benchmark = True

_LAP_KERNEL = torch.tensor(
    [[[
        [[1 / 13, 3 / 26, 1 / 13], [3 / 26, 3 / 13, 3 / 26], [1 / 13, 3 / 26, 1 / 13]],
        [[3 / 26, 3 / 13, 3 / 26], [3 / 13, -44 / 13, 3 / 13], [3 / 26, 3 / 13, 3 / 26]],
        [[1 / 13, 3 / 26, 1 / 13], [3 / 26, 3 / 13, 3 / 26], [1 / 13, 3 / 26, 1 / 13]],
    ]]]
)


def masked_mean(values, mask):
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def masked_l1(pred, target, mask):
    return masked_mean((pred - target).abs(), mask)


def calculate_nrmse(pred, target):
    return torch.linalg.vector_norm(pred - target) / torch.linalg.vector_norm(target).clamp_min(1e-8)


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
    weight = W * mask
    return (weight * residual.square()).sum() / weight.sum().clamp_min(1e-6)


def hybrid_qsm_loss(chi_pred, chi_gt, phi_pred, phi_gt, phase, mask, W, D,
                    lam_chi=LAM_CHI, lam_phi=LAM_PHI, lam_grad=LAM_GRAD,
                    lam_wh=LAM_WH, lam_data=LAM_DATA):
    loss_chi = masked_l1(chi_pred, chi_gt, mask)
    loss_phi = masked_l1(phi_pred, phi_gt, mask)
    loss_grad = gradient_loss(chi_pred, chi_gt, mask)
    loss_wh = weak_harmonic_loss(phi_pred, mask)
    loss_data = data_consistency_loss(chi_pred, phi_pred, phase, mask, W, D)
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
        indices = list(range(count))
    else:
        indices = sorted(set(torch.linspace(1, count - 1, DEEP_SUP_K).round().int().tolist()))

    weights = torch.arange(1, len(indices) + 1, device=chi_gt.device, dtype=chi_gt.dtype)
    weights = weights / weights.sum()
    total = chi_gt.new_zeros(())
    parts_last = None
    for weight, index in zip(weights, indices):
        chi_k, phi_k = iterates[index]
        loss_k, parts = hybrid_qsm_loss(
            chi_k, chi_gt, phi_k, phi_gt, phase, mask, W, D, lam_wh=lam_wh
        )
        total = total + weight * loss_k
        parts_last = parts
    return total, parts_last


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
    noise = torch.randn(local.shape, generator=generator, dtype=local.dtype)
    noise = noise * (field_peak / COSMOS_SNR)
    phase = mask * (local + phi_h + noise)
    return _external_case("cosmos", phase, mask, mask, D, chi_target, COSMOS_FACTOR)


def build_sim2_evaluation():
    """Reproduce the crop, centered padding, and weighting in test_sim2.py."""
    mask_raw = loadmat("mask_final.mat")["mask_final"]
    phase_raw = loadmat("Sim2.mat")["phase"]
    W_raw = loadmat("sim2_w.mat")["w"]
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
        "sim2_w.mat", "Sim2ChiGT.nii.gz",
    ]
    missing = [path for path in required if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            "External epoch-end evaluation requires: " + ", ".join(missing)
        )
    return [build_cosmos_evaluation(), build_sim2_evaluation()]


@torch.no_grad()
def evaluate_external_case(model, case, device):
    """Run one fixed simulation and return a CPU result suitable for Comet."""
    was_training = model.training
    model.eval()
    phase = case["phase"].to(device, non_blocking=True)
    mask = case["mask"].to(device, non_blocking=True)
    W = case["W"].to(device, non_blocking=True)
    D = case["D"].to(device, non_blocking=True)
    chi_target = case["chi_target"].to(device, non_blocking=True)
    with amp_context(device):
        chi_pred, _ = model(phase, mask, D, W, tol=EVAL_TOL, max_iters=EVAL_MAX_ITERS)
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


def evaluate_and_log_external_cases(experiment, model, cases, device, epoch, step):
    metrics = {}
    for case in cases:
        result = evaluate_external_case(model, case, device)
        metric_name = f"external_{result['name']}_nrmse_chi"
        experiment.log_metric(metric_name, result["nrmse"], step=step)
        log_external_result(experiment, result, epoch)
        metrics[result["name"]] = result["nrmse"]
    return metrics


def iter_upper_for_epoch(epoch):
    fraction = min(1.0, epoch / max(1, ITER_RAMP_EPOCHS))
    return int(round(ITER_START + (ITER_SAMPLE_MAX - ITER_START) * fraction))


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
    return autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    )


@torch.no_grad()
def evaluate(model, loader, device):
    """Deterministic validation at the deployment stopping criterion."""
    was_training = model.training
    model.eval()
    nrmse_chi, nrmse_phi, count = 0.0, 0.0, 0
    for phase, mask, W, D, chi_gt, phi_gt in loader:
        phase = phase.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        W = W.to(device, non_blocking=True)
        D = D.to(device, non_blocking=True)
        chi_gt = chi_gt.to(device, non_blocking=True)
        phi_gt = phi_gt.to(device, non_blocking=True)
        with amp_context(device):
            chi_pred, phi_pred = model(
                phase, mask, D, W, tol=EVAL_TOL, max_iters=EVAL_MAX_ITERS
            )
        nrmse_chi += calculate_nrmse(chi_pred * mask, chi_gt * mask).item()
        nrmse_phi += calculate_nrmse(phi_pred * mask, phi_gt * mask).item()
        count += 1
    model.train(was_training)
    return nrmse_chi / max(1, count), nrmse_phi / max(1, count)


if __name__ == "__main__":
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
        "mode": "restored_unit_penalty_admm",
        "init_ckpt": str(INIT_CKPT),
        "ckpt_dir": CKPT_DIR,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "data_device": str(data_device),
        "loader_workers": loader_workers,
        "peak_lr": PEAK_LR,
        "min_lr_factor": MIN_LR_FACTOR,
        "warmup_epochs": WARMUP_EPOCHS,
        "wh_warmup_epochs": WH_WARMUP_EPOCHS,
        "lam_chi": LAM_CHI,
        "lam_phi": LAM_PHI,
        "lam_grad": LAM_GRAD,
        "lam_wh": LAM_WH,
        "lam_data": LAM_DATA,
        "iter_start": ITER_START,
        "iter_min": ITER_MIN,
        "iter_sample_max": ITER_SAMPLE_MAX,
        "iter_ramp_epochs": ITER_RAMP_EPOCHS,
        "deep_sup_k": DEEP_SUP_K,
        "eval_tol": EVAL_TOL,
        "eval_max_iters": EVAL_MAX_ITERS,
        "ema_decay": EMA_DECAY,
        "val_frac": VAL_FRAC,
        "source_id_modulus": SOURCE_ID_MODULUS,
        "background_scale": BACKGROUND_SCALE,
        "tv_regularization": TV_REGULARIZATION,
        "tv_probability": TV_PROBABILITY,
        "weight_augmentation_probability": WEIGHT_AUGMENTATION_PROBABILITY,
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
        tv_regularization=TV_REGULARIZATION,
        tv_probability=TV_PROBABILITY,
        weight_augmentation_probability=WEIGHT_AUGMENTATION_PROBABILITY,
        device=data_device,
    )
    val_full = QSMDataset(
        training=False,
        seed=SEED + 10_000,
        source_id_modulus=SOURCE_ID_MODULUS,
        background_scale=BACKGROUND_SCALE,
        tv_regularization=TV_REGULARIZATION,
        tv_probability=TV_PROBABILITY,
        weight_augmentation_probability=WEIGHT_AUGMENTATION_PROBABILITY,
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

    net_chi = ProximalNetwork().to(device)
    net_phi = ProximalNetwork().to(device)
    model = ADMMUnrolledNet(
        net_chi,
        net_phi,
        num_iters=ITER_START,
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
        print("Training restored unit-penalty ADMM model from scratch")

    torch.save(
        {
            "solver": "restored unit-penalty ADMM",
            "eps": model.eps,
            "initial_num_iters": model.num_iters,
            "penalties": {"y": 1.0, "u": 1.0, "v": 1.0},
            "conditioning": "raw iteration index and update RMS",
            "chi_update": "unit-penalty Fourier closed form followed by ROI masking",
            "weight_semantics": "W is a non-negative data-fidelity weight",
        },
        os.path.join(CKPT_DIR, "solver_config.pth"),
    )

    optimizer = optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=1e-5)
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
    ema = EMA(model, EMA_DECAY)
    global_step = 0
    best_score = float("inf")
    model.train()

    for epoch in range(EPOCHS):
        wh_factor = min(1.0, (epoch + 1) / max(1, WH_WARMUP_EPOCHS))
        iter_upper = iter_upper_for_epoch(epoch)
        skipped = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} (iters<={iter_upper})")

        for phase, mask, W, D, chi_gt, phi_gt in pbar:
            model.num_iters = int(
                torch.randint(ITER_MIN, iter_upper + 1, (1,), generator=loader_generator).item()
            )
            phase = phase.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            W = W.to(device, non_blocking=True)
            D = D.to(device, non_blocking=True)
            chi_gt = chi_gt.to(device, non_blocking=True)
            phi_gt = phi_gt.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with amp_context(device):
                iterates = model(phase, mask, D, W, return_iterates=True)
                loss, parts = deep_supervised_loss(
                    iterates, chi_gt, phi_gt, phase, mask, W, D,
                    lam_wh=LAM_WH * wh_factor,
                )
                chi_pred, phi_pred = iterates[-1]

            if not torch.isfinite(loss):
                skipped += 1
                continue

            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None:
                    torch.nan_to_num_(parameter.grad, nan=0.0, posinf=0.0, neginf=0.0)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            ema.update(model)

            with torch.no_grad():
                nrmse_chi = calculate_nrmse(chi_pred * mask, chi_gt * mask)
                nrmse_phi = calculate_nrmse(phi_pred * mask, phi_gt * mask)

            experiment.log_metric("loss", loss.item(), step=global_step)
            experiment.log_metric("nrmse_chi", nrmse_chi.item(), step=global_step)
            experiment.log_metric("nrmse_phi", nrmse_phi.item(), step=global_step)
            experiment.log_metric("lr", scheduler.get_last_lr()[0], step=global_step)
            experiment.log_metric("grad_norm", grad_norm.item(), step=global_step)
            experiment.log_metric("num_iters", model.num_iters, step=global_step)
            experiment.log_metric("ema_decay", ema.current_decay, step=global_step)
            for name, value in parts.items():
                experiment.log_metric(name, value, step=global_step)

            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                chi=f"{nrmse_chi.item():.3f}",
                phi=f"{nrmse_phi.item():.3f}",
                it=model.num_iters,
                lr=f"{scheduler.get_last_lr()[0]:.1e}",
            )
            global_step += 1

        experiment.log_metric("skipped_frac", skipped / max(1, len(train_loader)), step=global_step)

        backup = {key: value.detach().clone() for key, value in model.state_dict().items()}
        model.load_state_dict(ema.shadow, strict=True)
        model.num_iters = EVAL_MAX_ITERS
        val_chi, val_phi = evaluate(model, val_loader, device)
        external_metrics = evaluate_and_log_external_cases(
            experiment, model, external_cases, device, epoch + 1, global_step
        )
        model.load_state_dict(backup, strict=True)
        model.train()

        experiment.log_metric("val_nrmse_chi", val_chi, step=global_step)
        experiment.log_metric("val_nrmse_phi", val_phi, step=global_step)
        experiment.log_metric("wh_factor", wh_factor, step=global_step)
        for name, value in external_metrics.items():
            print(f"[epoch {epoch + 1}] external {name} NRMSE: {value:.4f}")

        score = val_chi + VAL_PHI_WEIGHT * val_phi
        if score < best_score:
            best_score = score
            torch.save(ema.shadow, os.path.join(CKPT_DIR, "model_best.pth"))
        torch.save(model.state_dict(), os.path.join(CKPT_DIR, "model_last.pth"))

        if (epoch + 1) % 2 == 0:
            model.load_state_dict(ema.shadow, strict=True)
            model.eval()
            with torch.no_grad(), amp_context(device):
                chi_pred, phi_pred = model(
                    phase, mask, D, W, tol=EVAL_TOL, max_iters=EVAL_MAX_ITERS
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

    model.load_state_dict(
        torch.load(os.path.join(CKPT_DIR, "model_best.pth"), map_location=device, weights_only=True),
        strict=True,
    )
    val_chi, val_phi = evaluate(model, val_loader, device)
    print(f"[BEST EMA] Validation NRMSE chi: {val_chi:.4f} | phi_h: {val_phi:.4f}")
    experiment.end()

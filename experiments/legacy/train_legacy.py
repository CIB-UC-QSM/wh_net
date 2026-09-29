# %%
from __future__ import annotations

from comet_ml import Experiment
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, TypeAlias, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from whnet.data import QSMDataset
from whnet.model import ADMMUnrolledNet, ProximalNetwork
from whnet.paths import CHECKPOINTS_DIR, TRAINING_DATA_DIR

# ----------------------------- Configuracion -----------------------------
# --- Entrenamiento DESDE CERO (sin warm-start) ---
INIT_CKPT = CHECKPOINTS_DIR / "checkpoints_scratch6" / "model_best.pth"  # None = desde cero. La normalizacion espectral
# (parametrizations.weight.original + buffers _u/_v)
# y el MLP de rho cambian el state_dict: los
# checkpoints previos NO cargan con strict=True.
CKPT_DIR = CHECKPOINTS_DIR / "checkpoints_scratch7"  # carpeta para esta corrida

EPOCHS = 50
BATCH_SIZE = 5
NUM_WORKERS = 5
PEAK_LR = 1e-3  # subido: 0.5e-4 era demasiado bajo para mover el warm-start
MIN_LR_FACTOR = 0.001  # LR final = PEAK_LR * MIN_LR_FACTOR (coseno)
WARMUP_EPOCHS = 4  # warmup corto: evita que el primer paso desestabilice el warm-start
WH_WARMUP_EPOCHS = 4  # mas largo: el modelo aun no aprendio el termino WH
LAM_WH = 100000.0  # BAJADO: para ENFATIZAR chi, lam_chi debe dominar a lam_wh

# --- Iteraciones del ADMM desenrollado (NIVEL 1: robustez a la profundidad) ---
NUM_ITERS_MAX = 60  # objetivo de produccion (referencia)
ITER_START = 50  # warm-start de un modelo ya entrenado a 30 -> se mantiene 30
ITER_MIN = 50  # cota INFERIOR del muestreo aleatorio por batch
ITER_SAMPLE_MAX = 60  # cota SUPERIOR del muestreo: un poco MAS ALLA del objetivo (30)
ITER_RAMP_EPOCHS = 10  # epocas para subir la cota de ITER_START a ITER_SAMPLE_MAX
DEEP_SUP_K = 10  # nº de iterados intermedios supervisados (supervision profunda)

# --- Inferencia / produccion: iteracion adaptativa con criterio de parada ---
EVAL_TOL = 1e-3  # parar cuando ||chi_k - chi_{k-1}|| / ||chi_k|| < tol
EVAL_MAX_ITERS = 55  # tope de seguridad (puede exceder NUM_ITERS_MAX)

GRAD_CLIP = 1.0  # el clip ACOTA el paso de los batches dificiles
EMA_DECAY = 0.9  # promedio movil exponencial de peso
VAL_FRAC = 0.1  # fraccion para validacion (split por volumen)
SEED = 0
# --------------------------------------------------------------------------


Batch: TypeAlias = tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
Iterate: TypeAlias = tuple[Tensor, Tensor]
LossParts: TypeAlias = dict[str, float]
TensorLossParts: TypeAlias = dict[str, Tensor]
StateDict: TypeAlias = Mapping[str, Tensor]


_LAP_KERNEL = torch.tensor(
    [
        [
            [
                [
                    [1 / 13, 3 / 26, 1 / 13],
                    [3 / 26, 3 / 13, 3 / 26],
                    [1 / 13, 3 / 26, 1 / 13],
                ],
                [
                    [3 / 26, 3 / 13, 3 / 26],
                    [3 / 13, -44 / 13, 3 / 13],
                    [3 / 26, 3 / 13, 3 / 26],
                ],
                [
                    [1 / 13, 3 / 26, 1 / 13],
                    [3 / 26, 3 / 13, 3 / 26],
                    [1 / 13, 3 / 26, 1 / 13],
                ],
            ]
        ]
    ],
    dtype=torch.float32,
)


def _validate_same_shape(*tensors: Tensor) -> None:
    shapes = {tuple(tensor.shape) for tensor in tensors}
    if len(shapes) != 1:
        raise ValueError(
            f"Expected tensors with identical shapes, got {sorted(shapes)}"
        )


def calculate_nrmse(pred: Tensor, target: Tensor) -> Tensor:
    """Return a finite, scalar NRMSE for equally shaped, non-empty tensors."""
    _validate_same_shape(pred, target)
    if pred.numel() == 0:
        raise ValueError("NRMSE is undefined for empty tensors")
    if not pred.is_floating_point() or not target.is_floating_point():
        pred = pred.float()
        target = target.float()

    numerator = torch.linalg.vector_norm((pred - target).reshape(-1))
    denominator = torch.linalg.vector_norm(target.reshape(-1))
    epsilon = torch.finfo(denominator.dtype).eps
    return numerator / denominator.clamp_min(epsilon)


def spatial_gradient_3d(x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    if x.ndim != 5:
        raise ValueError(f"Expected a 5-D [N, C, D, H, W] tensor, got shape {x.shape}")
    if any(size < 2 for size in x.shape[-3:]):
        raise ValueError(
            f"All spatial dimensions must be at least 2, got {x.shape[-3:]}"
        )

    dx = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
    dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
    dz = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
    return dx, dy, dz


def gradient_loss(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    _validate_same_shape(pred, target, mask)
    px, py, pz = spatial_gradient_3d(pred)
    tx, ty, tz = spatial_gradient_3d(target)
    mask_x = mask[:, :, 1:, :, :] * mask[:, :, :-1, :, :]
    mask_y = mask[:, :, :, 1:, :] * mask[:, :, :, :-1, :]
    mask_z = mask[:, :, :, :, 1:] * mask[:, :, :, :, :-1]
    return (
        F.l1_loss(px * mask_x, tx * mask_x)
        + F.l1_loss(py * mask_y, ty * mask_y)
        + F.l1_loss(pz * mask_z, tz * mask_z)
    )


def _weak_harmonic_loss(phi: Tensor, mask: Tensor, kernel: Tensor) -> Tensor:
    laplacian_phi = F.conv3d(phi, kernel, padding=1)
    return F.smooth_l1_loss(
        mask * laplacian_phi,
        torch.zeros_like(laplacian_phi),
        beta=0.1,
    )


def weak_harmonic_loss(phi: Tensor, mask: Tensor) -> Tensor:
    _validate_same_shape(phi, mask)
    kernel = _LAP_KERNEL.to(device=phi.device, dtype=phi.dtype)
    return _weak_harmonic_loss(phi, mask, kernel)


def _hybrid_qsm_loss_tensors(
    chi_pred: Tensor,
    chi_gt: Tensor,
    phi_pred: Tensor,
    phi_gt: Tensor,
    mask: Tensor,
    lam_chi: float,
    lam_phi: float,
    lam_grad: float,
    lam_wh: float,
    laplacian_kernel: Tensor | None = None,
) -> tuple[Tensor, TensorLossParts]:
    _validate_same_shape(chi_pred, chi_gt, phi_pred, phi_gt, mask)
    loss_chi = F.l1_loss(chi_pred * mask, chi_gt * mask)
    loss_phi = F.l1_loss(phi_pred * mask, phi_gt * mask)
    loss_grad = gradient_loss(chi_pred, chi_gt, mask)
    kernel = (
        _LAP_KERNEL.to(device=phi_pred.device, dtype=phi_pred.dtype)
        if laplacian_kernel is None
        else laplacian_kernel
    )
    loss_wh = _weak_harmonic_loss(phi_pred, mask, kernel)
    total = (
        lam_chi * loss_chi
        + lam_phi * loss_phi
        + lam_grad * loss_grad
        + lam_wh * loss_wh
    )
    return total, {
        "loss_chi": lam_chi * loss_chi,
        "loss_phi": lam_phi * loss_phi,
        "loss_grad": lam_grad * loss_grad,
        "loss_wh": lam_wh * loss_wh,
    }


# def hybrid_qsm_loss(
#     chi_pred: Tensor,
#     chi_gt: Tensor,
#     phi_pred: Tensor,
#     phi_gt: Tensor,
#     mask: Tensor,
#     lam_chi: float = 150.0,
#     lam_phi: float = 100.0,
#     lam_grad: float = 1.0,
#     lam_wh: float = 10.0,
# ) -> tuple[Tensor, LossParts]:
#     total, tensor_parts = _hybrid_qsm_loss_tensors(
#         chi_pred,
#         chi_gt,
#         phi_pred,
#         phi_gt,
#         mask,
#         lam_chi,
#         lam_phi,
#         lam_grad,
#         lam_wh,
#     )
#     return total, {name: value.detach().item() for name, value in tensor_parts.items()}


def deep_supervised_loss(
    iterates: Sequence[Iterate],
    chi_gt: Tensor,
    phi_gt: Tensor,
    mask: Tensor,
    lam_wh: float,
) -> tuple[Tensor, LossParts]:
    """Penalize evenly spaced iterates, weighted toward the deepest output."""
    num_iterates = len(iterates)
    if num_iterates == 0:
        raise ValueError("deep_supervised_loss requires at least one iterate")
    if DEEP_SUP_K < 1:
        raise ValueError(f"DEEP_SUP_K must be at least 1, got {DEEP_SUP_K}")

    if num_iterates <= DEEP_SUP_K:
        indices = list(range(num_iterates))
    else:
        indices = sorted(
            set(torch.linspace(1, num_iterates - 1, DEEP_SUP_K).round().int().tolist())
        )

    weights = [position + 1 for position in range(len(indices))]
    weight_sum = float(sum(weights))
    laplacian_kernel = _LAP_KERNEL.to(device=phi_gt.device, dtype=phi_gt.dtype)
    total: Tensor | None = None
    last_parts: TensorLossParts | None = None

    for weight, index in zip(weights, indices, strict=True):
        chi_k, phi_k = iterates[index]
        loss_k, parts = _hybrid_qsm_loss_tensors(
            chi_k,
            chi_gt,
            phi_k,
            phi_gt,
            mask,
            lam_chi=10.0,
            lam_phi=40.0,
            lam_grad=1.0,
            lam_wh=lam_wh,
            laplacian_kernel=laplacian_kernel,
        )
        weighted_loss = (weight / weight_sum) * loss_k
        total = weighted_loss if total is None else total + weighted_loss
        last_parts = parts

    if total is None or last_parts is None:  # Defensive; indices cannot be empty here.
        raise RuntimeError("Failed to select an iterate for deep supervision")

    return total, {name: value.detach().item() for name, value in last_parts.items()}


class EMA:
    """Exponential moving average of parameters, with direct buffer copies."""

    def __init__(self, model: nn.Module, decay: float):
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"EMA decay must be in [0, 1), got {decay}")
        self.decay = decay
        self.param_names = {name for name, _ in model.named_parameters()}
        self.shadow = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        state = model.state_dict()
        self._validate_state(state)
        for key, value in state.items():
            if key in self.param_names and value.dtype.is_floating_point:
                self.shadow[key].mul_(self.decay).add_(
                    value.detach(), alpha=1.0 - self.decay
                )
            else:
                # Spectral-normalization vectors and other buffers use their latest value.
                self.shadow[key].copy_(value)

    def _validate_state(self, state: StateDict) -> None:
        if state.keys() != self.shadow.keys():
            missing = sorted(self.shadow.keys() - state.keys())
            unexpected = sorted(state.keys() - self.shadow.keys())
            raise ValueError(
                f"EMA/model state mismatch; missing={missing}, unexpected={unexpected}"
            )
        for key, value in state.items():
            shadow_value = self.shadow[key]
            if value.shape != shadow_value.shape or value.dtype != shadow_value.dtype:
                raise ValueError(
                    f"EMA/model tensor mismatch for {key!r}: "
                    f"model={value.shape}/{value.dtype}, "
                    f"shadow={shadow_value.shape}/{shadow_value.dtype}"
                )

    @torch.no_grad()
    def _swap_state(self, model: nn.Module) -> None:
        state = model.state_dict()
        self._validate_state(state)
        for key, value in state.items():
            temporary = value.detach().clone()
            value.copy_(self.shadow[key])
            self.shadow[key].copy_(temporary)

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        """Temporarily install EMA state and restore both states even on failure."""
        self._swap_state(model)
        try:
            yield
        finally:
            self._swap_state(model)


class DeterministicDataset(Dataset[Batch]):
    """Make stochastic dataset transforms repeatable for validation."""

    def __init__(self, dataset: Dataset[Batch], seed: int):
        self.dataset = dataset
        self.seed = seed

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Batch:
        # QSMDataset uses the CPU torch RNG for all augmentation and noise generation.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed + index)
            return self.dataset[index]


def log_ortho_slices(
    experiment: Experiment,
    vol: np.ndarray,
    name: str,
    epoch: int,
    rango: tuple[float, float] = (-0.1, 0.1),
) -> None:
    if vol.ndim != 3 or any(size == 0 for size in vol.shape):
        raise ValueError(f"Expected a non-empty 3-D volume, got shape {vol.shape}")
    if rango[0] >= rango[1]:
        raise ValueError(f"Invalid display range {rango}; expected min < max")

    depth, height, width = vol.shape
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    try:
        slices = (
            vol[depth // 2],
            vol[:, height // 2],
            vol[:, :, width // 2],
        )
        for axis, image_slice in zip(axes, slices, strict=True):
            axis.imshow(image_slice, cmap="gray", vmin=rango[0], vmax=rango[1])
            axis.axis("off")
        fig.tight_layout()
        experiment.log_figure(figure_name=f"{name}_epoch_{epoch}", figure=fig)
    finally:
        plt.close(fig)


def iter_upper_for_epoch(epoch: int) -> int:
    """Return the linearly ramped upper iteration bound for a zero-based epoch."""
    if epoch < 0:
        raise ValueError(f"epoch must be non-negative, got {epoch}")
    if ITER_RAMP_EPOCHS <= 0:
        return ITER_SAMPLE_MAX
    fraction = min(1.0, epoch / ITER_RAMP_EPOCHS)
    return int(round(ITER_START + (ITER_SAMPLE_MAX - ITER_START) * fraction))


def _autocast_context(device: torch.device) -> AbstractContextManager[Any]:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def _move_batch(batch: Sequence[Tensor], device: torch.device) -> Batch:
    if len(batch) != 6:
        raise ValueError(f"Expected a six-tensor batch, got {len(batch)} elements")
    non_blocking = device.type == "cuda"
    moved = tuple(tensor.to(device, non_blocking=non_blocking) for tensor in batch)
    return cast(Batch, moved)


def _nrmse_from_squared_sums(error_squared: Tensor, target_squared: Tensor) -> float:
    denominator = target_squared.sqrt()
    epsilon = torch.finfo(denominator.dtype).eps
    return (error_squared.sqrt() / denominator.clamp_min(epsilon)).item()


@torch.no_grad()
def evaluate(
    model: ADMMUnrolledNet,
    loader: Iterable[Batch],
    device: torch.device | str,
) -> tuple[float, float]:
    """Evaluate adaptive inference and restore the model's original mode afterward."""
    resolved_device = torch.device(device)
    was_training = model.training
    model.eval()
    chi_error_squared: Tensor | None = None
    chi_target_squared: Tensor | None = None
    phi_error_squared: Tensor | None = None
    phi_target_squared: Tensor | None = None
    num_batches = 0

    try:
        for batch in loader:
            phase, mask, weights, dipole, chi_gt, phi_gt = _move_batch(
                batch, resolved_device
            )
            with _autocast_context(resolved_device):
                chi_pred, phi_pred = model(
                    phase,
                    mask,
                    dipole,
                    weights,
                    tol=EVAL_TOL,
                    max_iters=EVAL_MAX_ITERS,
                )

            chi_difference = (chi_pred - chi_gt) * mask
            masked_chi_gt = chi_gt * mask
            phi_difference = (phi_pred - phi_gt) * mask
            masked_phi_gt = phi_gt * mask

            batch_chi_error = chi_difference.square().sum(dtype=torch.float64)
            batch_chi_target = masked_chi_gt.square().sum(dtype=torch.float64)
            batch_phi_error = phi_difference.square().sum(dtype=torch.float64)
            batch_phi_target = masked_phi_gt.square().sum(dtype=torch.float64)

            chi_error_squared = (
                batch_chi_error
                if chi_error_squared is None
                else chi_error_squared + batch_chi_error
            )
            chi_target_squared = (
                batch_chi_target
                if chi_target_squared is None
                else chi_target_squared + batch_chi_target
            )
            phi_error_squared = (
                batch_phi_error
                if phi_error_squared is None
                else phi_error_squared + batch_phi_error
            )
            phi_target_squared = (
                batch_phi_target
                if phi_target_squared is None
                else phi_target_squared + batch_phi_target
            )
            num_batches += 1
    finally:
        model.train(was_training)

    if num_batches == 0:
        raise ValueError("Validation loader produced no batches")
    if any(
        value is None
        for value in (
            chi_error_squared,
            chi_target_squared,
            phi_error_squared,
            phi_target_squared,
        )
    ):
        raise RuntimeError("Validation metric accumulation failed")

    return (
        _nrmse_from_squared_sums(
            cast(Tensor, chi_error_squared), cast(Tensor, chi_target_squared)
        ),
        _nrmse_from_squared_sums(
            cast(Tensor, phi_error_squared), cast(Tensor, phi_target_squared)
        ),
    )


def _validate_configuration() -> None:
    errors: list[str] = []
    if EPOCHS < 1:
        errors.append("EPOCHS must be at least 1")
    if BATCH_SIZE < 1:
        errors.append("BATCH_SIZE must be at least 1")
    if NUM_WORKERS < 0:
        errors.append("NUM_WORKERS must be non-negative")
    if PEAK_LR <= 0:
        errors.append("PEAK_LR must be positive")
    if not 0 < MIN_LR_FACTOR <= 1:
        errors.append("MIN_LR_FACTOR must be in (0, 1]")
    if not 0 <= WARMUP_EPOCHS < EPOCHS:
        errors.append("WARMUP_EPOCHS must be in [0, EPOCHS)")
    if WH_WARMUP_EPOCHS < 0:
        errors.append("WH_WARMUP_EPOCHS must be non-negative")
    if LAM_WH < 0:
        errors.append("LAM_WH must be non-negative")
    if GRAD_CLIP <= 0:
        errors.append("GRAD_CLIP must be positive")
    if not 0 <= EMA_DECAY < 1:
        errors.append("EMA_DECAY must be in [0, 1)")
    if not 0 < VAL_FRAC < 1:
        errors.append("VAL_FRAC must be in (0, 1)")
    if NUM_ITERS_MAX < 1:
        errors.append("NUM_ITERS_MAX must be at least 1")
    if ITER_MIN < 1:
        errors.append("ITER_MIN must be at least 1")
    if ITER_START < ITER_MIN:
        errors.append("ITER_START must be greater than or equal to ITER_MIN")
    if ITER_SAMPLE_MAX < ITER_START:
        errors.append("ITER_SAMPLE_MAX must be greater than or equal to ITER_START")
    if ITER_RAMP_EPOCHS < 0:
        errors.append("ITER_RAMP_EPOCHS must be non-negative")
    if DEEP_SUP_K < 1:
        errors.append("DEEP_SUP_K must be at least 1")
    if EVAL_TOL <= 0:
        errors.append("EVAL_TOL must be positive")
    if EVAL_MAX_ITERS < 1:
        errors.append("EVAL_MAX_ITERS must be at least 1")
    if errors:
        raise ValueError("Invalid training configuration:\n- " + "\n- ".join(errors))


def _learning_rate_multiplier(
    step: int,
    warmup_steps: int,
    total_steps: int,
) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = min(
        1.0,
        max(0.0, (step - warmup_steps) / max(1, total_steps - warmup_steps)),
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LR_FACTOR + (1.0 - MIN_LR_FACTOR) * cosine


def _load_state_dict(path: Path, device: torch.device) -> StateDict:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    loaded = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(loaded, Mapping):
        raise TypeError(f"Checkpoint {path} does not contain a state mapping")
    if not all(
        isinstance(key, str) and isinstance(value, Tensor)
        for key, value in loaded.items()
    ):
        raise TypeError(f"Checkpoint {path} is not a tensor state_dict")
    return cast(StateDict, loaded)


def _atomic_torch_save(state: StateDict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)

    try:
        torch.save(state, temporary_path)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _log_run_parameters(experiment: Experiment) -> None:
    experiment.log_parameters(
        {
            "mode": "from_scratch" if INIT_CKPT is None else "warm_start",
            "init_ckpt": str(INIT_CKPT),
            "ckpt_dir": CKPT_DIR,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "peak_lr": PEAK_LR,
            "min_lr_factor": MIN_LR_FACTOR,
            "warmup_epochs": WARMUP_EPOCHS,
            "wh_warmup_epochs": WH_WARMUP_EPOCHS,
            "lam_wh": LAM_WH,
            "grad_clip": GRAD_CLIP,
            "ema_decay": EMA_DECAY,
            "num_iters_max": NUM_ITERS_MAX,
            "iter_start": ITER_START,
            "iter_min": ITER_MIN,
            "iter_sample_max": ITER_SAMPLE_MAX,
            "iter_ramp_epochs": ITER_RAMP_EPOCHS,
            "deep_sup_k": DEEP_SUP_K,
            "eval_tol": EVAL_TOL,
            "eval_max_iters": EVAL_MAX_ITERS,
            "seed": SEED,
        }
    )


def _run_training(experiment: Experiment) -> None:
    checkpoint_dir = Path(CKPT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = device.type == "cuda"

    net_chi = ProximalNetwork().to(device)
    net_phi = ProximalNetwork().to(device)
    model = ADMMUnrolledNet(net_chi, net_phi, num_iters=ITER_START).to(device)

    if INIT_CKPT is not None:
        initial_checkpoint = Path(INIT_CKPT)
        model.load_state_dict(
            _load_state_dict(initial_checkpoint, device),
            strict=True,
        )
        print(f"Pesos iniciales cargados desde {initial_checkpoint}")
    else:
        print("Entrenamiento desde cero (sin warm-start)")

    optimizer = optim.Adam(model.parameters(), lr=PEAK_LR)

    full_dataset = QSMDataset(TRAINING_DATA_DIR)
    dataset_size = len(full_dataset)
    if dataset_size < 2:
        raise ValueError(
            f"QSMDataset requires at least two volumes for train/validation, got {dataset_size}"
        )

    num_validation = min(dataset_size - 1, max(1, int(dataset_size * VAL_FRAC)))
    num_training = dataset_size - num_validation
    train_set, stochastic_val_set = random_split(
        full_dataset,
        [num_training, num_validation],
        generator=torch.Generator().manual_seed(SEED),
    )
    val_set = DeterministicDataset(stochastic_val_set, seed=SEED)

    pin_memory = device.type == "cuda"
    loader_options = {
        "num_workers": NUM_WORKERS,
        "pin_memory": pin_memory,
        "persistent_workers": NUM_WORKERS > 0,
    }
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        **loader_options,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        **loader_options,
    )

    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise ValueError("Training loader produced no batches")
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = steps_per_epoch * WARMUP_EPOCHS

    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _learning_rate_multiplier(step, warmup_steps, total_steps),
    )
    ema = EMA(model, EMA_DECAY)

    global_step = 0
    best_val = math.inf
    best_checkpoint = checkpoint_dir / "model_best.pth"
    model.train()

    for epoch in range(EPOCHS):
        wh_factor = 1.0 if WH_WARMUP_EPOCHS == 0 else min(1.0, epoch / WH_WARMUP_EPOCHS)
        iteration_upper_bound = iter_upper_for_epoch(epoch)
        skipped = 0
        last_batch: Batch | None = None

        description = f"Epoch {epoch + 1}/{EPOCHS} (iters<={iteration_upper_bound})"
        with tqdm(train_loader, desc=description) as progress_bar:
            for cpu_batch in progress_bar:
                model.num_iters = int(
                    torch.randint(ITER_MIN, iteration_upper_bound + 1, (1,)).item()
                )
                batch = _move_batch(cpu_batch, device)
                phase, mask, weights, dipole, chi_gt, phi_gt = batch
                last_batch = batch

                optimizer.zero_grad(set_to_none=True)
                with _autocast_context(device):
                    iterates = model(
                        phase,
                        mask,
                        dipole,
                        weights,
                        return_iterates=True,
                    )
                    loss, parts = deep_supervised_loss(
                        iterates,
                        chi_gt,
                        phi_gt,
                        mask,
                        lam_wh=LAM_WH * wh_factor,
                    )
                    chi_pred, phi_pred = iterates[-1]

                if not torch.isfinite(loss).item():
                    skipped += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue

                loss.backward()
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        torch.nan_to_num_(
                            parameter.grad,
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0,
                        )
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=GRAD_CLIP,
                )

                optimizer.step()
                scheduler.step()
                ema.update(model)

                with torch.no_grad():
                    nrmse_chi = calculate_nrmse(chi_pred * mask, chi_gt * mask)
                    nrmse_phi = calculate_nrmse(phi_pred * mask, phi_gt * mask)

                loss_value = loss.item()
                nrmse_chi_value = nrmse_chi.item()
                nrmse_phi_value = nrmse_phi.item()
                learning_rate = scheduler.get_last_lr()[0]
                grad_norm_value = grad_norm.item()
                experiment.log_metrics(
                    {
                        "loss": loss_value,
                        "nrmse_chi": nrmse_chi_value,
                        "nrmse_phi": nrmse_phi_value,
                        "lr": learning_rate,
                        "grad_norm": grad_norm_value,
                        "num_iters": model.num_iters,
                        **parts,
                    },
                    step=global_step,
                )

                progress_bar.set_postfix(
                    loss=f"{loss_value:.3f}",
                    nrmse_chi=f"{nrmse_chi_value:.3f}",
                    nrmse_phi=f"{nrmse_phi_value:.3f}",
                    it=model.num_iters,
                    lr=f"{learning_rate:.1e}",
                )
                global_step += 1

        if last_batch is None:
            raise RuntimeError("Training loader unexpectedly produced no batches")

        experiment.log_metric(
            "skipped_frac",
            skipped / steps_per_epoch,
            step=global_step,
        )

        with ema.average_parameters(model):
            val_chi, val_phi = evaluate(model, val_loader, device)

            if (epoch + 1) % 2 == 0:
                phase, mask, weights, dipole, chi_gt, phi_gt = last_batch
                model.eval()
                with torch.no_grad(), _autocast_context(device):
                    chi_pred, phi_pred = model(
                        phase,
                        mask,
                        dipole,
                        weights,
                        tol=EVAL_TOL,
                        max_iters=EVAL_MAX_ITERS,
                    )
                volumes = (
                    (phase, "phase_in"),
                    (chi_pred, "chi_pred"),
                    (phi_pred, "phih_pred"),
                    (chi_gt, "chi_gt"),
                    (phi_gt, "phih_gt"),
                )
                for volume, name in volumes:
                    image = (volume * mask).detach().cpu().numpy()[0, 0]
                    log_ortho_slices(experiment, image, name, epoch + 1)
                model.train()

        experiment.log_metrics(
            {
                "val_nrmse_chi": val_chi,
                "val_nrmse_phi": val_phi,
                "wh_factor": wh_factor,
            },
            step=global_step,
        )

        score = val_chi + val_phi
        if not math.isfinite(score):
            raise FloatingPointError(
                f"Validation score is not finite: chi={val_chi}, phi={val_phi}"
            )
        if score < best_val:
            best_val = score
            _atomic_torch_save(ema.shadow, best_checkpoint)

        _atomic_torch_save(model.state_dict(), checkpoint_dir / "model_last.pth")
        if (epoch + 1) % 10 == 0:
            _atomic_torch_save(
                model.state_dict(),
                checkpoint_dir / f"model_epoch_{epoch + 1}.pth",
            )

    model.load_state_dict(_load_state_dict(best_checkpoint, device), strict=True)
    val_chi, val_phi = evaluate(model, val_loader, device)
    print(f"[BEST EMA] Val NRMSE Chi: {val_chi:.4f} | Val NRMSE Phi: {val_phi:.4f}")


def main() -> None:
    _validate_configuration()
    experiment = Experiment(project_name="wh-net-qsm")
    try:
        _log_run_parameters(experiment)
        _run_training(experiment)
    finally:
        experiment.end()


if __name__ == "__main__":
    main()

# %%

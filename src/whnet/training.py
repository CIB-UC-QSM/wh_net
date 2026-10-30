# %%
"""Joint chi/background training with subject-level validation and resumable runs.

New checkpoints require load_trained_model(), which restores the physics and
conditioning configuration as well as the weights. Legacy state_dicts can be
used explicitly with --init-ckpt; they do not provide a clean held-out baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from collections import defaultdict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TypeAlias

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.amp import autocast
from torch.utils.data import DataLoader, Sampler

from .adaptive_wh_tv_validation import (
    ExternalValidationCase,
    ExternalValidationResult,
    _crop_tensor,
    comparison_figure,
    load_external_cases,
    masked_nrmse,
    physical_external_cases,
)
from .data import FieldSynthesizer, TrainingVolumes, training_records
from .model import EfficientADMMUnrolledNet, ProximalNetwork
from .paths import PROJECT_ROOT, RUNS_DIR, TRAINING_CACHE_DIR, TRAINING_DATA_DIR

LAM_WH = 1000
DEEP_SUP_K = 10
Batch: TypeAlias = tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
Iterate: TypeAlias = tuple[Tensor, Tensor]
StateDict: TypeAlias = Mapping[str, Tensor]
STOP_REQUESTED = False


@dataclass
class TrainConfig:
    data_dir: str = str(TRAINING_DATA_DIR)
    cache_dir: str = str(TRAINING_CACHE_DIR)
    source_manifest: str | None = None
    checkpoint_dir: str = ""
    init_ckpt: str | None = None
    resume: str | None = None
    epochs: int = 50
    batch_size: int = 5
    accumulation_steps: int = 1
    num_workers: int = 5
    peak_lr: float = 1e-4
    phi_lr_factor: float = 1.0
    min_lr_factor: float = 0.05
    warmup_epochs: int = 3
    wh_warmup_epochs: int = 5
    lam_chi: float = 1.0
    lam_chi_high: float = 3.0
    normalize_high_chi_loss: bool = True
    input_scale_floor: float = 1e-6
    lam_phi: float = 0.25
    lam_grad: float = 0.1
    lam_wh: float = LAM_WH
    deep_sup_k: int = DEEP_SUP_K
    final_weight: float = 0.7
    iter_start: int = 12
    iter_max: int = 55
    iter_ramp_epochs: int = 15
    final_iters: int = 55
    final_epochs: int = 10
    grad_clip: float = 1.0
    ema_decay: float = 0.99
    val_frac: float = 0.1
    val_views: int = 3
    seed: int = 0
    log_every: int = 1
    save_every: int = 1
    device: str = "cuda:0"
    memory_fraction: float = 0.99
    compile_proximal: bool = True
    channels_last_3d: bool = True
    comet: bool = False
    local_perturbations: bool = True
    chi_scale_mode: str = "rms_mixture"
    background_mode: str = "ratio"
    synthetic_fraction: float = 0.25
    external_validation: bool = True
    external_validation_every: int = 1
    cosmos_simulation: str | None = None
    sim2_factor: float = 1.0
    sim2_pad: int = 20
    # Bounded diagnostics only; zero means the full configured run.
    max_steps: int = 0
    max_train_volumes: int = 0
    max_val_subjects: int = 0

    def validate(self):
        positive = (
            "epochs",
            "batch_size",
            "accumulation_steps",
            "peak_lr",
            "phi_lr_factor",
            "min_lr_factor",
            "deep_sup_k",
            "iter_start",
            "iter_max",
            "final_iters",
            "grad_clip",
            "val_views",
            "log_every",
            "save_every",
            "external_validation_every",
            "sim2_factor",
        )
        if any(getattr(self, key) <= 0 for key in positive):
            raise ValueError(
                "Counts, learning rates and gradient clipping must be positive"
            )
        if self.num_workers < 0 or self.sim2_pad < 0 or not 0 < self.val_frac < 1:
            raise ValueError("Invalid workers or validation fraction")
        if not 0 < self.final_weight <= 1 or not 0 <= self.ema_decay < 1:
            raise ValueError("Invalid final supervision weight or EMA decay")
        if not 0 < self.memory_fraction <= 1 or not 0 < self.min_lr_factor <= 1:
            raise ValueError("Invalid memory fraction or LR floor")
        if not self.iter_start <= self.iter_max <= self.final_iters:
            raise ValueError("Require iter_start <= iter_max <= final_iters")
        if (
            min(
                self.warmup_epochs,
                self.wh_warmup_epochs,
                self.iter_ramp_epochs,
                self.final_epochs,
                self.max_steps,
                self.max_train_volumes,
                self.max_val_subjects,
            )
            < 0
        ):
            raise ValueError("Schedules and diagnostic limits cannot be negative")
        if (
            min(self.lam_chi, self.lam_phi) <= 0
            or min(self.lam_chi_high, self.lam_grad, self.lam_wh) < 0
        ):
            raise ValueError(
                "Chi and phi weights must be positive; other loss weights nonnegative"
            )
        if self.init_ckpt and self.resume:
            raise ValueError("Choose either warm-start or resume")
        if self.input_scale_floor <= 0:
            raise ValueError("Input scale floor must be positive")
        if self.chi_scale_mode not in ("legacy", "rms_mixture"):
            raise ValueError("Invalid chi scale mode")
        if self.background_mode not in ("legacy", "ratio"):
            raise ValueError("Invalid background mode")
        if self.synthetic_fraction != -1 and not 0 <= self.synthetic_fraction <= 1:
            raise ValueError("Synthetic fraction must be in [0, 1] or -1")


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


def _validate_same_shape(*tensors):
    if len({tuple(x.shape) for x in tensors}) != 1:
        raise ValueError("Expected tensors with identical shapes")


def calculate_nrmse(pred, target):
    _validate_same_shape(pred, target)
    if not pred.numel():
        raise ValueError("NRMSE is undefined for empty tensors")
    return (pred.float() - target.float()).norm() / target.float().norm().clamp_min(
        1e-8
    )


def _sample_mean(values, mask):
    dims = tuple(range(1, values.ndim))
    return (values * mask).sum(dim=dims) / mask.sum(dim=dims).clamp_min(1)


def _sample_nrmse(pred, target, mask):
    error = _sample_mean((pred.float() - target.float()).square(), mask)
    energy = _sample_mean(target.float().square(), mask).clamp_min(1e-8)
    return (error / energy).sqrt()


def spatial_gradient_3d(x):
    if x.ndim != 5 or min(x.shape[-3:]) < 2:
        raise ValueError("Expected [N,C,D,H,W] with spatial dimensions >= 2")
    return (
        x[:, :, 1:] - x[:, :, :-1],
        x[:, :, :, 1:] - x[:, :, :, :-1],
        x[..., 1:] - x[..., :-1],
    )


def gradient_loss(pred, target, mask):
    _validate_same_shape(pred, target, mask)
    valid = (
        mask[:, :, 1:] * mask[:, :, :-1],
        mask[:, :, :, 1:] * mask[:, :, :, :-1],
        mask[..., 1:] * mask[..., :-1],
    )
    return sum(
        _sample_mean((a - b).abs(), m).mean()
        for a, b, m in zip(
            spatial_gradient_3d(pred), spatial_gradient_3d(target), valid
        )
    )


def complete_stencil_mask(mask):
    if mask.ndim != 5 or mask.shape[1] != 1:
        raise ValueError("Expected a single-channel 5D mask")
    with torch.autocast(mask.device.type, enabled=False):
        binary = (mask > 0.5).float()
        count = F.conv3d(
            binary, torch.ones(1, 1, 3, 3, 3, device=mask.device), padding=1
        )
        interior = (count == 27).float()
    if bool((interior.sum(dim=(1, 2, 3, 4)) == 0).any()):
        raise ValueError("Every mask needs at least one complete harmonic stencil")
    return interior


def weak_harmonic_loss(phi, mask):
    _validate_same_shape(phi, mask)
    interior = complete_stencil_mask(mask)
    with torch.autocast(phi.device.type, enabled=False):
        laplacian = F.conv3d(phi.float(), _LAP_KERNEL.to(phi.device), padding=1)
        loss = F.smooth_l1_loss(
            laplacian, torch.zeros_like(laplacian), beta=0.1, reduction="none"
        )
        return _sample_mean(loss, interior).mean()


def supervision_indices(depth, count):
    if depth < 1 or count < 1:
        raise ValueError("Depth and supervision count must be positive")
    count = min(depth, count)
    if count == 1:
        return [depth - 1]
    return sorted(set(torch.linspace(0, depth - 1, count).round().int().tolist()))


def deep_supervised_loss(iterates, chi_gt, phi_gt, mask, lam_wh, config=None):
    """Inputs are selected iterates; the final output owns most of the loss."""
    cfg = config or TrainConfig()
    if not iterates:
        raise ValueError("No iterates for supervision")
    with torch.autocast(mask.device.type, enabled=False):
        mask, chi_gt, phi_gt = mask.float(), chi_gt.float(), phi_gt.float()
        chi_energy = _sample_mean(chi_gt.square(), mask).clamp_min(1e-8)
        phi_energy = _sample_mean(phi_gt.square(), mask).clamp_min(1e-6)
        gradient_masks = (
            mask[:, :, 1:] * mask[:, :, :-1],
            mask[:, :, :, 1:] * mask[:, :, :, :-1],
            mask[..., 1:] * mask[..., :-1],
        )
        target_gradients = spatial_gradient_3d(chi_gt)
        harmonic_mask = complete_stencil_mask(mask) if lam_wh else None
        high_weight = getattr(cfg, "lam_chi_high", 0.0)
        high_mask = None
        if high_weight:
            magnitude = chi_gt.abs()
            threshold = torch.quantile(
                magnitude.flatten(start_dim=1), 0.8, dim=1
            ).reshape(-1, 1, 1, 1, 1)
            high_mask = (magnitude > threshold).to(chi_gt.dtype)
        kernel = _LAP_KERNEL.to(mask.device)
        weights = (
            [1.0]
            if len(iterates) == 1
            else [(1 - cfg.final_weight) / (len(iterates) - 1)] * (len(iterates) - 1)
            + [cfg.final_weight]
        )
        total = mask.new_zeros(())
        names = ["loss_chi", "loss_phi", "loss_grad", "loss_wh"]
        if high_mask is not None:
            names.append("loss_chi_high")
        parts = {key: mask.new_zeros(()) for key in names}
        for weight, (chi, phi) in zip(weights, iterates):
            chi, phi = chi.float(), phi.float()
            chi_loss = (_sample_mean((chi - chi_gt).square(), mask) / chi_energy).mean()
            phi_loss = (_sample_mean((phi - phi_gt).square(), mask) / phi_energy).mean()
            grad_loss = sum(
                (_sample_mean((p - q).abs(), m) / chi_energy.sqrt()).mean()
                for p, q, m in zip(
                    spatial_gradient_3d(chi), target_gradients, gradient_masks
                )
            )
            harmonic_loss = mask.new_zeros(())
            if harmonic_mask is not None:
                normalized_phi = phi / phi_energy.sqrt().reshape(-1, 1, 1, 1, 1)
                laplacian = F.conv3d(normalized_phi, kernel, padding=1)
                pointwise = F.smooth_l1_loss(
                    laplacian, torch.zeros_like(laplacian), beta=0.1, reduction="none"
                )
                harmonic_loss = _sample_mean(pointwise, harmonic_mask).mean()
            terms = dict(
                loss_chi=cfg.lam_chi * chi_loss,
                loss_phi=cfg.lam_phi * phi_loss,
                loss_grad=cfg.lam_grad * grad_loss,
                loss_wh=lam_wh * harmonic_loss,
            )
            if high_mask is not None:
                high_error = high_mask * (chi - chi_gt).abs()
                if getattr(cfg, "normalize_high_chi_loss", False):
                    high_loss = (
                        high_error.flatten(start_dim=1).mean(dim=1)
                        / chi_energy.sqrt().clamp_min(1e-4)
                    ).mean()
                else:
                    high_loss = high_error.mean()
                terms["loss_chi_high"] = high_weight * high_loss
            total = total + weight * sum(terms.values())
            for key, value in terms.items():
                parts[key] = parts[key] + weight * value.detach()
    return total, parts


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


def _atomic_torch_save(state, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=destination.parent, suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(state, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path, value):
    path = Path(path)
    with NamedTemporaryFile(
        dir=path.parent, mode="w", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _autocast_context(device):
    return (
        autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else nullcontext()
    )


def make_subject_split(records, fraction, seed):
    cohorts = defaultdict(set)
    for record in records:
        cohort = record["subject"].split(":")[0]
        if cohort != "synthetic":
            cohorts[cohort].add(record["subject"])
    generator = torch.Generator().manual_seed(seed)
    validation_subjects = set()
    for subjects in cohorts.values():
        subjects = sorted(subjects)
        if len(subjects) < 2:
            raise ValueError("Each source cohort needs at least two subjects")
        count = max(1, min(len(subjects) - 1, round(len(subjects) * fraction)))
        validation_subjects.update(
            subjects[i]
            for i in torch.randperm(len(subjects), generator=generator)[:count].tolist()
        )
    train = [r for r in records if r["subject"] not in validation_subjects]
    val = [r for r in records if r["subject"] in validation_subjects]
    assert not {r["subject"] for r in train} & {r["subject"] for r in val}
    return train, val


def _limit_training_records(records, limit):
    """Keep both source types visible in bounded training diagnostics."""
    if not limit or limit >= len(records):
        return records
    synthetic = [
        record for record in records if record["subject"].startswith("synthetic:")
    ]
    original = [
        record for record in records if not record["subject"].startswith("synthetic:")
    ]
    if not synthetic or not original:
        return records[:limit]
    synthetic_count = max(1, round(limit * len(synthetic) / len(records)))
    synthetic_count = min(synthetic_count, len(synthetic), limit)
    original_count = min(limit - synthetic_count, len(original))
    synthetic_count = limit - original_count
    return original[:original_count] + synthetic[:synthetic_count]


class BalancedSourceSampler(Sampler[int]):
    """Draw a fixed source mix with fresh deterministic views for repeated records."""

    def __init__(self, records: list[dict], synthetic_fraction: float, seed: int):
        self.original = [
            index
            for index, row in enumerate(records)
            if not row["subject"].startswith("synthetic:")
        ]
        self.synthetic = [
            index
            for index, row in enumerate(records)
            if row["subject"].startswith("synthetic:")
        ]
        self.size = len(records)
        self.synthetic_fraction = synthetic_fraction
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return self.size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        n_synthetic = round(self.size * self.synthetic_fraction)
        if not self.synthetic:
            n_synthetic = 0
        if not self.original:
            n_synthetic = self.size

        def draw(pool, count, cohort):
            selected = []
            offset = self.epoch * count
            while len(selected) < count:
                cycle, within = divmod(offset, len(pool))
                cycle_generator = torch.Generator().manual_seed(
                    self.seed + cohort * 1000003 + cycle
                )
                order = torch.randperm(len(pool), generator=cycle_generator).tolist()
                take = min(count - len(selected), len(pool) - within)
                selected.extend(pool[index] for index in order[within : within + take])
                offset += take
            return selected

        selected = draw(self.original, self.size - n_synthetic, 0) + draw(
            self.synthetic, n_synthetic, 1
        )
        occurrences = defaultdict(int)
        encoded = []
        for index in selected:
            encoded.append(index + occurrences[index] * self.size)
            occurrences[index] += 1
        order = torch.randperm(len(encoded), generator=generator).tolist()
        return iter(encoded[index] for index in order)


def build_loaders(cfg, checkpoint_dir):
    records = training_records(cfg.data_dir, cfg.source_manifest)
    train, val = make_subject_split(records, cfg.val_frac, cfg.seed)
    manifest = {
        "seed": cfg.seed,
        "validation_fraction": cfg.val_frac,
        "source_mapping": "explicit JSON"
        if cfg.source_manifest or (Path(cfg.data_dir) / "subjects.json").is_file()
        else "legacy modulo 105 confirmed by user; separate modulo 56 from generator",
        "train": train,
        "validation": val,
    }
    path = checkpoint_dir / "split.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Saved split differs from the current dataset/configuration")
    _write_json(path, manifest)
    # Equal subject weighting, with independent fixed realizations for each subject.
    representatives = {r["subject"]: r for r in reversed(val)}
    val = [representatives[subject] for subject in sorted(representatives)]
    train = _limit_training_records(train, cfg.max_train_volumes)
    if cfg.max_val_subjects:
        val = val[: cfg.max_val_subjects]
    train_set = TrainingVolumes(
        train,
        cfg.cache_dir,
        seed=cfg.seed,
        chi_scale_mode=getattr(cfg, "chi_scale_mode", "legacy"),
    )
    val_set = TrainingVolumes(
        val, cfg.cache_dir, validation=True, views=cfg.val_views, seed=cfg.seed + 100000
    )
    generator = torch.Generator().manual_seed(cfg.seed)
    sampler = (
        BalancedSourceSampler(train, cfg.synthetic_fraction, cfg.seed)
        if getattr(cfg, "synthetic_fraction", -1) >= 0
        else None
    )
    options = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.device.startswith("cuda"),
        persistent_workers=cfg.num_workers > 0,
    )
    if cfg.num_workers:
        options.update(prefetch_factor=2, worker_init_fn=_initialize_worker)
    train_loader = DataLoader(
        train_set,
        shuffle=sampler is None,
        sampler=sampler,
        generator=generator,
        **options,
    )
    val_loader = DataLoader(val_set, shuffle=False, **options)
    shape = tuple(
        torch.load(
            records[0]["image"], map_location="cpu", weights_only=True, mmap=True
        ).shape[-3:]
    )
    return train_loader, val_loader, shape


def _initialize_worker(worker_id):
    torch.set_num_threads(1)


def learning_rate_multiplier(step, warmup_steps, total_steps, floor):
    if warmup_steps and step < warmup_steps:
        return max(floor, (step + 1) / warmup_steps)
    progress = min(
        1.0, max(0.0, (step - warmup_steps) / max(1, total_steps - warmup_steps - 1))
    )
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def depth_for_step(cfg, epoch, group):
    if cfg.final_epochs and epoch >= max(0, cfg.epochs - cfg.final_epochs):
        return cfg.final_iters
    fraction = min(1.0, epoch / max(1, cfg.iter_ramp_epochs))
    upper = round(cfg.iter_start + fraction * (cfg.iter_max - cfg.iter_start))
    choices = sorted(
        set(
            [cfg.iter_start]
            + [k for k in (8, 12, 20, 30) if cfg.iter_start <= k <= upper]
        )
    )
    generator = torch.Generator().manual_seed(cfg.seed + epoch * 1000003 + group)
    return choices[int(torch.randint(len(choices), (), generator=generator))]


def build_model(cfg, device):
    model = EfficientADMMUnrolledNet(
        ProximalNetwork(normalized_conditioning=True),
        ProximalNetwork(normalized_conditioning=True),
        num_iters=cfg.iter_start,
        input_scale_floor=cfg.input_scale_floor,
    ).to(device)
    if cfg.init_ckpt:
        state = torch.load(cfg.init_ckpt, map_location=device, weights_only=True)
        if "_pipeline_version" not in state:
            state = dict(state, _pipeline_version=model._pipeline_version.clone())
        model.load_state_dict(state, strict=True)
    # Both branches are optimized, including the previously frozen harmonic MLP.
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.endswith(".alpha"))
    model.refresh_spectral_norm(iterations=15)
    if cfg.channels_last_3d:
        model.to(memory_format=torch.channels_last_3d)
    if cfg.compile_proximal and device.type == "cuda":
        model.chinet.compile()
        model.phinet.compile()
    return model


def load_trained_model(checkpoint, device="cpu"):
    """Load a new run's weights with its matching inference implementation."""
    checkpoint = Path(checkpoint)
    saved_config = json.loads((checkpoint.parent / "config.json").read_text())
    saved_config.setdefault("input_scale_floor", 1e-3)
    cfg = TrainConfig(**saved_config)
    cfg.init_ckpt = None
    cfg.compile_proximal = False
    model = build_model(cfg, torch.device(device))
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True), strict=True
    )
    model.num_iters = cfg.final_iters
    return model.eval()


@torch.no_grad()
def evaluate(model, loader, synthesizer, device, depths):
    mode = model.training
    model.eval()
    metrics = {depth: defaultdict(list) for depth in depths}
    try:
        for cpu_batch in loader:
            phase, mask, weights, dipole, chi_gt, phi_gt = synthesizer(cpu_batch)
            with _autocast_context(device):
                outputs = model(
                    phase,
                    mask,
                    dipole,
                    weights,
                    return_iterates=True,
                    max_iters=max(depths),
                    supervision_indices=[k - 1 for k in depths],
                )
            for depth, (chi, phi) in zip(depths, outputs):
                values = (
                    torch.stack(
                        (
                            _sample_nrmse(chi, chi_gt, mask),
                            _sample_nrmse(phi, phi_gt, mask),
                        ),
                        dim=1,
                    )
                    .cpu()
                    .tolist()
                )
                for subject, pair in zip(cpu_batch["subject"], values):
                    metrics[depth][subject].append(pair)
    finally:
        model.train(mode)
    result = {}
    for depth, subjects in metrics.items():
        if not subjects:
            raise ValueError("Empty validation loader")
        mean = np.mean(
            [np.mean(values, axis=0) for values in subjects.values()], axis=0
        )
        result[f"val_chi_k{depth}"], result[f"val_phi_k{depth}"] = map(float, mean)
    return result


@torch.inference_mode()
def evaluate_whnet_external_cases(
    model: EfficientADMMUnrolledNet,
    cases: tuple[ExternalValidationCase, ...],
    device: torch.device,
    iterations: int,
) -> ExternalValidationResult:
    """Score external cases and render physical-unit comparison figures."""
    was_training = model.training
    model.eval()
    metrics: dict[str, float] = {}
    figures = {}
    try:
        for case in cases:
            phase = case.phase.to(device, non_blocking=True)
            mask = case.mask.to(device, non_blocking=True)
            dipole = case.dipole.to(device, non_blocking=True)
            weight = (
                mask
                if case.weight is None
                else case.weight.to(device, non_blocking=True)
            )
            chi_reference = case.chi.to(device, non_blocking=True)
            if phase.shape != mask.shape or phase.shape != chi_reference.shape:
                raise ValueError(f"{case.name} phase/mask/reference shapes differ")
            if weight.shape != mask.shape:
                raise ValueError(f"{case.name} weight/mask shapes differ")
            with _autocast_context(device):
                chi, background = model(
                    phase, mask, dipole, weight, max_iters=iterations
                )
            if not bool(torch.isfinite(chi).all() and torch.isfinite(background).all()):
                raise FloatingPointError(f"Nonfinite {case.name} reconstruction")
            chi_nrmse = masked_nrmse(chi, chi_reference, mask)
            metrics[f"external_{case.name}_chi_nrmse"] = chi_nrmse
            figures[f"{case.name}_chi_comparison"] = comparison_figure(
                _crop_tensor(chi_reference * case.factor, case.output_pad),
                _crop_tensor(chi * case.factor, case.output_pad),
                _crop_tensor(
                    (chi - chi_reference) * mask * case.factor, case.output_pad
                ),
                title=f"WHNet {case.name} χ (ppm), NRMSE={chi_nrmse:.4f}",
                method_label="WHNet",
                display_limit=0.1,
            )
            if case.background is not None:
                background_reference = case.background.to(device, non_blocking=True)
                if background_reference.shape != mask.shape:
                    raise ValueError(f"{case.name} background/mask shapes differ")
                background_nrmse = masked_nrmse(background, background_reference, mask)
                metrics[f"external_{case.name}_background_nrmse"] = background_nrmse
                figures[f"{case.name}_background_comparison"] = comparison_figure(
                    _crop_tensor(background_reference * case.factor, case.output_pad),
                    _crop_tensor(background * case.factor, case.output_pad),
                    _crop_tensor(
                        (background - background_reference) * mask * case.factor,
                        case.output_pad,
                    ),
                    title=(
                        f"WHNet {case.name} φ_h (ppm), NRMSE={background_nrmse:.4f}"
                    ),
                    method_label="WHNet",
                )
            del phase, mask, dipole, weight, chi_reference, chi, background
    except BaseException:
        import matplotlib.pyplot as plt

        for figure in figures.values():
            plt.close(figure)
        raise
    finally:
        model.train(was_training)
    return ExternalValidationResult(metrics, {}, figures)


def _configure_comet_api_key() -> None:
    """Use an exported key, or load COMET_API_KEY from the ignored project .env."""
    if os.environ.get("COMET_API_KEY"):
        return
    env_file = PROJECT_ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            key, separator, value = line.strip().partition("=")
            if separator and key.removeprefix("export ").strip() == "COMET_API_KEY":
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                if value:
                    os.environ["COMET_API_KEY"] = value
                    return
    raise RuntimeError("Set COMET_API_KEY in the environment or the project .env")


class RunLogger:
    def __init__(self, directory: Path, cfg: TrainConfig) -> None:
        self.directory = directory
        self.experiment = None
        self._comet_resumed = False
        if cfg.comet:
            _configure_comet_api_key()
            import comet_ml

            key_file = directory / "comet.json"
            saved_key = (
                json.loads(key_file.read_text())["experiment_key"]
                if cfg.resume and key_file.is_file()
                else None
            )
            self._comet_resumed = bool(saved_key)
            experiment = comet_ml.start(
                project_name=os.environ.get("COMET_PROJECT_NAME", "wh-net-qsm"),
                experiment_key=saved_key,
                mode="get" if saved_key else "create",
                experiment_config=comet_ml.ExperimentConfig(
                    name=directory.name,
                    tags=["whnet", "joint-training"],
                    log_code=False,
                    log_graph=False,
                    log_git_metadata=False,
                    log_git_patch=False,
                    log_env_details=False,
                    parse_args=False,
                    auto_output_logging=False,
                    auto_param_logging=False,
                    auto_metric_logging=False,
                    auto_log_co2=False,
                ),
            )
            try:
                if not saved_key:
                    experiment.log_parameters(asdict(cfg))
                _write_json(key_file, {"experiment_key": experiment.get_key()})
            except BaseException:
                experiment.end()
                raise
            self.experiment = experiment
        try:
            self.handle = (directory / "metrics.jsonl").open("a", buffering=1)
        except BaseException:
            if self.experiment:
                self.experiment.end()
            raise

    def log(self, values: dict) -> None:
        values = dict(values, timestamp=datetime.now().isoformat(timespec="seconds"))
        payload = json.dumps(values, allow_nan=False)
        self.handle.write(payload + "\n")
        print(payload, flush=True)
        if self.experiment:
            event = values.get("event")
            if event == "training_started" and not self._comet_resumed:
                self.experiment.log_parameters(
                    {
                        key: value
                        for key, value in values.items()
                        if key
                        in (
                            "train_volumes",
                            "synthetic_train_volumes",
                            "synthetic_epoch_samples",
                            "validation_realizations",
                            "optimizer_steps_per_epoch",
                            "chi_trainable",
                            "phi_trainable",
                            "initialization",
                        )
                    },
                    prefix="run",
                )
            elif event in ("train", "validation"):
                numeric = {
                    key: value
                    for key, value in values.items()
                    if key not in ("step", "epoch", "group")
                    and isinstance(value, (float, int))
                    and not isinstance(value, bool)
                }
                self.experiment.log_metrics(
                    numeric,
                    prefix=event,
                    step=values["step"],
                    epoch=values["epoch"],
                )

    def log_figure(self, figure, *, name: str, step: int, epoch: int) -> None:
        """Save a comparison image locally and upload it to Comet when enabled."""
        import matplotlib.pyplot as plt

        image_dir = self.directory / "images"
        image_dir.mkdir(exist_ok=True)
        try:
            figure.savefig(image_dir / f"{name}.png", dpi=120, bbox_inches="tight")
            if self.experiment:
                self.experiment.log_figure(
                    figure_name=name,
                    figure=figure,
                    step=step,
                    metadata={"epoch": epoch},
                    format="png",
                    overwrite=True,
                )
        finally:
            plt.close(figure)

    def close(self) -> None:
        try:
            self.handle.close()
        finally:
            if self.experiment:
                self.experiment.end()


def _branch_grad_norm(network):
    norms = [
        p.grad.detach().float().norm()
        for p in network.parameters()
        if p.grad is not None
    ]
    return (
        torch.stack(norms).norm()
        if norms
        else torch.tensor(0.0, device=next(network.parameters()).device)
    )


def _request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def run_training(cfg):
    global STOP_REQUESTED
    STOP_REQUESTED = False
    cfg.validate()
    if cfg.comet:
        _configure_comet_api_key()
    directory = Path(cfg.checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if not cfg.resume and any(directory.iterdir()):
        raise FileExistsError(
            f"Use a new checkpoint directory or --resume: {directory}"
        )
    _write_json(directory / "config.json", asdict(cfg))
    # Keep the exact implementation next to the weights for reproducible inference.
    import hashlib
    import shutil

    source_dir = directory / "source"
    source_dir.mkdir(exist_ok=True)
    fingerprints = {}
    for name in (
        "training.py",
        "model.py",
        "data.py",
        "physics.py",
        "paths.py",
        "adaptive_wh_tv_validation.py",
    ):
        source = Path(__file__).resolve().parent / name
        fingerprints[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        if not cfg.resume:
            shutil.copy2(source, source_dir / name)
    if not cfg.resume:
        _write_json(
            directory / "environment.json",
            {
                "torch_version": str(torch.__version__),
                "cuda_version": torch.version.cuda,
                "source_sha256": fingerprints,
            },
        )
    torch.set_num_threads(4)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable; use --device cpu for diagnostics"
            )
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(cfg.memory_fraction, device)
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.benchmark = True
        # Harmonic stencils rely on cancellation; use IEEE FP32 outside BF16 AMP.
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    train_loader, val_loader, shape = build_loaders(cfg, directory)
    synthesizer = FieldSynthesizer(shape, device)
    model = build_model(cfg, device)
    optimizer = torch.optim.Adam(
        [
            {
                "params": [p for p in model.chinet.parameters() if p.requires_grad],
                "lr": cfg.peak_lr,
                "name": "chi",
            },
            {
                "params": [p for p in model.phinet.parameters() if p.requires_grad],
                "lr": cfg.peak_lr * cfg.phi_lr_factor,
                "name": "phi",
            },
        ],
        foreach=device.type == "cuda",
    )
    groups_per_epoch = math.ceil(len(train_loader) / cfg.accumulation_steps)
    total_steps = groups_per_epoch * cfg.epochs
    warmup_steps = min(cfg.warmup_epochs * groups_per_epoch, max(0, total_steps - 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: learning_rate_multiplier(
            step, warmup_steps, total_steps, cfg.min_lr_factor
        ),
    )
    ema = EMA(model, cfg.ema_decay)
    start_epoch = start_group = global_step = 0
    best_val = math.inf
    if cfg.resume:
        state = torch.load(cfg.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        ema.shadow = state["ema"]
        start_epoch, start_group, global_step = (
            state["epoch"],
            state["next_group"],
            state["step"],
        )
        best_val = state["best_val"]
        torch.set_rng_state(state["rng_cpu"].cpu())
        if device.type == "cuda" and state["rng_cuda"] is not None:
            torch.cuda.set_rng_state(state["rng_cuda"].cpu(), device)
    logger = RunLogger(directory, cfg)
    start_time = time.perf_counter()
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    def save(epoch, next_group, status):
        _atomic_torch_save(model.state_dict(), directory / "model_last.pth")
        _atomic_torch_save(
            {
                "model": model.state_dict(),
                "ema": ema.shadow,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "next_group": next_group,
                "step": global_step,
                "best_val": best_val,
                "config": asdict(cfg),
                "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state(device)
                if device.type == "cuda"
                else None,
            },
            directory / "training_last.pt",
        )
        _write_json(
            directory / "status.json",
            {
                "status": status,
                "epoch": epoch,
                "next_group": next_group,
                "step": global_step,
                "best_val_chi": best_val if math.isfinite(best_val) else None,
                "updated": datetime.now().isoformat(timespec="seconds"),
            },
        )

    external_cases = None
    try:
        logger.log(
            {
                "event": "training_started",
                "step": global_step,
                "train_volumes": len(train_loader.dataset),
                "synthetic_train_volumes": sum(
                    record["subject"].startswith("synthetic:")
                    for record in train_loader.dataset.records
                ),
                "synthetic_epoch_samples": (
                    round(len(train_loader.dataset) * cfg.synthetic_fraction)
                    if cfg.synthetic_fraction >= 0
                    and any(
                        record["subject"].startswith("synthetic:")
                        for record in train_loader.dataset.records
                    )
                    else sum(
                        record["subject"].startswith("synthetic:")
                        for record in train_loader.dataset.records
                    )
                ),
                "validation_realizations": len(val_loader.dataset),
                "optimizer_steps_per_epoch": groups_per_epoch,
                "chi_trainable": sum(
                    p.numel() for p in model.chinet.parameters() if p.requires_grad
                ),
                "phi_trainable": sum(
                    p.numel() for p in model.phinet.parameters() if p.requires_grad
                ),
                "initialization": "resume" if cfg.resume else cfg.init_ckpt or "fresh",
            }
        )
        for epoch in range(start_epoch, cfg.epochs):
            model.train()
            train_loader.dataset.set_epoch(epoch)
            if isinstance(train_loader.sampler, BalancedSourceSampler):
                train_loader.sampler.set_epoch(epoch)
            train_loader.generator.manual_seed(cfg.seed + epoch)
            batches = iter(train_loader)
            first_group = start_group if epoch == start_epoch else 0
            for _ in range(
                min(first_group * cfg.accumulation_steps, len(train_loader))
            ):
                next(batches)
            skipped = 0
            for group in range(first_group, groups_per_epoch):
                step_start = time.perf_counter()
                depth = depth_for_step(cfg, epoch, group)
                model.num_iters = depth
                selected = supervision_indices(depth, cfg.deep_sup_k)
                micro_count = min(
                    cfg.accumulation_steps,
                    len(train_loader) - group * cfg.accumulation_steps,
                )
                expected_samples = min(
                    cfg.batch_size * micro_count,
                    len(train_loader.dataset)
                    - group * cfg.accumulation_steps * cfg.batch_size,
                )
                wh_factor = (
                    1.0
                    if not cfg.wh_warmup_epochs
                    else min(
                        1.0, (epoch + group / groups_per_epoch) / cfg.wh_warmup_epochs
                    )
                )
                log_step = global_step == 0 or (global_step + 1) % cfg.log_every == 0
                optimizer.zero_grad(set_to_none=True)
                model.refresh_spectral_norm()
                bad = False
                accum = (
                    defaultdict(lambda: torch.zeros((), device=device))
                    if log_step
                    else None
                )
                for _ in range(micro_count):
                    cpu_batch = next(batches)
                    if bad:
                        continue
                    synthesized = synthesizer(
                        cpu_batch,
                        perturb_local_field=cfg.local_perturbations,
                        background_mode=cfg.background_mode,
                        return_diagnostics=log_step,
                    )
                    phase, mask, weights, dipole, chi_gt, phi_gt = synthesized[:6]
                    fraction = phase.shape[0] / expected_samples
                    if log_step:
                        for name, values in synthesized[6].items():
                            accum[name] += values.mean() * fraction
                    with _autocast_context(device):
                        iterates = model(
                            phase,
                            mask,
                            dipole,
                            weights,
                            return_iterates=True,
                            supervision_indices=selected,
                        )
                    loss, parts = deep_supervised_loss(
                        iterates, chi_gt, phi_gt, mask, cfg.lam_wh * wh_factor, cfg
                    )
                    if not bool(torch.isfinite(loss)):
                        bad = True
                        continue
                    (loss * fraction).backward()
                    if accum is not None:
                        accum["loss"] += loss.detach() * fraction
                        for key, value in parts.items():
                            accum[key] += value * fraction
                        chi_pred, phi_pred = iterates[-1]
                        with torch.no_grad():
                            accum["nrmse_chi"] += (
                                _sample_nrmse(chi_pred, chi_gt, mask).mean() * fraction
                            )
                            accum["nrmse_phi"] += (
                                _sample_nrmse(phi_pred, phi_gt, mask).mean() * fraction
                            )
                chi_norm, phi_norm = (
                    _branch_grad_norm(model.chinet),
                    _branch_grad_norm(model.phinet),
                )
                chi_norm_value, phi_norm_value = float(chi_norm), float(phi_norm)
                grad_norm_value = math.hypot(chi_norm_value, phi_norm_value)
                if bad or not math.isfinite(grad_norm_value):
                    optimizer.zero_grad(set_to_none=True)
                    skipped += 1
                    logger.log(
                        {
                            "event": "skipped_nonfinite_update",
                            "epoch": epoch + 1,
                            "group": group,
                            "step": global_step,
                        }
                    )
                    if skipped >= 3:
                        raise FloatingPointError(
                            "Three nonfinite updates in one epoch; inspect metrics and training state"
                        )
                    continue
                if global_step == 0 and (chi_norm_value == 0 or phi_norm_value == 0):
                    raise RuntimeError("Both branches must receive nonzero gradients")
                # The previously frozen phi branch must not be suppressed by a
                # much larger chi gradient; clip each optimizer group separately.
                torch.nn.utils.clip_grad_norm_(model.chinet.parameters(), cfg.grad_clip)
                torch.nn.utils.clip_grad_norm_(model.phinet.parameters(), cfg.grad_clip)
                rates = [
                    group_options["lr"] for group_options in optimizer.param_groups
                ]
                optimizer.step()
                scheduler.step()
                ema.update(model)
                global_step += 1
                if log_step:
                    values = {key: float(value) for key, value in accum.items()}
                    values.update(
                        event="train",
                        epoch=epoch + 1,
                        group=group,
                        step=global_step,
                        num_iters=depth,
                        lr_chi=rates[0],
                        lr_phi=rates[1],
                        wh_factor=wh_factor,
                        grad_norm=grad_norm_value,
                        grad_norm_chi=chi_norm_value,
                        grad_norm_phi=phi_norm_value,
                        clipped=int(
                            max(chi_norm_value, phi_norm_value) > cfg.grad_clip
                        ),
                        clipped_chi=int(chi_norm_value > cfg.grad_clip),
                        clipped_phi=int(phi_norm_value > cfg.grad_clip),
                        seconds_per_update=time.perf_counter() - step_start,
                        elapsed_seconds=time.perf_counter() - start_time,
                    )
                    if device.type == "cuda":
                        values["peak_memory_gib"] = (
                            torch.cuda.max_memory_allocated(device) / 2**30
                        )
                    logger.log(values)
                if global_step == 1 or global_step % cfg.save_every == 0:
                    save(epoch, group + 1, "running")
                if STOP_REQUESTED or (cfg.max_steps and global_step >= cfg.max_steps):
                    save(epoch, group + 1, "paused")
                    logger.log(
                        {"event": "paused", "epoch": epoch + 1, "step": global_step}
                    )
                    return
            depths = sorted(
                {k for k in (20, 30, cfg.final_iters) if k <= cfg.final_iters}
            )
            with ema.average_parameters(model):
                model.refresh_spectral_norm(iterations=5)
                metrics = evaluate(model, val_loader, synthesizer, device, depths)
                external_figures = {}
                if (
                    cfg.external_validation
                    and (epoch + 1) % cfg.external_validation_every == 0
                ):
                    if external_cases is None:
                        external_cases = physical_external_cases(
                            load_external_cases(
                                cosmos_simulation=cfg.cosmos_simulation,
                                sim2_factor=cfg.sim2_factor,
                                sim2_pad=cfg.sim2_pad,
                            )
                        )
                    external_result = evaluate_whnet_external_cases(
                        model, external_cases, device, cfg.final_iters
                    )
                    metrics.update(external_result.metrics)
                    external_figures = external_result.figures
                score = metrics[f"val_chi_k{cfg.final_iters}"]
                if not all(math.isfinite(value) for value in metrics.values()):
                    raise FloatingPointError("Nonfinite validation metrics")
                if score < best_val:
                    best_val = score
                    _atomic_torch_save(model.state_dict(), directory / "model_best.pth")
                logger.log(
                    dict(
                        metrics,
                        event="validation",
                        epoch=epoch + 1,
                        step=global_step,
                        best_val_chi=best_val,
                        skipped_updates=skipped,
                    )
                )
                for name, figure in external_figures.items():
                    logger.log_figure(
                        figure,
                        name=f"external_{name}_epoch_{epoch + 1:04d}",
                        step=global_step,
                        epoch=epoch + 1,
                    )
            save(epoch + 1, 0, "running" if epoch + 1 < cfg.epochs else "complete")
            start_group = 0
        logger.log({"event": "complete", "step": global_step, "best_val_chi": best_val})
    except BaseException as error:
        _write_json(
            directory / "status.json",
            {
                "status": "failed",
                "step": global_step,
                "error": repr(error),
                "updated": datetime.now().isoformat(timespec="seconds"),
            },
        )
        raise
    finally:
        logger.close()


def main():
    from dataclasses import fields

    parser = argparse.ArgumentParser(description=__doc__)
    defaults = TrainConfig()
    for field in fields(defaults):
        value = getattr(defaults, field.name)
        flag = "--" + field.name.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(
                flag, action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS
            )
        else:
            parser.add_argument(
                flag,
                type=type(value) if value is not None else str,
                default=argparse.SUPPRESS,
            )
    args = vars(parser.parse_args())
    if args.get("init_ckpt") and not args.get("resume") and "peak_lr" not in args:
        # A warm start should not jump back to the fresh-training peak LR.
        args["peak_lr"] = 1e-5
    if args.get("resume"):
        saved = torch.load(args["resume"], map_location="cpu", weights_only=False)
        saved_config = dict(saved["config"])
        # Older runs keep their original loss and synthesis unless changed.
        saved_config.setdefault("lam_chi_high", 0.0)
        saved_config.setdefault("normalize_high_chi_loss", False)
        saved_config.setdefault("input_scale_floor", 1e-3)
        saved_config.setdefault("local_perturbations", False)
        saved_config.setdefault("chi_scale_mode", "legacy")
        saved_config.setdefault("background_mode", "legacy")
        saved_config.setdefault("synthetic_fraction", -1.0)
        saved_config.setdefault("external_validation", False)
        cfg = TrainConfig(**(saved_config | args))
        cfg.init_ckpt = None
    else:
        cfg = TrainConfig(**args)
    if not cfg.checkpoint_dir:
        cfg.checkpoint_dir = str(
            RUNS_DIR / ("joint_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        )
    run_training(cfg)


if __name__ == "__main__":
    main()

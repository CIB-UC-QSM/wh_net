"""Train the adaptive WH-TV parameter predictor on WHNet's synthetic data."""

from __future__ import annotations

import argparse
import json
import math
import signal
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast

from .adaptive_wh_tv import AdaptiveWHTV, AdaptiveWHTVOutput, WHTVParameterNet
from .adaptive_wh_tv_validation import (
    evaluate_external_cases,
    load_external_cases,
    physical_external_cases,
)
from .data import FieldSynthesizer, TRAINING_CONDITIONS
from .paths import RUNS_DIR, TRAINING_CACHE_DIR, TRAINING_DATA_DIR
from .training import (
    EMA,
    _atomic_torch_save,
    _sample_nrmse,
    _write_json,
    build_loaders,
    deep_supervised_loss,
    learning_rate_multiplier,
)

STOP_REQUESTED = False


@dataclass
class AdaptiveWHTVTrainConfig:
    data_dir: str = str(TRAINING_DATA_DIR)
    cache_dir: str = str(TRAINING_CACHE_DIR)
    source_manifest: str | None = None
    checkpoint_dir: str = ""
    resume: str | None = None
    epochs: int = 50
    batch_size: int = 1
    accumulation_steps: int = 5
    num_workers: int = 4
    peak_lr: float = 1e-4
    min_lr_factor: float = 0.05
    warmup_epochs: int = 3
    wh_warmup_epochs: int = 5
    lam_chi: float = 1.0
    lam_phi: float = 0.25
    lam_grad: float = 0.1
    lam_wh: float = 0.1
    final_weight: float = 1.0
    solver_iterations: int = 2000
    gradient_tail: int = 25
    checkpoint_block: int = 5
    features: tuple[int, ...] = (16, 32, 64)
    grad_clip: float = 1.0
    ema_decay: float = 0.99
    val_frac: float = 0.1
    val_views: int = 3
    seed: int = 0
    log_every: int = 5
    save_every: int = 10
    device: str = "cuda:0"
    memory_fraction: float = 0.25
    channels_last_3d: bool = True
    comet: bool = False
    local_perturbations: bool = True
    external_validation: bool = True
    external_validation_every: int = 1
    cosmos_simulation: str | None = None
    sim2_factor: float = 1.0
    sim2_pad: int = 20
    allow_short_solver: bool = False
    max_steps: int = 0
    max_train_volumes: int = 0
    max_val_subjects: int = 0

    def validate(self) -> None:
        positive = (
            "epochs",
            "batch_size",
            "accumulation_steps",
            "peak_lr",
            "min_lr_factor",
            "solver_iterations",
            "checkpoint_block",
            "grad_clip",
            "val_views",
            "log_every",
            "save_every",
            "external_validation_every",
            "sim2_factor",
        )
        if any(getattr(self, name) <= 0 for name in positive):
            raise ValueError("Counts, learning rates, and clipping must be positive")
        if not self.allow_short_solver and self.solver_iterations < 1000:
            raise ValueError(
                "Production WH-TV training requires at least 1000 iterations; "
                "use allow_short_solver only for diagnostics"
            )
        if not 0 < self.gradient_tail <= self.solver_iterations:
            raise ValueError("gradient_tail must be within the solver horizon")
        if self.num_workers < 0 or not 0 < self.val_frac < 1:
            raise ValueError("Invalid workers or validation fraction")
        if not 0 < self.memory_fraction <= 1 or not 0 < self.min_lr_factor <= 1:
            raise ValueError("Invalid memory fraction or learning-rate floor")
        if not 0 <= self.ema_decay < 1:
            raise ValueError("Invalid EMA decay")
        if min(self.lam_chi, self.lam_phi) <= 0 or min(self.lam_grad, self.lam_wh) < 0:
            raise ValueError("Invalid loss weights")
        if (
            min(
                self.warmup_epochs,
                self.wh_warmup_epochs,
                self.max_steps,
                self.max_train_volumes,
                self.max_val_subjects,
                self.sim2_pad,
            )
            < 0
        ):
            raise ValueError("Schedules and diagnostic limits cannot be negative")
        if not self.features or min(self.features) < 1:
            raise ValueError("features must contain positive channels")


def _autocast_context(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def _gradient_norm(parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    """Accumulate a global gradient norm in FP64 to avoid FP32 overflow."""
    return torch.linalg.vector_norm(
        torch.stack(
            [
                torch.linalg.vector_norm(parameter.grad.detach().double())
                for parameter in parameters
            ]
        )
    )


def build_adaptive_wh_tv_model(
    cfg: AdaptiveWHTVTrainConfig, device: torch.device
) -> AdaptiveWHTV:
    model = AdaptiveWHTV(
        WHTVParameterNet(tuple(cfg.features)),
        solver_iterations=cfg.solver_iterations,
        gradient_tail=cfg.gradient_tail,
        checkpoint_block=cfg.checkpoint_block,
    ).to(device)
    if cfg.channels_last_3d:
        model.to(memory_format=torch.channels_last_3d)
    return model


def load_adaptive_wh_tv(
    checkpoint: str | Path, device: str | torch.device = "cpu"
) -> AdaptiveWHTV:
    """Load an adaptive WH-TV checkpoint with its exact solver configuration."""
    checkpoint = Path(checkpoint)
    saved = json.loads((checkpoint.parent / "config.json").read_text())
    saved["features"] = tuple(saved["features"])
    cfg = AdaptiveWHTVTrainConfig(**saved)
    cfg.resume = None
    model = build_adaptive_wh_tv_model(cfg, torch.device(device))
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True), strict=True
    )
    return model.eval()


def _parameter_statistics(output: AdaptiveWHTVOutput) -> dict[str, float]:
    values: dict[str, float] = {}
    for name, parameter in output.parameters.as_dict().items():
        logged = parameter.detach().float().log10()
        values[f"log10_{name}_mean"] = float(logged.mean())
        values[f"log10_{name}_min"] = float(logged.min())
        values[f"log10_{name}_max"] = float(logged.max())
    selected = output.precision.detach()[output.precision.detach() > 0]
    variance = output.variance.detach()[output.variance.detach() > 0]
    values.update(
        precision_mean=float(selected.mean()),
        precision_min=float(selected.min()),
        precision_max=float(selected.max()),
        variance_mean=float(variance.mean()),
        variance_min=float(variance.min()),
        variance_max=float(variance.max()),
    )
    return values


def _validation_figure(
    phase: torch.Tensor,
    chi_target: torch.Tensor,
    background_target: torch.Tensor,
    output: AdaptiveWHTVOutput,
):
    import matplotlib.pyplot as plt

    arrays = [
        ("Phase", phase),
        ("Chi target", chi_target),
        ("Chi prediction", output.chi),
        ("Chi error", output.chi - chi_target),
        ("Background target", background_target),
        ("Background prediction", output.background),
        ("Background error", output.background - background_target),
        ("Relative variance", output.variance),
        ("Fidelity precision", output.precision),
    ]
    figure, axes = plt.subplots(len(arrays), 3, figsize=(10, 2.6 * len(arrays)))
    for row, (label, tensor) in enumerate(arrays):
        volume = tensor[0, 0].detach().float().cpu().numpy()
        finite = volume[np.isfinite(volume)]
        if label in {"Relative variance", "Fidelity precision"}:
            low, high = 0.0, float(np.quantile(finite, 0.995))
            cmap = "viridis"
        else:
            high = float(np.quantile(np.abs(finite), 0.995))
            low, cmap = -high, "gray"
        high = max(high, float(np.finfo(np.float32).eps))
        slices = (
            volume[volume.shape[0] // 2],
            volume[:, volume.shape[1] // 2],
            volume[:, :, volume.shape[2] // 2],
        )
        for column, image in enumerate(slices):
            axes[row, column].imshow(np.rot90(image), cmap=cmap, vmin=low, vmax=high)
            axes[row, column].axis("off")
            if column == 0:
                axes[row, column].set_title(label)
    figure.tight_layout()
    return figure


class RunLogger:
    def __init__(self, directory: Path, cfg: AdaptiveWHTVTrainConfig) -> None:
        self.directory = directory
        self.handle = (directory / "metrics.jsonl").open("a", buffering=1)
        self.experiment = None
        if cfg.comet:
            from comet_ml import Experiment

            self.experiment = Experiment(project_name="wh-net-qsm")
            self.experiment.add_tag("adaptive-wh-tv")
            self.experiment.log_parameters(asdict(cfg))

    def log(self, values: dict) -> None:
        values = dict(values, timestamp=datetime.now().isoformat(timespec="seconds"))
        self.handle.write(json.dumps(values, allow_nan=False) + "\n")
        print(json.dumps(values, allow_nan=False), flush=True)
        if self.experiment:
            numeric = {
                key: value
                for key, value in values.items()
                if isinstance(value, (float, int))
            }
            self.experiment.log_metrics(numeric, step=values.get("step"))

    def log_figure(
        self, figure, *, epoch: int, step: int, name: str | None = None
    ) -> None:
        image_dir = self.directory / "images"
        image_dir.mkdir(exist_ok=True)
        stem = name or f"validation_epoch_{epoch:04d}"
        destination = image_dir / f"{stem}.png"
        figure.savefig(destination, dpi=120, bbox_inches="tight")
        if self.experiment:
            self.experiment.log_figure(
                figure_name=stem,
                figure=figure,
                step=step,
            )
        import matplotlib.pyplot as plt

        plt.close(figure)

    def close(self) -> None:
        self.handle.close()
        if self.experiment:
            self.experiment.end()


@torch.no_grad()
def evaluate(
    model: AdaptiveWHTV,
    loader,
    synthesizer: FieldSynthesizer,
    device: torch.device,
) -> tuple[dict[str, float], tuple | None]:
    mode = model.training
    model.eval()
    subjects: dict[str, list[tuple[float, float]]] = defaultdict(list)
    example = None
    try:
        for cpu_batch in loader:
            phase, mask, _, dipole, chi_target, background_target = synthesizer(
                cpu_batch
            )
            with _autocast_context(device):
                output = model(phase, mask, dipole)
            chi_scores = _sample_nrmse(output.chi, chi_target, mask).cpu().tolist()
            background_scores = (
                _sample_nrmse(output.background, background_target, mask).cpu().tolist()
            )
            for subject, chi_score, background_score in zip(
                cpu_batch["subject"],
                chi_scores,
                background_scores,
                strict=True,
            ):
                subjects[subject].append((chi_score, background_score))
            if example is None:
                example = (
                    phase[:1],
                    chi_target[:1],
                    background_target[:1],
                    AdaptiveWHTVOutput(
                        output.chi[:1],
                        output.background[:1],
                        output.variance[:1],
                        output.precision[:1],
                        output.parameters,
                    ),
                )
    finally:
        model.train(mode)
    if not subjects:
        raise ValueError("Empty validation loader")
    mean = np.mean([np.mean(values, axis=0) for values in subjects.values()], axis=0)
    return {
        "val_nrmse_chi": float(mean[0]),
        "val_nrmse_background": float(mean[1]),
    }, example


def _request_stop(_signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def run_adaptive_wh_tv_training(cfg: AdaptiveWHTVTrainConfig) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = False
    cfg.validate()
    directory = Path(cfg.checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if not cfg.resume and any(directory.iterdir()):
        raise FileExistsError(
            f"Use a new checkpoint directory or --resume: {directory}"
        )
    _write_json(directory / "config.json", asdict(cfg))

    import hashlib
    import shutil

    resume_path = Path(cfg.resume).resolve() if cfg.resume else None
    snapshot_source = resume_path is None or resume_path.parent != directory.resolve()

    source_dir = directory / "source"
    source_dir.mkdir(exist_ok=True)
    source_names = (
        "adaptive_wh_tv.py",
        "adaptive_wh_tv_training.py",
        "adaptive_wh_tv_validation.py",
        "wh_tv.py",
        "data.py",
        "training.py",
        "model.py",
        "physics.py",
        "paths.py",
    )
    fingerprints = {}
    package_dir = Path(__file__).resolve().parent
    for name in source_names:
        source = package_dir / name
        fingerprints[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        if snapshot_source:
            shutil.copy2(source, source_dir / name)
    if snapshot_source:
        _write_json(
            directory / "environment.json",
            {
                "torch_version": str(torch.__version__),
                "cuda_version": torch.version.cuda,
                "source_sha256": fingerprints,
                "training_conditions": TRAINING_CONDITIONS,
                "resumed_from": str(resume_path) if resume_path else None,
                "gradient_estimator": (
                    f"full {cfg.solver_iterations}-iteration forward, "
                    f"last-{cfg.gradient_tail} truncated backpropagation"
                ),
            },
        )

    if snapshot_source and resume_path is not None:
        parent_best = resume_path.parent / "model_best.pth"
        if parent_best.is_file():
            shutil.copy2(parent_best, directory / "model_best.pth")

    torch.set_num_threads(4)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(cfg.memory_fraction, device)
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

    train_loader, val_loader, shape = build_loaders(cfg, directory)
    synthesizer = FieldSynthesizer(shape, device)
    model = build_adaptive_wh_tv_model(cfg, device)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=cfg.peak_lr,
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
        start_epoch = state["epoch"]
        start_group = state["next_group"]
        global_step = state["step"]
        best_val = state["best_val"]
        torch.set_rng_state(state["rng_cpu"].cpu())
        if device.type == "cuda" and state["rng_cuda"] is not None:
            torch.cuda.set_rng_state(state["rng_cuda"].cpu(), device)

    logger = RunLogger(directory, cfg)
    started = time.perf_counter()
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    def save(epoch: int, next_group: int, status: str) -> None:
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
                "rng_cuda": (
                    torch.cuda.get_rng_state(device) if device.type == "cuda" else None
                ),
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
                "validation_realizations": len(val_loader.dataset),
                "optimizer_steps_per_epoch": groups_per_epoch,
                "solver_iterations": cfg.solver_iterations,
                "gradient_tail": cfg.gradient_tail,
                "training_snr_min": TRAINING_CONDITIONS["snr_min"],
                "training_snr_max": TRAINING_CONDITIONS["snr_max"],
                "training_snr_definition": TRAINING_CONDITIONS["snr_definition"],
                "training_exterior_amplitude_min": TRAINING_CONDITIONS[
                    "exterior_source_amplitude_min"
                ],
                "training_exterior_amplitude_max": TRAINING_CONDITIONS[
                    "exterior_source_amplitude_max"
                ],
                "training_background_model": TRAINING_CONDITIONS["background_model"],
                "trainable_parameters": sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
            }
        )
        for epoch in range(start_epoch, cfg.epochs):
            model.train()
            train_loader.dataset.set_epoch(epoch)
            train_loader.generator.manual_seed(cfg.seed + epoch)
            batches = iter(train_loader)
            first_group = start_group if epoch == start_epoch else 0
            for _ in range(
                min(first_group * cfg.accumulation_steps, len(train_loader))
            ):
                next(batches)
            skipped = 0
            for group in range(first_group, groups_per_epoch):
                update_started = time.perf_counter()
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
                        1.0,
                        (epoch + group / groups_per_epoch) / cfg.wh_warmup_epochs,
                    )
                )
                optimizer.zero_grad(set_to_none=True)
                accumulator = defaultdict(lambda: torch.zeros((), device=device))
                last_output = None
                bad_reason = None
                for _ in range(micro_count):
                    cpu_batch = next(batches)
                    phase, mask, _, dipole, chi_target, background_target = synthesizer(
                        cpu_batch, perturb_local_field=cfg.local_perturbations
                    )
                    fraction = phase.shape[0] / expected_samples
                    with _autocast_context(device):
                        output = model(phase, mask, dipole)
                    loss, parts = deep_supervised_loss(
                        [(output.chi, output.background)],
                        chi_target,
                        background_target,
                        mask,
                        cfg.lam_wh * wh_factor,
                        cfg,
                    )
                    if not bool(torch.isfinite(loss.detach())):
                        bad_reason = "nonfinite_loss"
                        break
                    (loss * fraction).backward()
                    accumulator["loss"] += loss.detach() * fraction
                    for name, value in parts.items():
                        accumulator[name] += value * fraction
                    accumulator["nrmse_chi"] += (
                        _sample_nrmse(output.chi, chi_target, mask).mean().detach()
                        * fraction
                    )
                    accumulator["nrmse_background"] += (
                        _sample_nrmse(output.background, background_target, mask)
                        .mean()
                        .detach()
                        * fraction
                    )
                    last_output = output

                gradient_parameters = [
                    parameter
                    for parameter in model.parameters()
                    if parameter.grad is not None
                ]
                grad_norm = (
                    _gradient_norm(gradient_parameters)
                    if gradient_parameters
                    else torch.zeros((), device=device, dtype=torch.float64)
                )
                if (
                    bad_reason
                    or not bool(torch.isfinite(grad_norm))
                    or not gradient_parameters
                ):
                    if bad_reason is None:
                        bad_reason = (
                            "nonfinite_gradients"
                            if gradient_parameters
                            else "missing_gradients"
                        )
                    optimizer.zero_grad(set_to_none=True)
                    skipped += 1
                    logger.log(
                        {
                            "event": "skipped_nonfinite_update",
                            "epoch": epoch + 1,
                            "group": group,
                            "step": global_step,
                            "reason": bad_reason,
                        }
                    )
                    if skipped >= 3:
                        raise FloatingPointError("Three invalid updates in one epoch")
                    continue
                if float(grad_norm) > cfg.grad_clip:
                    coefficient = cfg.grad_clip / float(grad_norm)
                    for parameter in gradient_parameters:
                        parameter.grad.mul_(coefficient)
                rate = optimizer.param_groups[0]["lr"]
                optimizer.step()
                scheduler.step()
                ema.update(model)
                global_step += 1
                log_step = global_step == 1 or global_step % cfg.log_every == 0
                if log_step:
                    values = {name: float(value) for name, value in accumulator.items()}
                    if last_output is not None:
                        values.update(_parameter_statistics(last_output))
                    values.update(
                        event="train",
                        epoch=epoch + 1,
                        group=group,
                        step=global_step,
                        lr=rate,
                        wh_factor=wh_factor,
                        grad_norm=float(grad_norm),
                        clipped=int(float(grad_norm) > cfg.grad_clip),
                        seconds_per_update=time.perf_counter() - update_started,
                        elapsed_seconds=time.perf_counter() - started,
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

            with ema.average_parameters(model):
                metrics, example = evaluate(model, val_loader, synthesizer, device)
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
                    external_result = evaluate_external_cases(
                        model,
                        external_cases,
                        device,
                        bf16=(device.type == "cuda" and torch.cuda.is_bf16_supported()),
                    )
                    metrics.update(external_result.metrics)
                    external_figures = external_result.figures
                score = metrics["val_nrmse_chi"]
                if not all(math.isfinite(value) for value in metrics.values()):
                    raise FloatingPointError("Nonfinite validation metrics")
                if score < best_val:
                    best_val = score
                    _atomic_torch_save(model.state_dict(), directory / "model_best.pth")
                logger.log(
                    {
                        **metrics,
                        "event": "validation",
                        "epoch": epoch + 1,
                        "step": global_step,
                        "best_val_chi": best_val,
                        "skipped_updates": skipped,
                    }
                )
                if example is not None:
                    logger.log_figure(
                        _validation_figure(*example),
                        epoch=epoch + 1,
                        step=global_step,
                    )
                for name, figure in external_figures.items():
                    logger.log_figure(
                        figure,
                        epoch=epoch + 1,
                        step=global_step,
                        name=f"external_{name}_epoch_{epoch + 1:04d}",
                    )
            save(
                epoch + 1,
                0,
                "running" if epoch + 1 < cfg.epochs else "complete",
            )
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = AdaptiveWHTVTrainConfig()
    for field in fields(defaults):
        value = getattr(defaults, field.name)
        flag = "--" + field.name.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(
                flag,
                action=argparse.BooleanOptionalAction,
                default=argparse.SUPPRESS,
            )
        elif field.name == "features":
            parser.add_argument(
                flag,
                type=lambda text: tuple(int(value) for value in text.split(",")),
                default=argparse.SUPPRESS,
            )
        else:
            parser.add_argument(
                flag,
                type=type(value) if value is not None else str,
                default=argparse.SUPPRESS,
            )
    arguments = vars(parser.parse_args())
    if arguments.get("resume"):
        saved = torch.load(arguments["resume"], map_location="cpu", weights_only=False)
        saved_config = dict(saved["config"])
        saved_config.setdefault("local_perturbations", False)
        merged = saved_config | arguments
        merged["features"] = tuple(merged["features"])
        cfg = AdaptiveWHTVTrainConfig(**merged)
    else:
        cfg = AdaptiveWHTVTrainConfig(**arguments)
    if not cfg.checkpoint_dir:
        cfg.checkpoint_dir = str(
            RUNS_DIR / ("adaptive_wh_tv_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        )
    run_adaptive_wh_tv_training(cfg)


if __name__ == "__main__":
    main()

"""Independent flow-matching training for WH-Net susceptibility and background priors.

The loss fits branch velocities, independently of the ensemble's weighted
data-consistency solve. Background Laplacian errors are validation metrics only.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import signal
from time import perf_counter

import torch
from torch import Tensor

from .data import FieldSynthesizer
from .flow_admm import (
    FlowVelocity3D,
    _laplacian,
    flow_denoise,
    interior_mask,
    phase_scale,
)
from .paths import RUNS_DIR, TRAINING_CACHE_DIR, TRAINING_DATA_DIR
from .training import (
    EMA,
    TrainConfig,
    _atomic_torch_save,
    build_loaders,
    learning_rate_multiplier,
)


VALIDATION_TIMES = (0.2, 0.5, 0.8, 0.95)
STOP_REQUESTED = False


def _request_stop(_signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


@dataclass
class FlowTrainConfig:
    branch: str = "chi"
    data_dir: str = str(TRAINING_DATA_DIR)
    cache_dir: str = str(TRAINING_CACHE_DIR)
    source_manifest: str | None = None
    run_dir: str = ""
    resume: str | None = None
    epochs: int = 50
    batch_size: int = 1
    accumulation_steps: int = 5
    num_workers: int = 4
    base_channels: int = 8
    peak_lr: float = 1e-4
    warmup_epochs: int = 3
    min_lr_factor: float = 0.05
    ema_decay: float = 0.99
    val_frac: float = 0.1
    val_views: int = 3
    seed: int = 0
    device: str = "cuda:0"
    memory_fraction: float = 0.25
    save_every: int = 50
    max_steps: int = 0
    comet: bool = False
    image_every: int = 1
    image_examples: int = 1

    def validate(self) -> None:
        if self.branch not in ("chi", "background"):
            raise ValueError("branch must be chi or background")
        if (
            min(
                self.epochs,
                self.batch_size,
                self.accumulation_steps,
                self.save_every,
                self.val_views,
                self.image_every,
            )
            < 1
        ):
            raise ValueError("training counts must be positive")
        if self.num_workers < 0 or self.max_steps < 0 or self.warmup_epochs < 0:
            raise ValueError("workers, warmup and max_steps must be nonnegative")
        if self.image_examples < 0:
            raise ValueError("image_examples must be nonnegative")
        if self.base_channels < 4 or self.base_channels % 4:
            raise ValueError("base_channels must be a multiple of four")
        if self.peak_lr <= 0 or not 0 < self.min_lr_factor <= 1:
            raise ValueError("invalid learning rate")
        if not 0 <= self.ema_decay < 1 or not 0 < self.val_frac < 1:
            raise ValueError("invalid EMA decay or validation fraction")
        if not 0 < self.memory_fraction <= 1:
            raise ValueError("invalid CUDA memory fraction")


class FlowRunLogger:
    """Persist metrics and epoch images, optionally resuming their Comet series."""

    def __init__(self, directory: Path, config: FlowTrainConfig) -> None:
        self.directory = directory
        self.branch = config.branch
        self.experiment = None
        if config.comet:
            from comet_ml import Experiment, ExistingExperiment

            comet_path = directory / "comet.json"
            if config.resume and comet_path.is_file():
                saved = json.loads(comet_path.read_text())
                self.experiment = ExistingExperiment(
                    experiment_key=saved["experiment_key"], auto_metric_logging=False
                )
            else:
                self.experiment = Experiment(
                    project_name="wh-net-qsm", auto_metric_logging=False
                )
                self.experiment.set_name(directory.name)
                comet_path.write_text(
                    json.dumps({"experiment_key": self.experiment.get_key()}, indent=2)
                    + "\n"
                )
            self.experiment.add_tag(f"flow-{config.branch}")
            self.experiment.log_parameters(asdict(config))
        self.handle = (directory / "metrics.jsonl").open("a", buffering=1)

    def __enter__(self) -> FlowRunLogger:
        return self

    def __exit__(self, _error_type, _error, _traceback) -> None:
        self.handle.close()
        if self.experiment:
            self.experiment.end()

    def log(self, values: dict) -> None:
        values = dict(values, timestamp=datetime.now().isoformat(timespec="seconds"))
        self.handle.write(json.dumps(values, allow_nan=False) + "\n")
        if self.experiment:
            numeric = {
                key: value
                for key, value in values.items()
                if isinstance(value, (float, int)) and key not in ("step", "epoch")
            }
            self.experiment.log_metrics(
                numeric,
                prefix=values.get("event"),
                step=values.get("step"),
                epoch=values.get("epoch"),
            )

    def log_image(
        self, figure, *, name: str, epoch: int, step: int, metadata: dict
    ) -> Path:
        """Use one stable Comet name and epoch as image step for its slider."""
        import matplotlib.pyplot as plt

        destination = self.directory / "images" / name / f"epoch_{epoch:04d}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        details = {
            **metadata,
            "epoch": epoch,
            "optimizer_step": step,
            "branch": self.branch,
        }
        try:
            figure.savefig(destination, dpi=120)
            destination.with_suffix(".json").write_text(
                json.dumps(details, indent=2) + "\n"
            )
            if self.experiment:
                self.experiment.log_image(
                    image_data=str(destination),
                    name=name,
                    step=epoch,
                    overwrite=False,
                    metadata=details,
                )
            self.handle.write(
                json.dumps(
                    {
                        "event": "image",
                        "name": name,
                        "epoch": epoch,
                        "step": step,
                        "path": str(destination.relative_to(self.directory)),
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                    }
                )
                + "\n"
            )
        finally:
            plt.close(figure)
        return destination


def _target(
    phase: Tensor, mask: Tensor, chi: Tensor, background: Tensor, branch: str
) -> Tensor:
    return ((chi if branch == "chi" else background) / phase_scale(phase, mask)) * mask


def flow_matching_loss(
    model: FlowVelocity3D,
    target: Tensor,
    mask: Tensor,
    times: Tensor,
    source: Tensor,
    condition: Tensor | None = None,
) -> Tensor:
    times = times[:, None, None, None, None]
    source = source * mask
    mixed = (times * target + (1 - times) * source) * mask
    residual = (
        model(mixed, times[:, 0, 0, 0, 0], mask, condition=condition).float()
        - (target - source)
    ) * mask
    count = mask.sum(dim=(-3, -2, -1)).clamp_min(1)
    return (residual.square().sum(dim=(-3, -2, -1)) / count).mean()


def _prior_figure(
    source: Tensor,
    mixed: Tensor,
    target: Tensor,
    prediction: Tensor,
    mask: Tensor,
    *,
    title: str,
):
    """Orthogonal views with scales derived only from the fixed validation inputs."""
    import matplotlib.pyplot as plt
    import numpy as np

    support = mask.detach().cpu().squeeze().bool().numpy()
    arrays = [
        value.detach().float().cpu().squeeze().numpy() * support
        for value in (source, mixed, target, prediction)
    ]
    error = arrays[3] - arrays[2]
    reference_limit = max(float(np.percentile(np.abs(arrays[2][support]), 99.5)), 1e-7)
    input_limits = [
        max(float(np.percentile(np.abs(value[support]), 99.5)), 1e-7)
        for value in arrays[:2]
    ]
    rows = (
        (arrays[0], "Source", "gray", input_limits[0]),
        (arrays[1], "Flow input", "gray", input_limits[1]),
        (arrays[2], "Reference", "gray", reference_limit),
        (arrays[3], "Predicted endpoint", "gray", reference_limit),
        (error, "Error", "coolwarm", reference_limit),
    )
    center = tuple(length // 2 for length in arrays[2].shape)
    figure, axes = plt.subplots(5, 3, figsize=(10, 12), constrained_layout=True)
    figure.suptitle(title)
    for row, (volume, label, cmap, limit) in enumerate(rows):
        views = (volume[center[0]], volume[:, center[1]], volume[:, :, center[2]])
        for column, view in enumerate(views):
            axes[row, column].imshow(np.rot90(view), cmap=cmap, vmin=-limit, vmax=limit)
            axes[row, column].axis("off")
            if column == 0:
                axes[row, column].set_title(label)
    return figure


@torch.no_grad()
def evaluate_prior(
    model,
    loader,
    synthesizer,
    branch: str,
    device: torch.device,
    *,
    logger: FlowRunLogger | None = None,
    epoch: int = 0,
    step: int = 0,
    image_examples: int = 0,
) -> dict[str, float]:
    model.eval()
    by_subject: dict[str, list[tuple[float, float, float]]] = {}
    by_time: dict[float, dict[str, list[tuple[float, float, float]]]] = {
        time: {} for time in VALIDATION_TIMES
    }
    separation_by_subject: dict[str, list[tuple[float, float]]] = {}
    example_count = 0
    for batch in loader:
        phase, mask, _, _, chi, background = synthesizer(batch)
        target = _target(phase, mask, chi, background, branch)
        measured_field = phase / phase_scale(phase, mask) * mask
        scale = phase_scale(phase, mask)
        image_indices = range(
            min(phase.shape[0], max(0, image_examples - example_count))
        )
        harmonic_mask = interior_mask(mask) if branch == "background" else None
        for index, time in enumerate(VALIDATION_TIMES):
            if branch == "chi":
                source = (
                    torch.stack(
                        [
                            torch.randn(
                                target[sample].shape,
                                device=device,
                                generator=torch.Generator(device=device).manual_seed(
                                    int(seed) + 10007 * index + 700001
                                ),
                            )
                            for sample, seed in enumerate(batch["seed"].tolist())
                        ]
                    )
                    * mask
                )
                condition = None
            else:
                source = measured_field
                condition = measured_field
            mixed = (time * target + (1 - time) * source) * mask
            with (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else nullcontext()
            ):
                prediction = flow_denoise(model, mixed, time, mask, condition).float()
            error = (
                ((prediction - target).square() * mask).sum(dim=(-3, -2, -1))
                / mask.sum(dim=(-3, -2, -1)).clamp_min(1)
            ).flatten()
            target_mse = (
                (target.square() * mask).sum(dim=(-3, -2, -1))
                / mask.sum(dim=(-3, -2, -1)).clamp_min(1)
            ).flatten()
            nrmse = (error / target_mse.clamp_min(1e-12)).sqrt()
            harmonic = (
                (_laplacian(prediction).square() * harmonic_mask).sum(dim=(-3, -2, -1))
                / harmonic_mask.sum(dim=(-3, -2, -1)).clamp_min(1)
                if harmonic_mask is not None
                else torch.zeros_like(error)
            ).flatten()
            for subject, mse, relative, lap in zip(
                batch["subject"], error.tolist(), nrmse.tolist(), harmonic.tolist()
            ):
                values = mse, relative, lap
                by_subject.setdefault(subject, []).append(values)
                by_time[time].setdefault(subject, []).append(values)
            if logger:
                for sample in image_indices:
                    name = f"validation_{branch}_example{example_count + sample:02d}_t{round(time * 100):03d}"
                    logger.log_image(
                        _prior_figure(
                            source[sample] * scale[sample],
                            mixed[sample] * scale[sample],
                            target[sample] * scale[sample],
                            prediction[sample] * scale[sample],
                            mask[sample],
                            title=f"{branch} endpoint, t={time:.2f}, epoch={epoch}, NRMSE={float(nrmse[sample]):.4f}",
                        ),
                        name=name,
                        epoch=epoch,
                        step=step,
                        metadata={
                            "subject": batch["subject"][sample],
                            "synthesis_seed": int(batch["seed"][sample]),
                            "flow_time": time,
                            "kind": "denoising_endpoint",
                            "nrmse": float(nrmse[sample]),
                        },
                    )
        # Show a measured-field-only separation as well as target-interpolated
        # denoising. This preview does not use the true background as an input.
        if branch == "background":
            with (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else nullcontext()
            ):
                separated = flow_denoise(
                    model, measured_field, 0.0, mask, measured_field
                ).float()
            separation_mse = (
                ((separated - target).square() * mask).sum(dim=(-3, -2, -1))
                / mask.sum(dim=(-3, -2, -1)).clamp_min(1)
            ).flatten()
            separation_nrmse = (separation_mse / target_mse.clamp_min(1e-12)).sqrt()
            for subject, mse, relative in zip(
                batch["subject"], separation_mse.tolist(), separation_nrmse.tolist()
            ):
                separation_by_subject.setdefault(subject, []).append((mse, relative))
            for sample in image_indices if logger else ():
                logger.log_image(
                    _prior_figure(
                        phase[sample],
                        phase[sample],
                        target[sample] * scale[sample],
                        separated[sample] * scale[sample],
                        mask[sample],
                        title=f"Background from measured total field, t=0, epoch={epoch}, NRMSE={float(separation_nrmse[sample]):.4f}",
                    ),
                    name=f"validation_background_example{example_count + sample:02d}_total_field",
                    epoch=epoch,
                    step=step,
                    metadata={
                        "subject": batch["subject"][sample],
                        "synthesis_seed": int(batch["seed"][sample]),
                        "flow_time": 0.0,
                        "kind": "total_field_separation",
                        "nrmse": float(separation_nrmse[sample]),
                    },
                )
        example_count += len(image_indices)
    if not by_subject:
        raise ValueError("empty validation loader")

    def macro_average(records, index):
        return sum(
            sum(row[index] for row in values) / len(values)
            for values in records.values()
        ) / len(records)

    metrics = {
        "denoise_mse": macro_average(by_subject, 0),
        "denoise_nrmse": macro_average(by_subject, 1),
        "harmonic_mse": macro_average(by_subject, 2),
    }
    for flow_time, records in by_time.items():
        suffix = f"t{round(flow_time * 100):03d}"
        metrics[f"denoise_mse_{suffix}"] = macro_average(records, 0)
        metrics[f"denoise_nrmse_{suffix}"] = macro_average(records, 1)
    if separation_by_subject:
        metrics["total_field_separation_mse"] = macro_average(separation_by_subject, 0)
        metrics["total_field_separation_nrmse"] = macro_average(
            separation_by_subject, 1
        )
    return metrics


def run_flow_training(config: FlowTrainConfig) -> Path:
    global STOP_REQUESTED
    STOP_REQUESTED = False
    config.validate()
    run_dir = Path(config.run_dir)
    if not config.run_dir:
        raise ValueError("run_dir must be set")
    if config.resume:
        if Path(config.resume).parent != run_dir:
            raise ValueError("resume checkpoint must belong to run_dir")
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"use a fresh run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    saved_config = asdict(config)
    if config.resume:
        prior_config = json.loads(config_path.read_text())
        for key in prior_config:
            if (
                key
                not in ("resume", "max_steps", "comet", "image_every", "image_examples")
                and saved_config[key] != prior_config[key]
            ):
                raise ValueError(f"resume configuration changed: {key}")
    else:
        config_path.write_text(json.dumps(saved_config, indent=2) + "\n")
        source_root = Path(__file__).resolve().parent
        source_hashes = {
            name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
            for name in ("flow_admm.py", "flow_training.py", "data.py", "physics.py")
        }
        (run_dir / "environment.json").write_text(
            json.dumps(
                {
                    "torch_version": str(torch.__version__),
                    "cuda_version": torch.version.cuda,
                    "source_sha256": source_hashes,
                },
                indent=2,
            )
            + "\n"
        )
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA unavailable; use --device cpu for a diagnostic run"
            )
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(config.memory_fraction, device)
        torch.cuda.manual_seed_all(config.seed)
    loader_config = TrainConfig(
        data_dir=config.data_dir,
        cache_dir=config.cache_dir,
        source_manifest=config.source_manifest,
        checkpoint_dir=str(run_dir),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        val_frac=config.val_frac,
        val_views=config.val_views,
        seed=config.seed,
        device=config.device,
    )
    train_loader, validation_loader, shape = build_loaders(loader_config, run_dir)
    synthesizer = FieldSynthesizer(shape, device)
    model = FlowVelocity3D(
        config.base_channels, condition_on_field=config.branch == "background"
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.peak_lr)
    groups_per_epoch = math.ceil(len(train_loader) / config.accumulation_steps)
    total_steps = groups_per_epoch * config.epochs
    warmup_steps = min(config.warmup_epochs * groups_per_epoch, max(0, total_steps - 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: learning_rate_multiplier(
            step, warmup_steps, total_steps, config.min_lr_factor
        ),
    )
    ema = EMA(model, config.ema_decay)
    start_epoch = start_group = global_step = 0
    best = math.inf
    if config.resume:
        state = torch.load(config.resume, map_location=device, weights_only=True)
        if (
            state.get("format") != "whnet-flow-training-v2"
            or state.get("branch") != config.branch
        ):
            raise ValueError("resume checkpoint uses an incompatible flow formulation")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        ema.shadow = state["ema"]
        start_epoch, start_group, global_step, best = (
            state["epoch"],
            state["group"],
            state["step"],
            state["best"],
        )
        torch.set_rng_state(state["rng_cpu"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["rng_cuda"].cpu(), device)
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    def save(epoch: int, group: int, status: str) -> None:
        _atomic_torch_save(
            {
                "format": "whnet-flow-training-v2",
                "branch": config.branch,
                "model": model.state_dict(),
                "ema": ema.shadow,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "group": group,
                "step": global_step,
                "best": best,
                "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state(device)
                if device.type == "cuda"
                else None,
            },
            run_dir / "training_last.pt",
        )
        (run_dir / "status.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "epoch": epoch,
                    "group": group,
                    "step": global_step,
                    "best_denoise_mse": best,
                },
                indent=2,
            )
            + "\n"
        )

    with FlowRunLogger(run_dir, config) as log:
        log.log(
            {
                "event": "started",
                "step": global_step,
                "epoch": start_epoch,
                "train_volumes": len(train_loader.dataset),
                "validation_realizations": len(validation_loader.dataset),
                "optimizer_steps_per_epoch": groups_per_epoch,
            }
        )
        for epoch in range(start_epoch, config.epochs):
            epoch_started = perf_counter()
            epoch_loss_total = 0.0
            epoch_micro_total = 0
            model.train()
            train_loader.dataset.set_epoch(epoch)
            train_loader.generator.manual_seed(config.seed + epoch)
            batches = iter(train_loader)
            first_group = start_group if epoch == start_epoch else 0
            for _ in range(
                min(first_group * config.accumulation_steps, len(train_loader))
            ):
                next(batches)
            for group in range(first_group, groups_per_epoch):
                micro_count = min(
                    config.accumulation_steps,
                    len(train_loader) - group * config.accumulation_steps,
                )
                optimizer.zero_grad(set_to_none=True)
                train_loss = 0.0
                for _ in range(micro_count):
                    batch = next(batches)
                    phase, mask, _, _, chi, background = synthesizer(batch)
                    target = _target(phase, mask, chi, background, config.branch)
                    times = torch.rand(target.shape[0], device=device)
                    measured_field = phase / phase_scale(phase, mask) * mask
                    source = (
                        torch.randn_like(target) * mask
                        if config.branch == "chi"
                        else measured_field
                    )
                    condition = (
                        measured_field if config.branch == "background" else None
                    )
                    with (
                        torch.autocast("cuda", dtype=torch.bfloat16)
                        if device.type == "cuda"
                        else nullcontext()
                    ):
                        loss = flow_matching_loss(
                            model, target, mask, times, source, condition
                        )
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("nonfinite flow matching loss")
                    (loss / micro_count).backward()
                    train_loss += float(loss.detach()) / micro_count
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                scheduler.step()
                ema.update(model)
                global_step += 1
                epoch_loss_total += train_loss * micro_count
                epoch_micro_total += micro_count
                log.log(
                    {
                        "event": "train",
                        "epoch": epoch + 1,
                        "step": global_step,
                        "loss": train_loss,
                        "lr": optimizer.param_groups[0]["lr"],
                        "grad_norm": float(grad_norm),
                        "clipped": int(float(grad_norm) > 1.0),
                    }
                )
                if global_step % config.save_every == 0:
                    save(epoch, group + 1, "running")
                if STOP_REQUESTED or (
                    config.max_steps and global_step >= config.max_steps
                ):
                    save(epoch, group + 1, "paused")
                    log.log(
                        {"event": "paused", "epoch": epoch + 1, "step": global_step}
                    )
                    return run_dir
            with ema.average_parameters(model):
                metrics = evaluate_prior(
                    model,
                    validation_loader,
                    synthesizer,
                    config.branch,
                    device,
                    logger=log if (epoch + 1) % config.image_every == 0 else None,
                    epoch=epoch + 1,
                    step=global_step,
                    image_examples=config.image_examples,
                )
                if not all(math.isfinite(value) for value in metrics.values()):
                    raise FloatingPointError("nonfinite flow validation metrics")
                if metrics["denoise_mse"] < best:
                    best = metrics["denoise_mse"]
                    _atomic_torch_save(
                        {
                            "format": "whnet-flow-v2",
                            "branch": config.branch,
                            "source": "gaussian"
                            if config.branch == "chi"
                            else "measured_total_field",
                            "base_channels": config.base_channels,
                            "model": model.state_dict(),
                            "normalization": "masked_phase_rms_clamp_1e-3",
                            "validation_times": VALIDATION_TIMES,
                        },
                        run_dir / "model_best.pt",
                    )
                log.log(
                    {
                        "event": "validation",
                        "epoch": epoch + 1,
                        "step": global_step,
                        "best": best,
                        **metrics,
                    }
                )
            epoch_metrics = {
                "event": "epoch",
                "epoch": epoch + 1,
                "step": global_step,
                "seconds": perf_counter() - epoch_started,
                "training_microbatches": epoch_micro_total,
            }
            if epoch_micro_total:
                epoch_metrics["loss"] = epoch_loss_total / epoch_micro_total
            log.log(epoch_metrics)
            save(epoch + 1, 0, "complete" if epoch + 1 == config.epochs else "running")
        log.log(
            {
                "event": "complete",
                "epoch": config.epochs,
                "step": global_step,
                "best": best,
            }
        )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=("chi", "background"), required=True)
    parser.add_argument("--run-dir")
    parser.add_argument("--resume")
    parser.add_argument("--data-dir", default=str(TRAINING_DATA_DIR))
    parser.add_argument("--cache-dir", default=str(TRAINING_CACHE_DIR))
    parser.add_argument("--source-manifest")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--peak-lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-fraction", type=float, default=0.25)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--comet", action="store_true")
    parser.add_argument("--image-every", type=int, default=1)
    parser.add_argument("--image-examples", type=int, default=1)
    args = parser.parse_args()
    run_dir = args.run_dir or (
        str(Path(args.resume).parent)
        if args.resume
        else str(RUNS_DIR / f"flow_{args.branch}_{datetime.now():%Y%m%d_%H%M%S_%f}")
    )
    config = FlowTrainConfig(**{**vars(args), "run_dir": run_dir})
    run_flow_training(config)


if __name__ == "__main__":
    main()

"""Validation-only selection of frozen FlowADMM priors and solver settings."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
import math
from pathlib import Path
import time

import torch

from .data import FieldSynthesizer
from .flow_admm import (
    FLOW_DATA_TERM,
    FLOW_ENSEMBLE_FORMAT,
    FLOW_SOLVER,
    FlowADMMConfig,
    load_flow_admm,
    phase_scale,
)
from .paths import RUNS_DIR, TRAINING_CACHE_DIR, TRAINING_DATA_DIR
from .training import TrainConfig, _sample_nrmse, build_loaders


@torch.no_grad()
def evaluate_solver(model, loader, synthesizer, *, seed: int) -> dict[str, float]:
    """Macro average metrics over subjects and deterministic synthesis views."""
    records: dict[str, list[tuple[float, float, float, float, float]]] = {}
    for batch in loader:
        phase, mask, weights, dipole, chi_target, background_target = synthesizer(batch)
        for index, subject in enumerate(batch["subject"]):
            result = model(
                phase[index : index + 1],
                mask[index : index + 1],
                dipole,
                weights[index : index + 1],
                seed=seed + int(batch["seed"][index]),
                return_diagnostics=True,
            )
            chi, background, trace = result
            scale = float(
                phase_scale(phase[index : index + 1], mask[index : index + 1]).mean()
            )
            values = (
                float(
                    _sample_nrmse(
                        chi, chi_target[index : index + 1], mask[index : index + 1]
                    ).mean()
                ),
                float(
                    _sample_nrmse(
                        background,
                        background_target[index : index + 1],
                        mask[index : index + 1],
                    ).mean()
                ),
                trace[-1]["data_rms"] * scale,
                trace[-1]["weighted_data_rms"] * scale,
                trace[-1]["harmonic_rms"] * scale,
            )
            records.setdefault(subject, []).append(values)
    if not records:
        raise ValueError("empty validation set")
    subject_means = [
        [sum(row[index] for row in values) / len(values) for index in range(5)]
        for values in records.values()
    ]
    means = [
        sum(row[index] for row in subject_means) / len(subject_means)
        for index in range(5)
    ]
    return dict(
        zip(
            (
                "chi_nrmse",
                "background_nrmse",
                "data_rms",
                "weighted_data_rms",
                "harmonic_rms",
            ),
            means,
        )
    )


def tune_flow_admm(
    chi_checkpoint: str | Path,
    background_checkpoint: str | Path,
    *,
    run_dir: str | Path,
    data_dir: str | Path = TRAINING_DATA_DIR,
    cache_dir: str | Path = TRAINING_CACHE_DIR,
    source_manifest: str | None = None,
    device: str = "cuda:0",
    seed: int = 0,
    val_frac: float = 0.1,
    val_views: int = 3,
    num_workers: int = 4,
    max_val_subjects: int = 0,
) -> Path:
    chi_checkpoint = Path(chi_checkpoint).resolve()
    background_checkpoint = Path(background_checkpoint).resolve()
    chi_split = json.loads((chi_checkpoint.parent / "split.json").read_text())
    background_split = json.loads(
        (background_checkpoint.parent / "split.json").read_text()
    )
    if chi_split != background_split:
        raise ValueError("priors were trained with different subject splits")
    directory = Path(run_dir)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"use a fresh tuning directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu for a diagnostic run")
    loader_config = TrainConfig(
        data_dir=str(data_dir),
        cache_dir=str(cache_dir),
        source_manifest=source_manifest,
        checkpoint_dir=str(directory),
        batch_size=1,
        num_workers=num_workers,
        val_frac=val_frac,
        val_views=val_views,
        max_val_subjects=max_val_subjects,
        seed=seed,
        device=device,
    )
    _, loader, shape = build_loaders(loader_config, directory)
    selected_split = json.loads((directory / "split.json").read_text())
    if selected_split != chi_split:
        raise ValueError("tuning split differs from prior training split")
    synthesizer = FieldSynthesizer(shape, target)
    model = load_flow_admm(chi_checkpoint, background_checkpoint, device=target)
    config = model.config
    history = []

    def score(candidate: FlowADMMConfig) -> tuple[float, float]:
        model.config = candidate
        if target.type == "cuda":
            torch.cuda.synchronize(target)
        started = time.perf_counter()
        metrics = evaluate_solver(model, loader, synthesizer, seed=seed + 3000001)
        if target.type == "cuda":
            torch.cuda.synchronize(target)
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("nonfinite FlowADMM validation metric")
        evaluations = sum(
            candidate.schedule(index)[1] for index in range(candidate.iterations)
        )
        history.append(
            {
                "data_term": FLOW_DATA_TERM,
                "solver_type": FLOW_SOLVER,
                "solver": asdict(candidate),
                **metrics,
                "chi_flow_evaluations": evaluations,
                "background_flow_evaluations": candidate.iterations,
                "seconds_per_realization": (time.perf_counter() - started)
                / len(loader.dataset),
            }
        )
        (directory / "search.json").write_text(json.dumps(history, indent=2) + "\n")
        return metrics["chi_nrmse"], metrics["background_nrmse"]

    # Coordinate search limits the cost of a full 3D validation sweep.
    for name, choices in (
        ("tau", (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)),
        ("t_min", (0.1, 0.3, 0.5)),
        ("gamma", (0.5, 1.0, 2.0)),
    ):
        ranked = [(score(replace(config, **{name: value})), value) for value in choices]
        config = replace(config, **{name: min(ranked)[1]})
    schedules = ((1, 2, 5), (1, 1, 6), (2, 2, 2))
    ranked = []
    for early, middle, late in schedules:
        candidate = replace(
            config, samples_early=early, samples_middle=middle, samples_late=late
        )
        ranked.append((score(candidate), candidate))
    config = min(ranked, key=lambda row: row[0])[1]
    ranked = []
    for iterations in (20, 30, 55):
        candidate = replace(config, iterations=iterations)
        ranked.append((score(candidate), candidate))
    config = min(ranked, key=lambda row: row[0])[1]
    model.config = config
    final_metrics = evaluate_solver(model, loader, synthesizer, seed=seed + 3000001)
    manifest = {
        "format": FLOW_ENSEMBLE_FORMAT,
        "data_term": FLOW_DATA_TERM,
        "solver_type": FLOW_SOLVER,
        "chi_checkpoint": str(chi_checkpoint),
        "background_checkpoint": str(background_checkpoint),
        "solver": asdict(config),
        "validation": final_metrics,
        "chi_flow_evaluations": sum(
            config.schedule(index)[1] for index in range(config.iterations)
        ),
        "background_flow_evaluations": config.iterations,
        "validation_subjects": len(
            {entry["subject"] for entry in selected_split["validation"]}
        )
        if not max_val_subjects
        else max_val_subjects,
        "selection": "subject-mean chi NRMSE, background NRMSE tie break",
    }
    path = directory / "ensemble.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chi-checkpoint", required=True)
    parser.add_argument("--background-checkpoint", required=True)
    parser.add_argument("--run-dir")
    parser.add_argument("--data-dir", default=str(TRAINING_DATA_DIR))
    parser.add_argument("--cache-dir", default=str(TRAINING_CACHE_DIR))
    parser.add_argument("--source-manifest")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--val-views", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-val-subjects", type=int, default=0)
    args = parser.parse_args()
    run_dir = args.run_dir or str(
        RUNS_DIR / f"flow_ensemble_{datetime.now():%Y%m%d_%H%M%S_%f}"
    )
    tune_flow_admm(
        args.chi_checkpoint,
        args.background_checkpoint,
        run_dir=run_dir,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        source_manifest=args.source_manifest,
        device=args.device,
        seed=args.seed,
        val_frac=args.val_frac,
        val_views=args.val_views,
        num_workers=args.num_workers,
        max_val_subjects=args.max_val_subjects,
    )


if __name__ == "__main__":
    main()

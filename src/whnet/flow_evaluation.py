"""Evaluate a FlowADMM ensemble and WH-Net on an unseen source-subject test set."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from .data import FieldSynthesizer, TrainingVolumes, training_records
from .flow_admm import (
    FLOW_DATA_TERM,
    FLOW_SOLVER,
    _laplacian,
    interior_mask,
    load_flow_run,
    read_flow_manifest,
)
from .flow_tuning import evaluate_solver
from .paths import RUNS_DIR, TRAINING_CACHE_DIR
from .training import _sample_nrmse, load_trained_model


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate_whnet(model, loader, synthesizer) -> dict[str, float]:
    records: dict[str, list[tuple[float, float, float, float, float]]] = {}
    for batch in loader:
        phase, mask, weights, dipole, chi_target, background_target = synthesizer(batch)
        for index, subject in enumerate(batch["subject"]):
            observed = phase[index : index + 1]
            support = mask[index : index + 1]
            reliability = weights[index : index + 1]
            with (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if observed.device.type == "cuda"
                else nullcontext()
            ):
                chi, background = model(observed, support, dipole, reliability)
            field = torch.fft.irfftn(
                dipole * torch.fft.rfftn(chi.float(), dim=(-3, -2, -1)),
                s=chi.shape[-3:],
                dim=(-3, -2, -1),
            )
            interior = interior_mask(support)
            data_residual = (field + background.float() - observed) * support
            values = (
                float(
                    _sample_nrmse(chi, chi_target[index : index + 1], support).mean()
                ),
                float(
                    _sample_nrmse(
                        background, background_target[index : index + 1], support
                    ).mean()
                ),
                float(
                    (data_residual.square().sum() / support.sum().clamp_min(1)).sqrt()
                ),
                float(
                    (
                        (reliability.float() * data_residual).square().sum()
                        / support.sum().clamp_min(1)
                    ).sqrt()
                ),
                float(
                    (
                        (_laplacian(background).square() * interior).sum()
                        / interior.sum().clamp_min(1)
                    ).sqrt()
                ),
            )
            records.setdefault(subject, []).append(values)
    if not records:
        raise ValueError("empty test dataset")
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


def evaluate_held_out(
    ensemble_manifest: str | Path,
    baseline_checkpoint: str | Path,
    *,
    data_dir: str | Path,
    source_manifest: str | Path,
    output: str | Path,
    device: str = "cuda:0",
    cache_dir: str | Path = TRAINING_CACHE_DIR,
    views: int = 3,
    seed: int = 0,
) -> Path:
    if views < 1:
        raise ValueError("views must be positive")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    ensemble_manifest = Path(ensemble_manifest).resolve()
    baseline_checkpoint = Path(baseline_checkpoint).resolve()
    metadata = read_flow_manifest(ensemble_manifest)
    records = training_records(data_dir, source_manifest)
    test_subjects = {record["subject"] for record in records}
    if not test_subjects:
        raise ValueError("test set is empty")
    for checkpoint in (
        metadata["chi_checkpoint"],
        metadata["background_checkpoint"],
        baseline_checkpoint,
    ):
        split_path = Path(checkpoint).parent / "split.json"
        if not split_path.is_file():
            raise ValueError(f"missing training split for {checkpoint}")
        split = json.loads(split_path.read_text())
        seen = {
            record["subject"]
            for group in ("train", "validation")
            for record in split[group]
        }
        overlap = test_subjects & seen
        if overlap:
            raise ValueError(
                f"test subjects overlap {checkpoint}: {sorted(overlap)[:5]}"
            )
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu for a diagnostic run")
    shape = tuple(
        torch.load(
            records[0]["image"], map_location="cpu", weights_only=True, mmap=True
        ).shape[-3:]
    )
    dataset = TrainingVolumes(
        records, cache_dir, validation=True, views=views, seed=seed + 2000000
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    synthesizer = FieldSynthesizer(shape, target)
    ensemble = load_flow_run(ensemble_manifest, device=target)
    baseline = load_trained_model(baseline_checkpoint, device=target)
    _synchronize(target)
    start = time.perf_counter()
    flow_metrics = evaluate_solver(ensemble, loader, synthesizer, seed=seed + 4000000)
    _synchronize(target)
    flow_seconds = time.perf_counter() - start
    _synchronize(target)
    start = time.perf_counter()
    baseline_metrics = evaluate_whnet(baseline, loader, synthesizer)
    _synchronize(target)
    baseline_seconds = time.perf_counter() - start
    count = len(dataset)
    report = {
        "format": "whnet-flow-test-v4",
        "data_term": FLOW_DATA_TERM,
        "solver_type": FLOW_SOLVER,
        "subjects": len(test_subjects),
        "realizations": count,
        "ensemble": {
            **flow_metrics,
            "seconds_per_realization": flow_seconds / count,
            "iterations": ensemble.config.iterations,
            "chi_flow_evaluations": sum(
                ensemble.config.schedule(index)[1]
                for index in range(ensemble.config.iterations)
            ),
            "background_flow_evaluations": ensemble.config.iterations,
        },
        "baseline": {
            **baseline_metrics,
            "seconds_per_realization": baseline_seconds / count,
            "iterations": baseline.num_iters,
        },
        "ensemble_manifest": str(ensemble_manifest),
        "baseline_checkpoint": str(baseline_checkpoint),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output")
    parser.add_argument("--cache-dir", default=str(TRAINING_CACHE_DIR))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--views", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    output = args.output or str(
        RUNS_DIR / f"flow_test_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    )
    evaluate_held_out(
        args.ensemble,
        args.baseline,
        data_dir=args.data_dir,
        source_manifest=args.source_manifest,
        output=output,
        device=args.device,
        cache_dir=args.cache_dir,
        views=args.views,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

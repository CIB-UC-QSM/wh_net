"""Fit WH-TV parameters to 48-iteration WHNet susceptibility on Graz 1/2."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.stats import qmc
import torch
from torch import Tensor

from benchmark_graz import (
    DATASETS,
    DEFAULT_CHECKPOINT,
    DEFAULT_COSMOS_RUN,
    comparison_figure,
    prepare_dataset,
    run_whnet,
    save_volume,
    solution_metrics,
    synchronize,
)
from whnet.paths import RUNS_DIR
from whnet.wh_tv import WHTVResult, wh_tv

MIN_BETA = 1e6
DIVERGENCE_PENALTY = 1e6
# log10(alpha1), log10(beta), log10(mu1 / alpha1), log10(muh)
LOG_BOUNDS = np.array([[-7, 1], [6, 10], [0, 5], [-1, 5]], dtype=float)
DEFAULT_STARTING_RUN = RUNS_DIR / "graz_whnet_wh_tv_20260915_153321_072659"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cosmos-run", type=Path, default=DEFAULT_COSMOS_RUN)
    parser.add_argument("--starting-run", type=Path, default=DEFAULT_STARTING_RUN)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--whnet-iterations", type=int, default=48)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--refine-trials", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--pad", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--julia", default="/home/mathias/.juliaup/bin/julia")
    parser.add_argument(
        "--cache-dir", type=Path, default=RUNS_DIR.parent / "cache" / "graz"
    )
    parser.add_argument("--refresh-unwrapping", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if min(args.whnet_iterations, args.iterations, args.trials, args.threads) < 1:
        parser.error("iterations, trials, and threads must be positive")
    if args.refine_trials < 0 or args.pad < 0:
        parser.error("refine-trials and padding must be nonnegative")
    if args.trials < 2:
        parser.error("at least two trials are required for the seeded candidates")
    return args


def parameters(log_values: np.ndarray) -> dict[str, float]:
    alpha1, beta, ratio, muh = 10.0 ** np.asarray(log_values, dtype=float)
    return {
        "alpha1": float(alpha1),
        "beta": max(MIN_BETA, float(beta)),
        "mu1": float(alpha1 * ratio),
        "mu2": 1.0,
        "muh": float(muh),
    }


def log_parameters(values: dict[str, float]) -> np.ndarray:
    return np.log10(
        [
            values["alpha1"],
            values["beta"],
            values["mu1"] / values["alpha1"],
            values["muh"],
        ]
    )


def starting_parameters(path: Path) -> dict[str, dict[str, float]]:
    """Read parameters from either a Graz benchmark or teacher-fit run."""
    summary = json.loads((path / "summary.json").read_text())
    output = {}
    for name in DATASETS:
        dataset = summary["datasets"][name]
        if "methods" in dataset:
            values = dataset["methods"]["wh_tv_lcurve"]["parameters"]
        else:
            values = dataset["wh_tv"]["parameters"]
        output[name] = {
            key: float(values[key]) for key in ("alpha1", "beta", "mu1", "mu2", "muh")
        }
    return output


def coordinate_refinement(
    evaluate: Callable[[np.ndarray], float],
    initial: np.ndarray,
    evaluations: int,
) -> None:
    """Greedily refine log parameters within an exact evaluation budget."""
    center = np.asarray(initial, dtype=float).copy()
    step = np.array([0.75, 0.5, 0.75, 0.75], dtype=float)
    used = 0
    while used < evaluations:
        improved = False
        center_score = evaluate(center)
        for dimension in range(center.size):
            for direction in (-1.0, 1.0):
                if used >= evaluations:
                    return
                candidate = center.copy()
                candidate[dimension] += direction * step[dimension]
                candidate = np.clip(candidate, LOG_BOUNDS[:, 0], LOG_BOUNDS[:, 1])
                score = evaluate(candidate)
                used += 1
                if score < center_score:
                    center, center_score = candidate, score
                    improved = True
        if not improved:
            step *= 0.5


def discrepancy_metrics(chi: Tensor, target: Tensor, mask: Tensor) -> dict[str, float]:
    """Measure masked chi mismatch without rewarding a zero reconstruction."""
    selected = mask.bool()
    estimate = chi[selected]
    reference = target[selected]
    estimate_norm = torch.linalg.vector_norm(estimate)
    reference_norm = torch.linalg.vector_norm(reference).clamp_min(1e-12)
    error_norm = torch.linalg.vector_norm(estimate - reference)
    geometric_norm = torch.sqrt(estimate_norm.clamp_min(1e-12) * reference_norm)
    cosine = torch.dot(estimate, reference) / (
        estimate_norm * reference_norm
    ).clamp_min(1e-12)
    return {
        "teacher_chi_nrmse": float(error_norm / reference_norm),
        "symmetric_chi_discrepancy": float(error_norm / geometric_norm),
        "chi_norm_ratio": float(estimate_norm / reference_norm),
        "chi_cosine_similarity": float(cosine),
    }


def optimize_dataset(
    name: str,
    data: dict[str, Tensor],
    factor: float,
    cosmos: dict[str, float],
    starting: dict[str, float],
    args: argparse.Namespace,
    device: torch.device,
    output: Path,
) -> dict:
    destination = output / name
    destination.mkdir()
    target_chi, target_background, target_seconds = run_whnet(data, args, device)
    target_norm = float(torch.linalg.vector_norm(target_chi[data["mask"].bool()]))
    if not math.isfinite(target_norm) or target_norm <= 0:
        raise ValueError(f"WHNet target for {name} has invalid norm")
    save_volume(
        destination / "whnet_target_chi.nii.gz", target_chi, data["mask"], factor
    )
    save_volume(
        destination / "whnet_target_background.nii.gz",
        target_background,
        data["mask"],
        factor,
    )

    records: list[dict] = []
    cache: dict[tuple[float, ...], float] = {}
    best_score = float("inf")
    best_log: np.ndarray | None = None
    best_result: WHTVResult | None = None

    def evaluate(log_values: np.ndarray) -> float:
        nonlocal best_log, best_result, best_score
        bounded = np.clip(
            np.asarray(log_values, dtype=float), LOG_BOUNDS[:, 0], LOG_BOUNDS[:, 1]
        )
        key = tuple(np.round(bounded, 10))
        if key in cache:
            return cache[key]
        trial = len(records) + 1
        values = parameters(bounded)
        synchronize(device)
        started = time.perf_counter()
        try:
            result = wh_tv(
                data["phase"],
                data["dipole"],
                data["mask"],
                weight=data["weight"],
                max_iterations=args.iterations,
                tol_update=None,
                check_every=args.iterations,
                **values,
            )
            synchronize(device)
            discrepancies = discrepancy_metrics(result.chi, target_chi, data["mask"])
            score = discrepancies["symmetric_chi_discrepancy"]
            if not all(math.isfinite(value) for value in discrepancies.values()):
                raise FloatingPointError("Nonfinite teacher discrepancy")
            status, error = "complete", None
        except (FloatingPointError, torch.OutOfMemoryError) as exc:
            synchronize(device)
            result = None
            discrepancies = {}
            score = DIVERGENCE_PENALTY
            status, error = "failed", f"{type(exc).__name__}: {exc}"
            if device.type == "cuda":
                torch.cuda.empty_cache()
        elapsed = time.perf_counter() - started
        record = {
            "trial": trial,
            **values,
            "log10_alpha1": float(bounded[0]),
            "log10_beta": float(bounded[1]),
            "log10_mu1_over_alpha1": float(bounded[2]),
            "log10_muh": float(bounded[3]),
            **discrepancies,
            "objective": score,
            "seconds": elapsed,
            "status": status,
            "error": error,
        }
        records.append(record)
        with (destination / "trials.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        cache[key] = score
        if result is not None and score < best_score:
            best_score, best_log, best_result = score, bounded.copy(), result
            save_volume(
                destination / "best_wh_tv_chi.nii.gz",
                result.chi,
                data["mask"],
                factor,
            )
            save_volume(
                destination / "best_wh_tv_background.nii.gz",
                result.background,
                data["mask"],
                factor,
            )
            (destination / "best.json").write_text(json.dumps(record, indent=2) + "\n")
        print(
            f"{name} trial {trial}: chi discrepancy={score:.6g}, "
            f"best={best_score:.6g}, {elapsed:.1f}s",
            flush=True,
        )
        return score

    seeded = [log_parameters(cosmos), log_parameters(starting)]
    for candidate in seeded[: args.trials]:
        evaluate(candidate)
    remaining = args.trials - len(seeded)
    if remaining > 0:
        samples = qmc.LatinHypercube(d=4, seed=args.seed).random(remaining)
        for candidate in qmc.scale(samples, LOG_BOUNDS[:, 0], LOG_BOUNDS[:, 1]):
            evaluate(candidate)
    if args.refine_trials and best_log is not None:
        coordinate_refinement(evaluate, best_log, args.refine_trials)
    if best_result is None or best_log is None:
        raise RuntimeError(f"All WH-TV trials failed for {name}")
    best_values = parameters(best_log)
    metrics = {
        "whnet": {
            "iterations": args.whnet_iterations,
            **solution_metrics(target_chi, target_background, data, target_seconds),
        },
        "wh_tv": {
            "iterations": args.iterations,
            "parameters": best_values,
            **discrepancy_metrics(best_result.chi, target_chi, data["mask"]),
            **solution_metrics(
                best_result.chi,
                best_result.background,
                data,
                float(
                    records[
                        [record["objective"] for record in records].index(best_score)
                    ]["seconds"]
                ),
            ),
        },
        "completed_trials": len(records),
        "finite_trials": sum(record["status"] == "complete" for record in records),
    }
    comparison_figure(
        [
            ("WHNet target", target_chi, target_background),
            ("WH-TV fitted", best_result.chi, best_result.background),
        ],
        data["mask"],
        factor,
        destination / "teacher_comparison.png",
    )
    (destination / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu")
    cosmos_record = json.loads((args.cosmos_run / "best.json").read_text())
    cosmos = {
        key: float(cosmos_record[key])
        for key in ("alpha1", "beta", "mu1", "mu2", "muh")
    }
    starting = starting_parameters(args.starting_run)
    output = args.output_dir or RUNS_DIR / datetime.now(timezone.utc).strftime(
        "wh_tv_to_whnet_graz_%Y%m%d_%H%M%S_%f"
    )
    output.mkdir(parents=True, exist_ok=False)
    config = {
        "objective": (
            "masked symmetric normalized chi discrepancy between WH-TV and "
            f"{args.whnet_iterations}-iteration WHNet; conventional NRMSE reported"
        ),
        "ground_truth_objective": False,
        "checkpoint": str(args.checkpoint.resolve()),
        "cosmos_run": str(args.cosmos_run.resolve()),
        "starting_run": str(args.starting_run.resolve()),
        "iterations_per_trial": args.iterations,
        "trials": args.trials,
        "refine_trials": args.refine_trials,
        "log10_bounds": LOG_BOUNDS.tolist(),
        "min_beta": MIN_BETA,
        "mu2": 1.0,
        "seed": args.seed,
        "device": str(device),
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    results = {}
    try:
        for name, (path, factor) in DATASETS.items():
            data, preprocessing = prepare_dataset(name, path, factor, args, device)
            result = optimize_dataset(
                name,
                data,
                factor,
                cosmos,
                starting[name],
                args,
                device,
                output,
            )
            results[name] = {"preprocessing": preprocessing, **result}
    except BaseException as exc:
        (output / "status.json").write_text(
            json.dumps({"status": "failed", "error": str(exc)}, indent=2) + "\n"
        )
        raise
    summary = {"config": config, "datasets": results}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "status.json").write_text(
        json.dumps({"status": "complete"}, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved WH-TV teacher-fit optimization to {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()

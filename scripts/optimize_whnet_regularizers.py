"""Tune fixed learned WHNet regularizer gains on COSMOS over 55 iterations."""

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
import torch

from whnet.learned_regularizers import (
    extract_fixed_regularizer_model,
    set_regularizer_weights,
)
from whnet.paths import CHECKPOINTS_DIR, RUNS_DIR

ITERATIONS = 100
# log10 fixed gain for susceptibility and harmonic-background learned residuals.
LOG_BOUNDS = np.array([[-5.0, 0.0], [-3.0, 0.5]], dtype=float)
DIVERGENCE_PENALTY = 1e6
DEFAULT_CHECKPOINT = CHECKPOINTS_DIR / "checkpoints_scratch7" / "model_best.pth"


def find_simulation() -> Path:
    candidates = []
    for path in RUNS_DIR.glob("wh_tv_cosmos_*/simulation.pt"):
        config_path, status_path = (
            path.parent / "config.json",
            path.parent / "status.json",
        )
        if not config_path.is_file() or not status_path.is_file():
            continue
        config, status = (
            json.loads(config_path.read_text()),
            json.loads(status_path.read_text()),
        )
        if (
            config.get("snr") == 100
            and config.get("bg") == 10
            and status.get("status") == "complete"
        ):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            "No completed COSMOS SNR=100/BG=10 simulation found; pass --simulation"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def weights(log_values: np.ndarray) -> tuple[float, float]:
    values = 10.0 ** np.asarray(log_values, dtype=float)
    return float(values[0]), float(values[1])


def nrmse(
    prediction: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor
) -> float:
    selected = mask.bool()
    return float(
        torch.linalg.vector_norm((prediction - reference)[selected])
        / torch.linalg.vector_norm(reference[selected]).clamp_min(1e-12)
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--simulation", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--trials",
        type=int,
        default=24,
        help="Latin-hypercube evaluations including the baseline",
    )
    parser.add_argument("--refine-trials", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--channels-last", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.trials < 1 or args.refine_trials < 0 or args.threads < 1:
        parser.error(
            "trials and threads must be positive; refine-trials must be nonnegative"
        )
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu")
    if args.bf16 and device.type != "cuda":
        raise ValueError("Use --no-bf16 for CPU optimization")
    simulation_path = args.simulation or find_simulation()
    data = torch.load(simulation_path, map_location=device, weights_only=True)
    required = {"phase", "mask", "dipole", "chi", "background"}
    if missing := required - data.keys():
        raise KeyError(f"Simulation is missing {sorted(missing)}")
    config_path = simulation_path.parent / "config.json"
    simulation_config = (
        json.loads(config_path.read_text()) if config_path.is_file() else {}
    )
    if simulation_config and (
        simulation_config.get("snr") != 100 or simulation_config.get("bg") != 10
    ):
        raise ValueError("Simulation must use SNR=100 and BG=10")

    # Medians of scratch7's learned gains at iterations 0:54 with update=0.
    baseline = np.log10(np.array([0.0068720924, 0.14581226]))
    chi_weight, background_weight = weights(baseline)
    model = extract_fixed_regularizer_model(
        args.checkpoint,
        chi_weight,
        background_weight,
        device=device,
        num_iters=ITERATIONS,
    )
    if args.channels_last:
        model.to(memory_format=torch.channels_last_3d)
    if args.compile:
        model.chinet.compile()
        model.phinet.compile()
    phase, mask, dipole = data["phase"], data["mask"], data["dipole"]
    if args.channels_last:
        phase = phase.contiguous(memory_format=torch.channels_last_3d)
        mask = mask.contiguous(memory_format=torch.channels_last_3d)
    autocast = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if args.bf16
        else nullcontext
    )
    if args.compile:
        with torch.inference_mode(), autocast():
            model(phase, mask, dipole, mask)
        torch.cuda.synchronize(device)

    output = args.output_dir or RUNS_DIR / datetime.now(timezone.utc).strftime(
        "whnet_fixed_regularizers_%Y%m%d_%H%M%S_%f"
    )
    output.mkdir(parents=True, exist_ok=False)
    run_config = {
        "checkpoint": str(args.checkpoint.resolve()),
        "simulation": str(simulation_path.resolve()),
        "snr": simulation_config.get("snr", 100),
        "bg": simulation_config.get("bg", 10),
        "iterations": ITERATIONS,
        "log10_bounds": LOG_BOUNDS.tolist(),
        "trials": args.trials,
        "refine_trials": args.refine_trials,
        "seed": args.seed,
        "objective": "masked susceptibility NRMSE",
        "bf16": args.bf16,
        "channels_last_3d": args.channels_last,
        "compiled_proximals": args.compile,
        "dynamic_mlp_predictors": False,
    }
    (output / "config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    best_score, best_log, completed, failed = float("inf"), None, 0, 0

    def evaluate(log_values: np.ndarray) -> float:
        nonlocal best_score, best_log, completed, failed
        fixed_chi, fixed_background = weights(log_values)
        set_regularizer_weights(model, fixed_chi, fixed_background)
        completed += 1
        started = time.perf_counter()
        try:
            with torch.inference_mode(), autocast():
                chi, background = model(phase, mask, dipole, mask)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if not bool(torch.isfinite(chi).all() and torch.isfinite(background).all()):
                raise FloatingPointError("nonfinite fixed-regularizer reconstruction")
            chi_score = nrmse(chi, data["chi"], mask)
            background_score = nrmse(background, data["background"], mask)
            if not np.isfinite(chi_score) or not np.isfinite(background_score):
                raise FloatingPointError("nonfinite NRMSE")
        except (FloatingPointError, torch.OutOfMemoryError) as exc:
            failed += 1
            if device.type == "cuda":
                torch.cuda.empty_cache()
            record = {
                "trial": completed,
                "status": "failed",
                "chi_weight": fixed_chi,
                "background_weight": fixed_background,
                "objective_penalty": DIVERGENCE_PENALTY,
                "error": f"{type(exc).__name__}: {exc}",
                "seconds": time.perf_counter() - started,
            }
            with (output / "trials.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(f"Trial {completed}: failed ({exc}); continuing", flush=True)
            return DIVERGENCE_PENALTY
        record = {
            "trial": completed,
            "status": "complete",
            "chi_weight": fixed_chi,
            "background_weight": fixed_background,
            "chi_nrmse": chi_score,
            "background_nrmse": background_score,
            "seconds": time.perf_counter() - started,
        }
        with (output / "trials.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if chi_score < best_score:
            best_score, best_log = chi_score, np.asarray(log_values).copy()
            (output / "best.json").write_text(json.dumps(record, indent=2) + "\n")
            torch.save(
                {"chi": chi.cpu(), "background": background.cpu()},
                output / "best_solution.pt",
            )
        print(
            f"Trial {completed}: chi={chi_score:.6f}, background={background_score:.6f}, "
            f"best={best_score:.6f}",
            flush=True,
        )
        return chi_score

    try:
        evaluate(baseline)
        if args.trials > 1:
            samples = qmc.LatinHypercube(2, seed=args.seed).random(args.trials - 1)
            for candidate in qmc.scale(samples, LOG_BOUNDS[:, 0], LOG_BOUNDS[:, 1]):
                evaluate(candidate)
        if args.refine_trials and best_log is not None:
            minimize(
                evaluate,
                best_log,
                method="Powell",
                bounds=LOG_BOUNDS,
                options={"maxfev": args.refine_trials, "xtol": 0.02, "ftol": 1e-4},
            )
    except BaseException as exc:
        (output / "status.json").write_text(
            json.dumps(
                {
                    "status": "interrupted"
                    if isinstance(exc, KeyboardInterrupt)
                    else "failed",
                    "completed_trials": completed,
                    "failed_trials": failed,
                    "error": str(exc),
                },
                indent=2,
            )
            + "\n"
        )
        raise
    status = "complete" if best_log is not None else "no_finite_solution"
    (output / "status.json").write_text(
        json.dumps(
            {
                "status": status,
                "completed_trials": completed,
                "failed_trials": failed,
                "best_chi_nrmse": None if best_log is None else best_score,
            },
            indent=2,
        )
        + "\n"
    )
    if best_log is None:
        raise RuntimeError("All fixed-weight trials failed; see trials.jsonl")
    print(f"Saved optimization to {output}")


if __name__ == "__main__":
    main()

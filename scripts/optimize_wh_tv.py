"""Tune WH-TV on COSMOS at SNR=100, BG=10 and 2,000 iterations per solve."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from scipy.stats import qmc
import torch
from torch import Tensor
import torch.nn.functional as F

from whnet.paths import EVALUATION_DATA_DIR, RUNS_DIR
from whnet.physics import continuous_dipole_kernel
from whnet.wh_tv import wh_tv

ITERATIONS = 10000
MIN_BETA = 1e6
DIVERGENCE_PENALTY = 1e6
# log10(alpha1), log10(beta), log10(mu1 / alpha1), log10(muh)
LOG_BOUNDS = np.array([[-6, -1], [6, 9], [0, 3], [-1, 5]], dtype=float)


def parameters(log_values: np.ndarray) -> dict[str, float]:
    alpha1, beta, ratio, muh = 10.0 ** np.asarray(log_values, dtype=float)
    return dict(
        alpha1=float(alpha1),
        beta=max(MIN_BETA, float(beta)),
        mu1=float(alpha1 * ratio),
        mu2=1.0,
        muh=float(muh),
    )


def cosmos_simulation(
    chi_path: Path,
    mask_path: Path,
    *,
    device: torch.device,
    seed: int = 0,
    pad: int = 12,
    factor: float = 0.2125,
) -> dict[str, Tensor]:
    """Match cosmos.py: peak-field SNR=100, exterior BG=10 in solver units.

    CPU noise generation makes the experiment independent of the solve device.
    No outlier injection. Multiply solver-unit sources by factor for ppm.
    """
    if pad < 0 or not np.isfinite(factor) or factor <= 0:
        raise ValueError("pad must be nonnegative and factor finite and positive")
    chi = np.asarray(loadmat(chi_path)["chi_cosmos"], dtype=np.float32)
    mask = np.asarray(loadmat(mask_path)["msk"], dtype=np.float32)
    if chi.ndim != 3 or chi.shape != mask.shape or min(chi.shape) < 2:
        raise ValueError("COSMOS chi_cosmos and msk must be matching 3D arrays")
    if not np.isfinite(chi).all() or not np.isfinite(mask).all():
        raise ValueError("COSMOS arrays must be finite")
    if not np.isin(mask, [0, 1]).all() or not mask.any():
        raise ValueError("COSMOS mask must be binary and nonempty")
    size = max(chi.shape) + 2 * pad
    padding = tuple((pad, size - n - pad) for n in chi.shape)
    chi = torch.from_numpy(np.pad(chi, padding))[None, None] / factor
    mask = torch.from_numpy(np.pad(mask, padding))[None, None]
    dipole = torch.from_numpy(continuous_dipole_kernel((size,) * 3)).float()[None, None]
    spatial = (-3, -2, -1)

    def field(source: Tensor) -> Tensor:
        return torch.fft.ifftn(
            dipole * torch.fft.fftn(source, dim=spatial), dim=spatial
        ).real

    local = field(chi * mask) * mask
    dilated = F.max_pool3d(mask, 7, stride=1, padding=3)
    background = field((1 - dilated) * 10.0) * mask
    noise_std = local[mask.bool()].abs().max() / 100.0
    if noise_std == 0:
        raise ValueError("COSMOS source has zero local field")
    noise = (
        torch.randn(chi.shape, generator=torch.Generator().manual_seed(seed))
        * noise_std
    )
    return {
        name: value.to(device)
        for name, value in dict(
            phase=(local + background + noise) * mask,
            mask=mask,
            dipole=dipole,
            chi=chi,
            background=background,
            noise_std=noise_std,
        ).items()
    }


def nrmse(pred: Tensor, reference: Tensor, mask: Tensor) -> float:
    """Masked NRMSE as a fraction; no fitted offsets or scale factors."""
    selected = mask.bool()
    return float(
        torch.linalg.vector_norm((pred - reference)[selected])
        / torch.linalg.vector_norm(reference[selected]).clamp_min(1e-12)
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chi", type=Path, default=EVALUATION_DATA_DIR / "chi_cosmos.mat"
    )
    parser.add_argument("--mask", type=Path, default=EVALUATION_DATA_DIR / "msk.mat")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--trials", type=int, default=32, help="Global evaluations including baseline"
    )
    parser.add_argument(
        "--refine-trials",
        type=int,
        default=16,
        help="Additional bounded Powell evaluations",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pad", type=int, default=12)
    parser.add_argument("--factor", type=float, default=0.2125)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--initial-solution",
        type=Path,
        help="Tensor dictionary: chi and background on padded grid, in solver units",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New directory; existing directories are refused",
    )
    args = parser.parse_args(argv)
    if args.trials < 1 or args.refine_trials < 0 or args.threads < 1:
        parser.error(
            "trials and threads must be positive; refine-trials must be nonnegative"
        )
    if args.pad < 0 or not np.isfinite(args.factor) or args.factor <= 0:
        parser.error("pad must be nonnegative; factor must be finite and positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable; pass --device cpu explicitly to use CPU"
            )
        free, total = torch.cuda.mem_get_info(device)
        print(f"GPU free/total: {free / 2**30:.2f}/{total / 2**30:.2f} GiB", flush=True)
    data = cosmos_simulation(
        args.chi,
        args.mask,
        device=device,
        seed=args.seed,
        pad=args.pad,
        factor=args.factor,
    )
    initial = None
    if args.initial_solution is not None:
        loaded = torch.load(
            args.initial_solution, map_location=device, weights_only=True
        )
        initial = (loaded["chi"], loaded["background"])
        for tensor in initial:
            if tensor.shape != data["phase"].shape or not bool(
                torch.isfinite(tensor).all()
            ):
                raise ValueError(
                    "Initial solutions must be finite and match the padded simulation shape"
                )
    output = args.output_dir or RUNS_DIR / datetime.now(timezone.utc).strftime(
        "wh_tv_cosmos_%Y%m%d_%H%M%S_%f"
    )
    output.mkdir(parents=True, exist_ok=False)

    def fingerprint(path: Path) -> dict[str, str]:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        return dict(path=str(path.resolve()), sha256=digest)

    config = dict(
        snr=100,
        snr_definition="max(abs(local[mask])) / noise_std",
        bg=10,
        bg_units="solver units; multiply by factor for physical susceptibility",
        physical_bg=10 * args.factor,
        factor=args.factor,
        pad=args.pad,
        iterations=ITERATIONS,
        min_beta=MIN_BETA,
        mu2=1.0,
        log10_bounds=LOG_BOUNDS.tolist(),
        seed=args.seed,
        trials=args.trials,
        refine_trials=args.refine_trials,
        device=str(device),
        torch=torch.__version__,
        chi_source=fingerprint(args.chi),
        mask_source=fingerprint(args.mask),
        initial_solution=None
        if args.initial_solution is None
        else fingerprint(args.initial_solution),
        shape=list(data["phase"].shape),
        noise_std=float(data["noise_std"]),
        objective="masked chi NRMSE fraction; development/tuning score",
        early_stopping=False,
    )
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    torch.save({k: v.cpu() for k, v in data.items()}, output / "simulation.pt")
    best_score, best_log, completed, failed = float("inf"), None, 0, 0

    def evaluate(log_values: np.ndarray) -> float:
        nonlocal best_score, best_log, completed, failed
        params = parameters(log_values)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        completed += 1
        try:
            result = wh_tv(
                data["phase"],
                data["dipole"],
                data["mask"],
                initial,
                weight=data["mask"],
                max_iterations=ITERATIONS,
                tol_update=None,
                **params,
            )
            score = nrmse(result.chi, data["chi"], data["mask"])
            bg_score = nrmse(result.background, data["background"], data["mask"])
            if not np.isfinite(score) or not np.isfinite(bg_score):
                raise FloatingPointError("Nonfinite objective")
        except (FloatingPointError, torch.OutOfMemoryError) as exc:
            failed += 1
            if device.type == "cuda":
                torch.cuda.empty_cache()
            record = dict(
                trial=completed,
                **params,
                chi_nrmse=None,
                background_nrmse=None,
                objective_penalty=DIVERGENCE_PENALTY,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                seconds=time.perf_counter() - started,
            )
            with (output / "trials.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(
                f"Trial {completed}: failed ({exc}); continuing with penalty",
                flush=True,
            )
            return DIVERGENCE_PENALTY
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        record = dict(
            trial=completed,
            **params,
            chi_nrmse=score,
            background_nrmse=bg_score,
            status="complete",
            iterations=result.iterations,
            update_percent=result.update_percent,
            seconds=time.perf_counter() - started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
        )
        with (output / "trials.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if score < best_score:
            best_score, best_log = score, np.asarray(log_values).copy()
            (output / "best.json").write_text(json.dumps(record, indent=2) + "\n")
            torch.save(
                dict(
                    chi=result.chi.cpu(),
                    background=result.background.cpu(),
                    chi_physical=(result.chi * args.factor).cpu(),
                    background_physical=(result.background * args.factor).cpu(),
                ),
                output / "best_solution.pt",
            )
        duration = record["seconds"]
        print(
            f"Trial {completed}: chi NRMSE={score:.6f}, background={bg_score:.6f}, "
            f"best={best_score:.6f}, {duration:.1f}s",
            flush=True,
        )
        return score

    try:
        evaluate(np.array([-3.0, 6.0, 2.0, np.log10(MIN_BETA / 50)]))
        if args.trials > 1:
            samples = qmc.LatinHypercube(d=4, seed=args.seed).random(args.trials - 1)
            for candidate in qmc.scale(samples, LOG_BOUNDS[:, 0], LOG_BOUNDS[:, 1]):
                evaluate(candidate)
        if args.refine_trials and best_log is not None:
            minimize(
                evaluate,
                best_log,
                method="Powell",
                bounds=LOG_BOUNDS,
                options={"maxfev": args.refine_trials, "xtol": 0.05, "ftol": 1e-4},
            )
    except BaseException as exc:
        (output / "status.json").write_text(
            json.dumps(
                dict(
                    status="interrupted"
                    if isinstance(exc, KeyboardInterrupt)
                    else "failed",
                    completed_trials=completed,
                    failed_trials=failed,
                    error=str(exc),
                ),
                indent=2,
            )
            + "\n"
        )
        raise
    status = "complete" if best_log is not None else "no_finite_solution"
    (output / "status.json").write_text(
        json.dumps(
            dict(
                status=status,
                completed_trials=completed,
                failed_trials=failed,
                best_chi_nrmse=None if best_log is None else best_score,
            ),
            indent=2,
        )
        + "\n"
    )
    if best_log is None:
        raise RuntimeError("All hyperparameter trials diverged; see trials.jsonl")
    print(f"Results: {output}", flush=True)


if __name__ == "__main__":
    main()

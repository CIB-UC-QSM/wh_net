"""Evaluate a tuned FlowADMM ensemble on the existing COSMOS and sim2 cases."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Iterable

import numpy as np
import torch

from .adaptive_wh_tv_validation import (
    ExternalValidationCase,
    _crop_tensor,
    comparison_figure,
    load_external_cases,
    masked_nrmse,
)
from .flow_admm import FLOW_DATA_TERM, FLOW_SOLVER, FlowADMMNet, load_flow_run
from .paths import RUNS_DIR


def _background_figure(reconstruction: np.ndarray, *, title: str):
    """Show a background estimate when no reference is available (sim2)."""
    import matplotlib.pyplot as plt

    center = tuple(length // 2 for length in reconstruction.shape)
    slices = (
        reconstruction[center[0]],
        reconstruction[:, center[1]],
        reconstruction[:, :, center[2]],
    )
    limit = max(float(np.percentile(np.abs(reconstruction), 99.5)), 1e-7)
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
    figure.suptitle(title)
    for axis, view in zip(axes, slices, strict=True):
        axis.imshow(np.rot90(view), cmap="gray", vmin=-limit, vmax=limit)
        axis.axis("off")
    return figure


@torch.inference_mode()
def evaluate_flow_external_cases(
    model: FlowADMMNet,
    cases: Iterable[ExternalValidationCase],
    *,
    device: str | torch.device,
    output_dir: str | Path,
    seed: int = 0,
) -> dict[str, dict]:
    """Save physical-unit volumes, orthogonal views, and masked reference NRMSE."""
    import matplotlib.pyplot as plt
    import nibabel as nib

    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"use a fresh evaluation directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    model.to(target).eval()
    results: dict[str, dict] = {}
    for case in cases:
        if case.phase.shape != case.mask.shape or case.chi.shape != case.mask.shape:
            raise ValueError(f"{case.name} tensors have inconsistent shapes")
        case_dir = destination / case.name
        case_dir.mkdir()
        phase = case.phase.to(target)
        mask = case.mask.to(target)
        dipole = case.dipole.to(target)
        if target.type == "cuda":
            torch.cuda.synchronize(target)
        started = time.perf_counter()
        chi, background, trace = model(
            phase, mask, dipole, weights=mask, seed=seed, return_diagnostics=True
        )
        if target.type == "cuda":
            torch.cuda.synchronize(target)
        duration = time.perf_counter() - started
        if not bool(torch.isfinite(chi).all() and torch.isfinite(background).all()):
            raise FloatingPointError(f"nonfinite {case.name} reconstruction")
        chi_reference = case.chi.to(target)
        metrics = {
            "data_term": FLOW_DATA_TERM,
            "solver_type": FLOW_SOLVER,
            "chi_nrmse": masked_nrmse(chi, chi_reference, mask),
            "inference_seconds": duration,
            "solver_iterations": len(trace),
            "factor": case.factor,
            "fidelity_weights": "binary_mask",
            "fidelity_weight_convention": "residual_multiplier",
            "metadata": case.metadata,
            "final_solver_diagnostics": trace[-1],
        }
        arrays = {
            "phase": _crop_tensor(phase * case.factor, case.output_pad),
            "mask": _crop_tensor(mask, case.output_pad),
            "chi_reference": _crop_tensor(chi_reference * case.factor, case.output_pad),
            "chi_reconstruction": _crop_tensor(chi * case.factor, case.output_pad),
            "chi_error": _crop_tensor(
                (chi - chi_reference) * mask * case.factor, case.output_pad
            ),
            "background_reconstruction": _crop_tensor(
                background * case.factor, case.output_pad
            ),
        }
        if case.background is not None:
            background_reference = case.background.to(target)
            metrics["background_nrmse"] = masked_nrmse(
                background, background_reference, mask
            )
            arrays["background_reference"] = _crop_tensor(
                background_reference * case.factor, case.output_pad
            )
            arrays["background_error"] = _crop_tensor(
                (background - background_reference) * mask * case.factor,
                case.output_pad,
            )
        for name, array in arrays.items():
            nib.save(nib.Nifti1Image(array, np.eye(4)), case_dir / f"{name}.nii.gz")
        figures = {
            "chi_comparison": comparison_figure(
                arrays["chi_reference"],
                arrays["chi_reconstruction"],
                arrays["chi_error"],
                title=f"FlowADMM {case.name} susceptibility, NRMSE={metrics['chi_nrmse']:.4f}",
                method_label="FlowADMM",
            )
        }
        if case.background is not None:
            figures["background_comparison"] = comparison_figure(
                arrays["background_reference"],
                arrays["background_reconstruction"],
                arrays["background_error"],
                title=(
                    f"FlowADMM {case.name} background, "
                    f"NRMSE={metrics['background_nrmse']:.4f}"
                ),
                method_label="FlowADMM",
            )
        else:
            figures["background_reconstruction"] = _background_figure(
                arrays["background_reconstruction"],
                title=f"FlowADMM {case.name} background (no reference)",
            )
        for name, figure in figures.items():
            figure.savefig(case_dir / f"{name}.png", dpi=180)
            plt.close(figure)
        (case_dir / "solver_trace.json").write_text(json.dumps(trace, indent=2) + "\n")
        (case_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        results[case.name] = metrics
        del phase, mask, dipole, chi, background
    (destination / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble", required=True)
    parser.add_argument("--run-dir")
    parser.add_argument(
        "--datasets", nargs="+", choices=("cosmos", "sim2"), default=("cosmos", "sim2")
    )
    parser.add_argument("--cosmos-simulation")
    parser.add_argument("--sim2-factor", type=float, default=0.264)
    parser.add_argument("--sim2-pad", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else RUNS_DIR / f"flow_external_{datetime.now():%Y%m%d_%H%M%S_%f}"
    )
    model = load_flow_run(args.ensemble, device=args.device)
    cases = load_external_cases(
        args.datasets,
        cosmos_simulation=args.cosmos_simulation,
        sim2_factor=args.sim2_factor,
        sim2_pad=args.sim2_pad,
    )
    results = evaluate_flow_external_cases(
        model, cases, device=args.device, output_dir=run_dir, seed=args.seed
    )
    (run_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "format": "whnet-flow-external-v4",
                "data_term": FLOW_DATA_TERM,
                "solver_type": FLOW_SOLVER,
                "ensemble": str(Path(args.ensemble).resolve()),
                "seed": args.seed,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"run_dir": str(run_dir.resolve()), "results": results}, indent=2))


if __name__ == "__main__":
    main()

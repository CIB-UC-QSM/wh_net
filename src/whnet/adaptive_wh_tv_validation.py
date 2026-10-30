"""Shared COSMOS and sim2 inputs and adaptive WH-TV validation.

The loaders intentionally reproduce the inputs used by the existing WHNet
evaluations. COSMOS uses the completed SNR=100/background=10 simulation, while
sim2 uses the legacy crop, scale factor, padding, and continuous dipole kernel.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import torch
from torch import Tensor

from .paths import EVALUATION_DATA_DIR, EVALUATION_VOLUMES_DIR, RUNS_DIR
from .physics import continuous_dipole_kernel

if TYPE_CHECKING:
    from .adaptive_wh_tv import AdaptiveWHTV

COSMOS_REQUIRED_KEYS = {"phase", "mask", "dipole", "chi", "background"}


@dataclass(frozen=True)
class ExternalValidationCase:
    """One WHNet-compatible external validation volume, held on the CPU."""

    name: str
    phase: Tensor
    mask: Tensor
    dipole: Tensor
    chi: Tensor
    background: Tensor | None
    factor: float
    weight: Tensor | None = None
    output_pad: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExternalValidationResult:
    metrics: dict[str, float]
    details: dict[str, dict[str, Any]]
    figures: dict[str, Any]


def find_cosmos_simulation() -> Path:
    """Find the newest completed WHNet COSMOS SNR=100/background=10 input."""
    candidates: list[Path] = []
    for path in RUNS_DIR.glob("wh_tv_cosmos_*/simulation.pt"):
        config_path = path.parent / "config.json"
        status_path = path.parent / "status.json"
        if not config_path.is_file() or not status_path.is_file():
            continue
        config = json.loads(config_path.read_text())
        status = json.loads(status_path.read_text())
        if (
            config.get("snr") == 100
            and config.get("bg") == 10
            and status.get("status") == "complete"
        ):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            "No completed COSMOS SNR=100/background=10 simulation was found"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _validate_tensor(name: str, value: Tensor, *, binary: bool = False) -> Tensor:
    value = torch.as_tensor(value).detach().cpu().float().contiguous()
    if value.ndim != 5 or value.shape[0] != 1 or value.shape[1] != 1:
        raise ValueError(f"{name} must have shape [1,1,D,H,W], got {value.shape}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains nonfinite values")
    if binary and not bool(((value == 0) | (value == 1)).all()):
        raise ValueError(f"{name} must be binary")
    return value


def load_cosmos_case(simulation: str | Path | None = None) -> ExternalValidationCase:
    """Load the exact completed COSMOS case used by ``evaluate_whnet.py``."""
    path = Path(simulation) if simulation else find_cosmos_simulation()
    data = torch.load(path, map_location="cpu", weights_only=True)
    if missing := COSMOS_REQUIRED_KEYS - data.keys():
        raise KeyError(f"COSMOS simulation is missing {sorted(missing)}")
    config_path = path.parent / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    if config and (config.get("snr") != 100 or config.get("bg") != 10):
        raise ValueError("COSMOS validation requires SNR=100 and background=10")
    tensors = {
        key: _validate_tensor(key, data[key], binary=key == "mask")
        for key in COSMOS_REQUIRED_KEYS
    }
    spatial_shapes = {tuple(value.shape[-3:]) for value in tensors.values()}
    if len(spatial_shapes) != 1:
        raise ValueError(f"COSMOS tensors have inconsistent shapes: {spatial_shapes}")
    return ExternalValidationCase(
        name="cosmos",
        phase=tensors["phase"],
        mask=tensors["mask"],
        dipole=tensors["dipole"],
        chi=tensors["chi"],
        background=tensors["background"],
        factor=float(config.get("factor", 0.2125)),
        output_pad=int(config.get("pad", 0)),
        metadata={
            "simulation": str(path.resolve()),
            "snr": int(config.get("snr", 100)),
            "snr_definition": config.get(
                "snr_definition", "max(abs(local[mask])) / noise_std"
            ),
            "background_source_amplitude": int(config.get("bg", 10)),
            "preparation": "saved WHNet COSMOS simulation",
        },
    )


def _center_pad(volume: np.ndarray, zero_pad: int) -> np.ndarray:
    """Match the legacy sim2 ``pad_to_sqr_shape`` centering exactly."""
    if volume.ndim != 3 or zero_pad < 0:
        raise ValueError("Expected a 3D volume and nonnegative padding")
    maximum = int(max(volume.shape))
    size = maximum + 2 * zero_pad
    result = np.zeros((size, size, size), dtype=np.float32)
    offsets = tuple((maximum - length) // 2 + zero_pad for length in volume.shape)
    slices = tuple(
        slice(offset, offset + length)
        for offset, length in zip(offsets, volume.shape, strict=True)
    )
    result[slices] = np.asarray(volume, dtype=np.float32)
    return result


def load_sim2_case(*, factor: float = 0.264, pad: int = 20) -> ExternalValidationCase:
    """Load sim2 using the same preprocessing as ``experiments/evaluation/sim2.py``."""
    import nibabel as nib
    from scipy.io import loadmat

    if factor <= 0 or pad < 0:
        raise ValueError("sim2 factor must be positive and padding nonnegative")
    phase_path = EVALUATION_DATA_DIR / "Sim2.mat"
    mask_path = EVALUATION_DATA_DIR / "mask_final.mat"
    weight_path = EVALUATION_DATA_DIR / "sim2_w.mat"
    chi_path = EVALUATION_VOLUMES_DIR / "Sim2ChiGT.nii.gz"
    for path in (phase_path, mask_path, weight_path, chi_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    phase = np.asarray(loadmat(phase_path)["phase"], dtype=np.float32)
    mask = np.asarray(loadmat(mask_path)["mask_final"], dtype=np.float32)
    # The adaptive model estimates fidelity precision; WHNet uses this map.
    legacy_weight = np.asarray(loadmat(weight_path)["w"], dtype=np.float32)
    chi = np.asarray(nib.load(chi_path).get_fdata(dtype=np.float32))
    if mask.shape != chi.shape or mask.shape != legacy_weight.shape:
        raise ValueError(
            "sim2 mask, susceptibility reference, and WHNet weights must align"
        )
    if phase.ndim != 3 or any(
        phase.shape[axis] < mask.shape[axis] for axis in range(3)
    ):
        raise ValueError("sim2 phase does not cover the legacy mask coordinates")
    if not np.isfinite(phase).all() or not np.isfinite(chi).all():
        raise ValueError("sim2 contains nonfinite values")
    if not np.isfinite(legacy_weight).all() or (legacy_weight < 0).any():
        raise ValueError("sim2 WHNet weights must be finite and nonnegative")
    if not np.isin(mask, (0, 1)).all() or not mask.any():
        raise ValueError("sim2 mask must be nonempty and binary")
    coordinates = np.argwhere(mask)
    lower = coordinates.min(axis=0)
    upper = coordinates.max(axis=0) + 1
    crop = tuple(
        slice(int(start), int(stop)) for start, stop in zip(lower, upper, strict=True)
    )
    phase = _center_pad(phase[crop], pad) / factor
    mask = _center_pad(mask[crop], pad)
    chi = _center_pad(chi[crop], pad) / factor
    legacy_weight = _center_pad(legacy_weight[crop], pad)
    if phase.shape != mask.shape or phase.shape != chi.shape:
        raise ValueError("sim2 preprocessing produced inconsistent shapes")
    if bool(np.any((legacy_weight > 0) & (mask == 0))):
        raise ValueError("sim2 WHNet weights extend outside the evaluation mask")
    shape = tuple(int(value) for value in phase.shape)
    dipole = continuous_dipole_kernel(shape).astype(np.float32, copy=False)

    def tensor(value: np.ndarray) -> Tensor:
        return torch.from_numpy(np.ascontiguousarray(value))[None, None]

    return ExternalValidationCase(
        name="sim2",
        phase=tensor(phase),
        mask=tensor(mask),
        dipole=tensor(dipole),
        chi=tensor(chi),
        background=None,
        factor=float(factor),
        weight=tensor(legacy_weight),
        output_pad=0,
        metadata={
            "phase": str(phase_path.resolve()),
            "mask": str(mask_path.resolve()),
            "legacy_whnet_weight": str(weight_path.resolve()),
            "chi_reference": str(chi_path.resolve()),
            "factor": float(factor),
            "pad": int(pad),
            "mask_bbox_start": lower.tolist(),
            "mask_bbox_stop": upper.tolist(),
            "preparation": "legacy WHNet sim2 crop and centered cube padding",
            "background_reference": "unavailable",
        },
    )


def load_external_cases(
    datasets: Iterable[str] = ("cosmos", "sim2"),
    *,
    cosmos_simulation: str | Path | None = None,
    sim2_factor: float = 0.264,
    sim2_pad: int = 20,
) -> tuple[ExternalValidationCase, ...]:
    names = tuple(datasets)
    unknown = set(names) - {"cosmos", "sim2"}
    if unknown or not names:
        raise ValueError(f"Invalid external datasets: {sorted(unknown)}")
    cases = []
    for name in names:
        if name == "cosmos":
            cases.append(load_cosmos_case(cosmos_simulation))
        else:
            cases.append(load_sim2_case(factor=sim2_factor, pad=sim2_pad))
    return tuple(cases)


def physical_external_cases(
    cases: Iterable[ExternalValidationCase],
) -> tuple[ExternalValidationCase, ...]:
    """Restore stored solver-unit inputs to physical units before inference."""
    physical = []
    for case in cases:
        if not np.isfinite(case.factor) or case.factor <= 0:
            raise ValueError(f"{case.name} has an invalid susceptibility factor")
        if case.factor == 1:
            physical.append(case)
            continue
        physical.append(
            replace(
                case,
                phase=case.phase * case.factor,
                chi=case.chi * case.factor,
                background=(
                    None if case.background is None else case.background * case.factor
                ),
                factor=1.0,
                metadata={**case.metadata, "stored_solver_factor": case.factor},
            )
        )
    return tuple(physical)


def masked_nrmse(prediction: Tensor, reference: Tensor, mask: Tensor) -> float:
    selected = mask.bool()
    return float(
        torch.linalg.vector_norm((prediction.float() - reference.float())[selected])
        / torch.linalg.vector_norm(reference.float()[selected]).clamp_min(1e-12)
    )


def _crop_tensor(value: Tensor, pad: int) -> np.ndarray:
    array = value.detach().float().cpu().squeeze().numpy()
    if array.ndim != 3:
        raise ValueError(f"Expected one 3D volume, got {array.shape}")
    if pad:
        if min(array.shape) <= 2 * pad:
            raise ValueError("Output padding exceeds the validation volume")
        array = array[pad:-pad, pad:-pad, pad:-pad]
    return np.asarray(array, dtype=np.float32)


def comparison_figure(
    reference: np.ndarray,
    reconstruction: np.ndarray,
    error: np.ndarray,
    *,
    title: str,
    method_label: str = "Adaptive WH-TV",
    display_limit: float | None = None,
):
    """Show orthogonal ground-truth, reconstruction, and difference slices."""
    import matplotlib.pyplot as plt

    center = tuple(size // 2 for size in reference.shape)
    selectors = (
        lambda value: value[center[0]],
        lambda value: value[:, center[1]],
        lambda value: value[:, :, center[2]],
    )
    if display_limit is not None and display_limit <= 0:
        raise ValueError("display_limit must be positive")
    image_limit = (
        display_limit
        if display_limit is not None
        else max(
            float(
                np.percentile(
                    np.abs(np.concatenate((reference.ravel(), reconstruction.ravel()))),
                    99.5,
                )
            ),
            float(np.finfo(np.float32).eps),
        )
    )
    error_limit = (
        display_limit
        if display_limit is not None
        else max(
            float(np.percentile(np.abs(error), 99.5)),
            float(np.finfo(np.float32).eps),
        )
    )
    rows = (
        (reference, f"Ground truth (±{image_limit:.3g} ppm)", "gray", image_limit),
        (
            reconstruction,
            f"{method_label} (±{image_limit:.3g} ppm)",
            "gray",
            image_limit,
        ),
        (
            error,
            f"Difference (reconstruction - truth, ±{error_limit:.3g} ppm)",
            "coolwarm",
            error_limit,
        ),
    )
    figure, axes = plt.subplots(3, 3, figsize=(11, 10), constrained_layout=True)
    figure.suptitle(title)
    for axis, orientation in zip(axes[0], ("Axial", "Coronal", "Sagittal")):
        axis.set_title(orientation)
    for row, (volume, label, cmap, limit) in enumerate(rows):
        for column, select in enumerate(selectors):
            axes[row, column].imshow(
                np.rot90(select(volume)), cmap=cmap, vmin=-limit, vmax=limit
            )
            axes[row, column].axis("off")
            if column == 0:
                axes[row, column].text(
                    -0.06,
                    0.5,
                    label,
                    transform=axes[row, column].transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                )
    return figure


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate_external_cases(
    model: AdaptiveWHTV,
    cases: Iterable[ExternalValidationCase],
    device: str | torch.device,
    *,
    bf16: bool = True,
    output_dir: str | Path | None = None,
    save_volumes: bool = False,
) -> ExternalValidationResult:
    """Evaluate cases, optionally saving NIfTI volumes and WHNet-style figures."""
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if bf16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 was requested but is unsupported")
    destination = Path(output_dir) if output_dir is not None else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
    mode = model.training
    model.eval()
    flat_metrics: dict[str, float] = {}
    details: dict[str, dict[str, Any]] = {}
    figures: dict[str, Any] = {}
    context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if bf16 and device.type == "cuda"
        else nullcontext
    )
    try:
        for case in cases:
            phase = case.phase.to(device, non_blocking=True)
            mask = case.mask.to(device, non_blocking=True)
            dipole = case.dipole.to(device, non_blocking=True)
            chi_reference = case.chi.to(device, non_blocking=True)
            background_reference = (
                None
                if case.background is None
                else case.background.to(device, non_blocking=True)
            )
            if phase.shape != mask.shape or phase.shape != chi_reference.shape:
                raise ValueError(f"{case.name} phase/mask/reference shapes differ")
            _synchronize(device)
            started = time.perf_counter()
            with torch.inference_mode(), context():
                output = model(
                    phase,
                    mask,
                    dipole,
                    gradient_tail=0,
                )
            _synchronize(device)
            duration = time.perf_counter() - started
            if not bool(
                torch.isfinite(output.chi).all()
                and torch.isfinite(output.background).all()
                and torch.isfinite(output.variance).all()
                and torch.isfinite(output.precision).all()
            ):
                raise FloatingPointError(f"Nonfinite {case.name} reconstruction")
            chi_nrmse = masked_nrmse(output.chi, chi_reference, mask)
            flat_metrics[f"external_{case.name}_chi_nrmse"] = chi_nrmse
            case_metrics: dict[str, Any] = {
                "chi_nrmse": chi_nrmse,
                "inference_seconds": duration,
                "solver_iterations": int(model.solver_iterations),
                "factor": case.factor,
                "metadata": case.metadata,
                "parameters": {
                    name: float(value.detach().float().mean())
                    for name, value in output.parameters.as_dict().items()
                },
                "precision": {
                    "mean": float(output.precision[mask.bool()].float().mean()),
                    "minimum": float(output.precision[mask.bool()].float().min()),
                    "maximum": float(output.precision[mask.bool()].float().max()),
                },
            }
            if background_reference is not None:
                background_nrmse = masked_nrmse(
                    output.background, background_reference, mask
                )
                flat_metrics[f"external_{case.name}_background_nrmse"] = (
                    background_nrmse
                )
                case_metrics["background_nrmse"] = background_nrmse

            physical_mask = mask
            arrays = {
                "chi_reference": _crop_tensor(
                    chi_reference * case.factor, case.output_pad
                ),
                "chi_reconstruction": _crop_tensor(
                    output.chi * case.factor, case.output_pad
                ),
                "chi_error": _crop_tensor(
                    (output.chi - chi_reference) * physical_mask * case.factor,
                    case.output_pad,
                ),
                "mask": _crop_tensor(mask, case.output_pad),
                "phase": _crop_tensor(phase * case.factor, case.output_pad),
                "relative_variance": _crop_tensor(output.variance, case.output_pad),
                "fidelity_precision": _crop_tensor(output.precision, case.output_pad),
            }
            figures[f"{case.name}_chi_comparison"] = comparison_figure(
                arrays["chi_reference"],
                arrays["chi_reconstruction"],
                arrays["chi_error"],
                title=f"Adaptive WH-TV {case.name} susceptibility, NRMSE={chi_nrmse:.4f}",
            )
            if background_reference is not None:
                arrays.update(
                    background_reference=_crop_tensor(
                        background_reference * case.factor, case.output_pad
                    ),
                    background_reconstruction=_crop_tensor(
                        output.background * case.factor, case.output_pad
                    ),
                    background_error=_crop_tensor(
                        (output.background - background_reference)
                        * physical_mask
                        * case.factor,
                        case.output_pad,
                    ),
                )
                figures[f"{case.name}_background_comparison"] = comparison_figure(
                    arrays["background_reference"],
                    arrays["background_reconstruction"],
                    arrays["background_error"],
                    title=(
                        f"Adaptive WH-TV {case.name} harmonic background, "
                        f"NRMSE={case_metrics['background_nrmse']:.4f}"
                    ),
                )
            details[case.name] = case_metrics
            if destination is not None:
                case_dir = destination / case.name
                case_dir.mkdir(parents=True, exist_ok=True)
                for name, figure in figures.items():
                    if name.startswith(f"{case.name}_"):
                        figure.savefig(
                            case_dir / f"{name.removeprefix(case.name + '_')}.png",
                            dpi=180,
                        )
                if save_volumes:
                    import nibabel as nib

                    for name, array in arrays.items():
                        nib.save(
                            nib.Nifti1Image(array, np.eye(4)),
                            case_dir / f"{name}.nii.gz",
                        )
                (case_dir / "metrics.json").write_text(
                    json.dumps(case_metrics, indent=2) + "\n"
                )
            del phase, mask, dipole, chi_reference, background_reference, output
    finally:
        model.train(mode)
    if destination is not None:
        (destination / "metrics.json").write_text(
            json.dumps({"metrics": flat_metrics, "datasets": details}, indent=2) + "\n"
        )
    return ExternalValidationResult(flat_metrics, details, figures)


def _latest_checkpoint() -> Path:
    candidates = list(RUNS_DIR.glob("adaptive_wh_tv_*/model_best.pth"))
    if not candidates:
        raise FileNotFoundError("No adaptive WH-TV model_best.pth was found")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--simulation", type=Path)
    parser.add_argument(
        "--datasets", nargs="+", choices=("cosmos", "sim2"), default=("cosmos", "sim2")
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--comet", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--save-volumes", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("threads must be positive")
    checkpoint = args.checkpoint or _latest_checkpoint()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output = args.output_dir or RUNS_DIR / datetime.now(timezone.utc).strftime(
        "adaptive_wh_tv_external_%Y%m%d_%H%M%S_%f"
    )
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    from .adaptive_wh_tv_training import load_adaptive_wh_tv

    model = load_adaptive_wh_tv(checkpoint, device=device)
    cases = load_external_cases(args.datasets, cosmos_simulation=args.simulation)
    result = evaluate_external_cases(
        model,
        cases,
        device,
        bf16=args.bf16,
        output_dir=output,
        save_volumes=args.save_volumes,
    )
    summary = {
        "checkpoint": str(checkpoint.resolve()),
        "output": str(output.resolve()),
        "metrics": result.metrics,
        "datasets": result.details,
    }
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.comet:
        from comet_ml import Experiment

        experiment = Experiment(project_name="wh-net-qsm")
        experiment.add_tag("adaptive-wh-tv")
        experiment.add_tag("external-validation")
        experiment.log_parameters(
            {
                "checkpoint": str(checkpoint.resolve()),
                "datasets": ",".join(args.datasets),
                "bf16": args.bf16,
            }
        )
        experiment.log_metrics(result.metrics)
        for name, figure in result.figures.items():
            experiment.log_figure(figure_name=name, figure=figure)
        experiment.end()
    import matplotlib.pyplot as plt

    for figure in result.figures.values():
        plt.close(figure)
    print(json.dumps(summary, indent=2))
    print(f"Saved external validation to {output}")


if __name__ == "__main__":
    main()

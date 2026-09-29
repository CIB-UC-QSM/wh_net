"""Benchmark WHNet and WH-TV on Graz 1/2 with L-curve model selection."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import tempfile
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import zoom
import torch
from torch import Tensor

from whnet.model import ADMMUnrolledNet, ProximalNetwork
from whnet.paths import CHECKPOINTS_DIR, EVALUATION_DATA_DIR, JULIA_ENV_DIR, RUNS_DIR
from whnet.physics import continuous_dipole_kernel
from whnet.wh_tv import WHTVResult, wh_tv

SPATIAL = (-3, -2, -1)
MIN_BETA = 1e6
DEFAULT_CHECKPOINT = CHECKPOINTS_DIR / "checkpoints_scratch7" / "model_best.pth"
DEFAULT_COSMOS_RUN = RUNS_DIR / "wh_tv_cosmos_20260914_000637_662208"
DATASETS = {
    "graz1": (EVALUATION_DATA_DIR / "data_blend_1_graz.mat", 0.325),
    "graz2": (EVALUATION_DATA_DIR / "data_blending2.mat", 0.300),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cosmos-run", type=Path, default=DEFAULT_COSMOS_RUN)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--whnet-iterations", type=int, default=48)
    parser.add_argument("--wh-tv-iterations", type=int, default=5000)
    parser.add_argument("--lcurve-iterations", type=int, default=2000)
    parser.add_argument("--lcurve-points", type=int, default=11)
    parser.add_argument("--lcurve-log-min", type=float, default=-2.0)
    parser.add_argument("--lcurve-log-max", type=float, default=2.0)
    parser.add_argument("--lcurve-mode", choices=("joint", "tv-only"), default="joint")
    parser.add_argument(
        "--fixed-parameter-run",
        type=Path,
        help="Prior Graz run providing fixed beta and center alpha1",
    )
    parser.add_argument("--pad", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--julia", default="/home/mathias/.juliaup/bin/julia")
    parser.add_argument(
        "--cache-dir", type=Path, default=RUNS_DIR.parent / "cache" / "graz"
    )
    parser.add_argument("--refresh-unwrapping", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if (
        min(
            args.whnet_iterations,
            args.wh_tv_iterations,
            args.lcurve_iterations,
            args.lcurve_points,
            args.threads,
        )
        < 1
    ):
        parser.error("iteration counts, L-curve points, and threads must be positive")
    if args.lcurve_points < 3:
        parser.error("L-curve requires at least three points")
    if args.lcurve_log_min >= args.lcurve_log_max or args.pad < 0:
        parser.error("invalid L-curve range or padding")
    return args


def unwrap_echoes(
    wrapped: np.ndarray,
    mask: np.ndarray,
    cache_path: Path,
    julia: str,
    refresh: bool,
) -> np.ndarray:
    if cache_path.is_file() and not refresh:
        unwrapped = np.load(cache_path)
        if unwrapped.shape == wrapped.shape and np.isfinite(unwrapped).all():
            print(f"Using cached unwrapped phase: {cache_path}", flush=True)
            return np.asarray(unwrapped, dtype=np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        input_path = directory / "wrapped.nii"
        mask_path = directory / "mask.nii"
        output_path = directory / "unwrapped.nii"
        script_path = directory / "unwrap.jl"
        nib.save(nib.Nifti1Image(wrapped.astype(np.float32), np.eye(4)), input_path)
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), np.eye(4)), mask_path)
        script_path.write_text(
            """using NIfTI
using ROMEO
input = niread(ARGS[1])
wrapped = input.raw
mask = niread(ARGS[2]).raw .> 0
unwrapped = similar(wrapped)
for echo in axes(wrapped, 4)
    unwrapped[:, :, :, echo] = unwrap(wrapped[:, :, :, echo]; mask=mask)
end
niwrite(ARGS[3], NIVolume(input.header, input.extensions, unwrapped))
"""
        )
        subprocess.run(
            [
                julia,
                f"--project={JULIA_ENV_DIR}",
                str(script_path),
                str(input_path),
                str(mask_path),
                str(output_path),
            ],
            check=True,
        )
        unwrapped = np.asarray(nib.load(output_path).dataobj, dtype=np.float32)
    if unwrapped.shape != wrapped.shape or not np.isfinite(unwrapped).all():
        raise ValueError("ROMEO returned an invalid unwrapped phase")
    np.save(cache_path, unwrapped)
    return unwrapped


def pad_cube(array: np.ndarray, size: int) -> np.ndarray:
    padding = []
    for length in array.shape:
        total = size - length
        padding.append((total // 2, total - total // 2))
    return np.pad(array, padding)


def prepare_dataset(
    name: str,
    path: Path,
    factor: float,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Tensor], dict]:
    source = loadmat(path)
    required = {"Phase", "Magn", "mask", "TE", "voxel_size", "phs_scale"}
    missing = required - source.keys()
    if missing:
        raise KeyError(f"{path} lacks {sorted(missing)}")
    wrapped = np.asarray(source["Phase"], dtype=np.float32)
    magnitude = np.asarray(source["Magn"], dtype=np.float32)
    mask = np.asarray(source["mask"], dtype=bool)
    echo_times = np.asarray(source["TE"], dtype=np.float32).ravel()
    if wrapped.shape != magnitude.shape or wrapped.shape[-1] != echo_times.size:
        raise ValueError(f"Invalid multi-echo arrays in {path}")
    cached = args.cache_dir / f"{name}_romeo_unwrapped.npy"
    if "phase2" in source and not args.refresh_unwrapping:
        unwrapped = np.asarray(source["phase2"], dtype=np.float32)
        print(f"Using embedded unwrapped phase for {name}", flush=True)
    else:
        unwrapped = unwrap_echoes(
            wrapped, mask, cached, args.julia, args.refresh_unwrapping
        )
    magnitude /= max(float(magnitude.max()), np.finfo(np.float32).eps)
    weights = np.square(magnitude)
    numerator = np.sum(weights * unwrapped * echo_times, axis=-1)
    denominator = np.sum(weights * np.square(echo_times), axis=-1)
    angular_frequency = numerator / np.maximum(denominator, 1e-12)
    phase_ppm = angular_frequency / float(np.asarray(source["phs_scale"]).item())
    confidence = np.sqrt(denominator)
    confidence /= max(float(confidence.max()), np.finfo(np.float32).eps)

    voxel_size = np.asarray(source["voxel_size"], dtype=float).ravel()
    phase_ppm = zoom(phase_ppm, voxel_size, order=3, mode="nearest")
    confidence = zoom(confidence, voxel_size, order=1, mode="nearest")
    mask = zoom(mask.astype(np.uint8), voxel_size, order=0, mode="nearest").astype(bool)
    coordinates = np.argwhere(mask)
    lower, upper = coordinates.min(0), coordinates.max(0) + 1
    slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper, strict=True))
    phase_ppm, confidence, mask = phase_ppm[slices], confidence[slices], mask[slices]
    size = max(mask.shape) + 2 * args.pad
    phase_ppm = pad_cube(phase_ppm * mask, size)
    confidence = pad_cube(confidence * mask, size)
    mask = pad_cube(mask, size).astype(np.float32)
    phase = torch.from_numpy(phase_ppm.astype(np.float32))[None, None] / factor
    mask_tensor = torch.from_numpy(mask)[None, None]
    dipole = torch.from_numpy(continuous_dipole_kernel((size,) * 3)).float()[None, None]
    data = {
        "phase": phase.to(device),
        "mask": mask_tensor.to(device),
        "weight": mask_tensor.to(device),
        "dipole": dipole.to(device),
    }
    metadata = {
        "source": str(path.resolve()),
        "original_shape": list(wrapped.shape[:3]),
        "resampled_cropped_shape": list(
            slices[i].stop - slices[i].start for i in range(3)
        ),
        "padded_shape": [size] * 3,
        "original_voxel_size_mm": voxel_size.tolist(),
        "target_voxel_size_mm": [1.0, 1.0, 1.0],
        "echo_times_seconds": echo_times.tolist(),
        "phase_conversion": "magnitude-squared weighted through-origin TE slope / phs_scale",
        "phs_scale": float(np.asarray(source["phs_scale"]).item()),
        "solver_factor": factor,
        "pad": args.pad,
    }
    return data, metadata


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def forward_field(chi: Tensor, dipole: Tensor) -> Tensor:
    return torch.fft.ifftn(dipole * torch.fft.fftn(chi, dim=SPATIAL), dim=SPATIAL).real


def positive_laplacian(volume: Tensor) -> Tensor:
    return sum(
        2 * volume - torch.roll(volume, -1, axis) - torch.roll(volume, 1, axis)
        for axis in SPATIAL
    )


def solution_metrics(
    chi: Tensor,
    background: Tensor,
    data: dict[str, Tensor],
    seconds: float,
) -> dict[str, float]:
    selected = data["mask"].bool()
    residual = (forward_field(chi, data["dipole"]) + background - data["phase"])[
        selected
    ]
    phase = data["phase"][selected]
    gradients = [chi - torch.roll(chi, -1, axis) for axis in SPATIAL]
    laplacian = positive_laplacian(background)[selected]
    return {
        "seconds": seconds,
        "normalized_data_residual": float(
            torch.linalg.vector_norm(residual)
            / torch.linalg.vector_norm(phase).clamp_min(1e-12)
        ),
        "data_residual_rms": float(residual.square().mean().sqrt()),
        "chi_tv_mean": float(
            sum(gradient.abs().sum() for gradient in gradients) / selected.sum()
        ),
        "background_laplacian_rms": float(laplacian.square().mean().sqrt()),
        "chi_rms": float(chi[selected].square().mean().sqrt()),
        "background_rms": float(background[selected].square().mean().sqrt()),
    }


def regularizer_measure(
    result: WHTVResult, data: dict[str, Tensor], baseline: dict[str, float]
) -> float:
    selected = data["mask"].bool()
    gradients = [result.chi - torch.roll(result.chi, -1, axis) for axis in SPATIAL]
    tv = sum(gradient.abs().sum() for gradient in gradients) / selected.sum()
    laplacian = positive_laplacian(result.background)[selected]
    harmonic = 0.5 * laplacian.square().mean()
    return float(baseline["alpha1"] * tv + baseline["beta"] * harmonic)


def timed_wh_tv(
    data: dict[str, Tensor],
    parameters: dict[str, float],
    iterations: int,
    device: torch.device,
) -> tuple[WHTVResult, float]:
    synchronize(device)
    started = time.perf_counter()
    result = wh_tv(
        data["phase"],
        data["dipole"],
        data["mask"],
        weight=data["weight"],
        max_iterations=iterations,
        tol_update=None,
        check_every=iterations,
        **parameters,
    )
    synchronize(device)
    return result, time.perf_counter() - started


def lcurve_search(
    data: dict[str, Tensor],
    baseline: dict[str, float],
    args: argparse.Namespace,
    device: torch.device,
    destination: Path,
) -> tuple[dict[str, float], list[dict]]:
    log_scales = np.linspace(
        args.lcurve_log_min, args.lcurve_log_max, args.lcurve_points
    )
    trials = []
    for index, log_scale in enumerate(log_scales, 1):
        scale = float(10**log_scale)
        parameters = dict(baseline)
        parameters["alpha1"] = baseline["alpha1"] * scale
        if args.lcurve_mode == "joint":
            parameters["beta"] = max(MIN_BETA, baseline["beta"] * scale)
        result, seconds = timed_wh_tv(data, parameters, args.lcurve_iterations, device)
        metrics = solution_metrics(result.chi, result.background, data, seconds)
        trial = {
            "index": index,
            "log10_scale": float(log_scale),
            "scale": scale,
            **parameters,
            **metrics,
            "regularizer_measure": (
                metrics["chi_tv_mean"]
                if args.lcurve_mode == "tv-only"
                else regularizer_measure(result, data, baseline)
            ),
        }
        trials.append(trial)
        print(
            f"L-curve {index}/{len(log_scales)}: scale={scale:.4g}, "
            f"residual={metrics['data_residual_rms']:.5g}, "
            f"regularizer={trial['regularizer_measure']:.5g}",
            flush=True,
        )
    x = np.log10(np.maximum([trial["data_residual_rms"] for trial in trials], 1e-30))
    y = np.log10(np.maximum([trial["regularizer_measure"] for trial in trials], 1e-30))
    x_normalized = (x - x.min()) / max(float(np.ptp(x)), 1e-30)
    y_normalized = (y - y.min()) / max(float(np.ptp(y)), 1e-30)
    start = np.array([x_normalized[0], y_normalized[0]])
    chord = np.array(
        [x_normalized[-1] - x_normalized[0], y_normalized[-1] - y_normalized[0]]
    )
    chord_norm = max(float(np.linalg.norm(chord)), 1e-30)
    offsets = np.column_stack((x_normalized, y_normalized)) - start
    corner_scores = np.abs(chord[0] * offsets[:, 1] - chord[1] * offsets[:, 0])
    corner_scores /= chord_norm
    corner_scores[[0, -1]] = np.nan
    selected = int(np.nanargmax(corner_scores))
    for trial, value in zip(trials, corner_scores, strict=True):
        trial["corner_score"] = None if not np.isfinite(value) else float(value)
        trial["selected"] = trial["index"] == selected + 1
    figure, axis = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    axis.plot(10**x, 10**y, "o-", color="#333333", linewidth=1.2, markersize=4)
    axis.scatter(
        10 ** x[selected],
        10 ** y[selected],
        s=75,
        color="#D55E00",
        zorder=3,
        label="Selected corner",
    )
    for trial, x_value, y_value in zip(trials, 10**x, 10**y, strict=True):
        axis.annotate(
            f"{trial['scale']:.2g}×",
            (x_value, y_value),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Data residual RMS")
    axis.set_ylabel(
        "χ TV mean"
        if args.lcurve_mode == "tv-only"
        else "Baseline-weighted regularizer"
    )
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(destination, dpi=240)
    plt.close(figure)
    chosen = dict(baseline)
    chosen["alpha1"] = trials[selected]["alpha1"]
    chosen["beta"] = trials[selected]["beta"]
    return chosen, trials


def run_whnet(
    data: dict[str, Tensor], args: argparse.Namespace, device: torch.device
) -> tuple[Tensor, Tensor, float]:
    model = ADMMUnrolledNet(ProximalNetwork(), ProximalNetwork(), args.whnet_iterations)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    model.eval().requires_grad_(False).to(device)
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last_3d)
    synchronize(device)
    started = time.perf_counter()
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ),
    ):
        chi, background = model(
            data["phase"], data["mask"], data["dipole"], data["weight"]
        )
    synchronize(device)
    return chi * data["mask"], background * data["mask"], time.perf_counter() - started


def crop_array(volume: Tensor, mask: Tensor) -> np.ndarray:
    array = volume.detach().float().cpu().squeeze().numpy()
    support = mask.detach().cpu().squeeze().bool().numpy()
    coordinates = np.argwhere(support)
    lower, upper = coordinates.min(0), coordinates.max(0) + 1
    slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper, strict=True))
    return np.asarray(array[slices], dtype=np.float32)


def save_volume(path: Path, volume: Tensor, mask: Tensor, factor: float) -> None:
    nib.save(nib.Nifti1Image(crop_array(volume * factor, mask), np.eye(4)), path)


def comparison_figure(
    solutions: list[tuple[str, Tensor, Tensor]],
    mask: Tensor,
    factor: float,
    destination: Path,
) -> None:
    arrays = [
        (name, crop_array(chi * factor, mask), crop_array(phi * factor, mask))
        for name, chi, phi in solutions
    ]
    center = tuple(size // 2 for size in arrays[0][1].shape)
    views = (
        ("Axial", lambda x: x[center[0]]),
        ("Coronal", lambda x: x[:, center[1]]),
        ("Sagittal", lambda x: x[:, :, center[2]]),
    )
    for quantity, position, label in (("chi", 1, "χ (ppm)"), ("phi", 2, "φₕ (ppm)")):
        limit = max(
            float(
                np.quantile(
                    np.abs(np.concatenate([item[position].ravel() for item in arrays])),
                    0.995,
                )
            ),
            1e-12,
        )
        figure, axes = plt.subplots(
            3, len(arrays), figsize=(9, 8), constrained_layout=True, squeeze=False
        )
        image = None
        for column, item in enumerate(arrays):
            for row, (view, selector) in enumerate(views):
                image = axes[row, column].imshow(
                    np.rot90(selector(item[position])),
                    cmap="gray",
                    vmin=-limit,
                    vmax=limit,
                )
                axes[row, column].axis("off")
                if row == 0:
                    axes[row, column].set_title(item[0], fontsize=9)
                if column == 0:
                    axes[row, column].text(
                        -0.03,
                        0.5,
                        view,
                        transform=axes[row, column].transAxes,
                        rotation=90,
                        va="center",
                        ha="right",
                    )
        figure.colorbar(image, ax=axes.ravel().tolist(), label=label, shrink=0.85)
        figure.savefig(
            destination.with_name(f"{destination.stem}_{quantity}.png"), dpi=240
        )
        plt.close(figure)


def benchmark_dataset(
    name: str,
    path: Path,
    factor: float,
    baseline: dict[str, float],
    lcurve_center: dict[str, float],
    args: argparse.Namespace,
    device: torch.device,
    output: Path,
) -> dict:
    destination = output / name
    destination.mkdir()
    data, preprocessing = prepare_dataset(name, path, factor, args, device)
    print(f"{name}: prepared {tuple(data['phase'].shape)}", flush=True)
    whnet_chi, whnet_phi, whnet_seconds = run_whnet(data, args, device)
    baseline_result, baseline_seconds = timed_wh_tv(
        data, baseline, args.wh_tv_iterations, device
    )
    optimized, trials = lcurve_search(
        data, lcurve_center, args, device, destination / "lcurve.png"
    )
    optimized_result, optimized_seconds = timed_wh_tv(
        data, optimized, args.wh_tv_iterations, device
    )
    solutions = [
        ("WHNet", whnet_chi, whnet_phi),
        ("WH-TV · COSMOS", baseline_result.chi, baseline_result.background),
        ("WH-TV · L-curve", optimized_result.chi, optimized_result.background),
    ]
    comparison_figure(
        solutions, data["mask"], factor, destination / "reconstruction.png"
    )
    for label, chi, background in solutions:
        stem = label.lower().replace(" · ", "_").replace("-", "_")
        save_volume(destination / f"{stem}_chi.nii.gz", chi, data["mask"], factor)
        save_volume(
            destination / f"{stem}_background.nii.gz", background, data["mask"], factor
        )
    records = {
        "whnet": {
            "iterations": args.whnet_iterations,
            **solution_metrics(whnet_chi, whnet_phi, data, whnet_seconds),
        },
        "wh_tv_cosmos": {
            "iterations": args.wh_tv_iterations,
            "parameters": baseline,
            **solution_metrics(
                baseline_result.chi, baseline_result.background, data, baseline_seconds
            ),
        },
        "wh_tv_lcurve": {
            "iterations": args.wh_tv_iterations,
            "parameters": optimized,
            **solution_metrics(
                optimized_result.chi,
                optimized_result.background,
                data,
                optimized_seconds,
            ),
        },
    }
    with (destination / "lcurve_trials.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trials[0]))
        writer.writeheader()
        writer.writerows(trials)
    result = {"preprocessing": preprocessing, "methods": records}
    (destination / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu")
    best = json.loads((args.cosmos_run / "best.json").read_text())
    baseline = {
        name: float(best[name]) for name in ("alpha1", "beta", "mu1", "mu2", "muh")
    }
    if baseline["beta"] < MIN_BETA:
        raise ValueError("COSMOS baseline beta is below 1e6")
    lcurve_centers = {name: dict(baseline) for name in DATASETS}
    if args.lcurve_mode == "tv-only":
        if args.fixed_parameter_run is None:
            raise ValueError("--fixed-parameter-run is required for tv-only mode")
        previous = json.loads((args.fixed_parameter_run / "summary.json").read_text())
        for name in DATASETS:
            parameters = previous["datasets"][name]["methods"]["wh_tv_lcurve"][
                "parameters"
            ]
            lcurve_centers[name] = {
                key: float(parameters[key])
                for key in ("alpha1", "beta", "mu1", "mu2", "muh")
            }
            if lcurve_centers[name]["beta"] < MIN_BETA:
                raise ValueError(f"Fixed beta for {name} is below 1e6")
    output = args.output_dir or RUNS_DIR / datetime.now(timezone.utc).strftime(
        "graz_whnet_wh_tv_%Y%m%d_%H%M%S_%f"
    )
    output.mkdir(parents=True, exist_ok=False)
    config = {
        "checkpoint": str(args.checkpoint.resolve()),
        "cosmos_run": str(args.cosmos_run.resolve()),
        "cosmos_parameters": baseline,
        "whnet_iterations": args.whnet_iterations,
        "wh_tv_iterations": args.wh_tv_iterations,
        "lcurve_iterations": args.lcurve_iterations,
        "lcurve_log10_scale_range": [args.lcurve_log_min, args.lcurve_log_max],
        "lcurve_points": args.lcurve_points,
        "lcurve_mode": args.lcurve_mode,
        "lcurve_parameterization": (
            "alpha1 scaled; beta, mu1, mu2, muh fixed"
            if args.lcurve_mode == "tv-only"
            else "alpha1 and beta share scale; mu1, mu2, muh fixed"
        ),
        "lcurve_centers": lcurve_centers,
        "fixed_parameter_run": (
            None
            if args.fixed_parameter_run is None
            else str(args.fixed_parameter_run.resolve())
        ),
        "lcurve_corner": (
            "maximum normalized distance from the endpoint chord in "
            "log residual/log regularizer space"
        ),
        "ground_truth_available": False,
        "device": str(device),
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    results = {}
    for name, (path, factor) in DATASETS.items():
        results[name] = benchmark_dataset(
            name,
            path,
            factor,
            baseline,
            lcurve_centers[name],
            args,
            device,
            output,
        )
    summary = {"config": config, "datasets": results}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    summary_rows = []
    for dataset, result in results.items():
        for method, metrics in result["methods"].items():
            summary_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    **{
                        key: value
                        for key, value in metrics.items()
                        if key != "parameters"
                    },
                }
            )
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved Graz benchmark to {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()

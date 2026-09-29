"""Compare WHNet and optimized WH-TV convergence on the COSMOS simulation."""

import argparse
from contextlib import nullcontext
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from torch import Tensor

from whnet.model import ADMMUnrolledNet, ProximalNetwork
from whnet.paths import CHECKPOINTS_DIR, RUNS_DIR
from whnet.wh_tv import wh_tv

SPATIAL = (-3, -2, -1)
PARAMETERS = ("alpha1", "beta", "mu1", "mu2", "muh")
DEFAULT_CHECKPOINT = CHECKPOINTS_DIR / "checkpoints_scratch7" / "model_best.pth"


def find_wh_tv_run() -> Path:
    candidates = []
    for directory in RUNS_DIR.glob("wh_tv_cosmos_*"):
        required = [
            directory / name
            for name in ("config.json", "best.json", "simulation.pt", "status.json")
        ]
        if not all(path.is_file() for path in required):
            continue
        config = json.loads(required[0].read_text())
        status = json.loads(required[3].read_text())
        if (
            config.get("snr") == 100
            and config.get("bg") == 10
            and status.get("status") == "complete"
        ):
            candidates.append(directory)
    if not candidates:
        raise FileNotFoundError("No completed SNR=100/BG=10 WH-TV run found")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--wh-tv-run", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--whnet-iterations", type=int, default=55)
    parser.add_argument(
        "--wh-tv-iterations",
        type=int,
        help="Default: iteration count used by the selected WH-TV optimization",
    )
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--channels-last", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--selection",
        choices=("matched", "final"),
        default="matched",
        help="matched: best WHNet chi and error-matched WH-TV; final: last iterate of both",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.whnet_iterations < 1 or (
        args.wh_tv_iterations is not None and args.wh_tv_iterations < 1
    ):
        parser.error("iteration counts must be positive")
    if args.threads < 1:
        parser.error("threads must be positive")
    return args


def nrmse(prediction: Tensor, reference: Tensor, mask: Tensor) -> Tensor:
    selected = mask.bool()
    return torch.linalg.vector_norm(
        (prediction - reference)[selected]
    ) / torch.linalg.vector_norm(reference[selected]).clamp_min(1e-12)


def update_percent(current: Tensor, previous: Tensor, mask: Tensor) -> Tensor:
    selected = mask.bool()
    return (
        100
        * torch.linalg.vector_norm((current - previous)[selected])
        / torch.linalg.vector_norm(current[selected]).clamp_min(1e-12)
    )


def positive_laplacian(volume: Tensor) -> Tensor:
    return sum(
        2 * volume - torch.roll(volume, -1, axis) - torch.roll(volume, 1, axis)
        for axis in SPATIAL
    )


def interior_mask(mask: Tensor) -> Tensor:
    inside = mask.bool()
    for axis in SPATIAL:
        inside = (
            inside
            & torch.roll(mask.bool(), -1, axis)
            & torch.roll(mask.bool(), 1, axis)
        )
    return inside


def trajectory_row(
    method: str,
    iteration: int,
    chi: Tensor,
    phi: Tensor,
    previous_chi: Tensor,
    previous_phi: Tensor,
    chi_reference: Tensor,
    phi_reference: Tensor,
    mask: Tensor,
    elapsed_seconds: float,
) -> tuple[str, int, Tensor, Tensor, Tensor, Tensor, float]:
    return (
        method,
        iteration,
        update_percent(chi, previous_chi, mask),
        update_percent(phi, previous_phi, mask),
        nrmse(chi, chi_reference, mask),
        nrmse(phi, phi_reference, mask),
        elapsed_seconds,
    )


def rows_to_cpu(rows: list[tuple]) -> list[dict[str, float | int | str]]:
    numeric = (
        torch.stack([torch.stack(list(row[2:6])) for row in rows]).float().cpu().numpy()
    )
    return [
        {
            "method": row[0],
            "iteration": row[1],
            "chi_update_percent": float(values[0]),
            "phi_update_percent": float(values[1]),
            "chi_nrmse": float(values[2]),
            "phi_nrmse": float(values[3]),
            "elapsed_seconds": float(row[6]),
        }
        for row, values in zip(rows, numeric, strict=True)
    ]


def crop(volume: Tensor, pad: int) -> np.ndarray:
    array = volume.detach().float().cpu().squeeze().numpy()
    if pad:
        array = array[pad:-pad, pad:-pad, pad:-pad]
    return np.asarray(array, dtype=np.float32)


def save_nifti(path: Path, volume: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(volume, np.eye(4)), path)


def convergence_figure(rows: list[dict], destination: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    fields = (
        ("chi_update_percent", "χ update (%)"),
        ("phi_update_percent", "φₕ update (%)"),
        ("chi_nrmse", "χ NRMSE"),
        ("phi_nrmse", "φₕ NRMSE"),
    )
    for method in ("WHNet", "WH-TV"):
        selected = [row for row in rows if row["method"] == method]
        x = [row["iteration"] for row in selected]
        for axis, (field, label) in zip(axes.flat, fields, strict=True):
            axis.plot(x, [max(row[field], 1e-12) for row in selected], label=method)
            axis.set_xlabel("Iteration")
            axis.set_ylabel(label)
            axis.set_yscale("log")
            if max(x) > 100:
                axis.set_xscale("log")
            axis.grid(alpha=0.25)
    for axis in axes.flat:
        axis.legend()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def time_convergence_figure(rows: list[dict], destination: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    fields = (
        ("chi_update_percent", "χ update (%)"),
        ("phi_update_percent", "φₕ update (%)"),
        ("chi_nrmse", "χ NRMSE"),
        ("phi_nrmse", "φₕ NRMSE"),
    )
    for method in ("WHNet", "WH-TV"):
        selected = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: row["iteration"],
        )
        if not selected or [row["iteration"] for row in selected] != list(
            range(1, len(selected) + 1)
        ):
            raise ValueError(f"Incomplete iteration trajectory for {method}")
        elapsed = [0.0, *[row["elapsed_seconds"] for row in selected]]
        if not np.isfinite(elapsed).all() or np.any(np.diff(elapsed) <= 0):
            raise ValueError(f"Invalid cumulative elapsed times for {method}")
        runtime_label = f"{method}: {len(selected):,} iterations, {elapsed[-1]:.2f} s"
        for axis, (field, label) in zip(axes.flat, fields, strict=True):
            initial_value = 1.0 if field.endswith("nrmse") else 0.0
            values = [initial_value, *[row[field] for row in selected]]
            (line,) = axis.plot(elapsed, values, label=runtime_label)
            axis.scatter(
                [elapsed[0], elapsed[-1]],
                [values[0], values[-1]],
                color=line.get_color(),
                s=25,
                zorder=3,
            )
            axis.set_xlabel("Wall-clock time (s)")
            axis.set_ylabel(label)
            if field.endswith("nrmse"):
                axis.set_yscale("log")
            else:
                axis.set_yscale("symlog", linthresh=0.1)
            axis.grid(alpha=0.25)
    maximum_time = max(row["elapsed_seconds"] for row in rows)
    for axis in axes.flat:
        axis.set_xlim(0.0, maximum_time * 1.03)
        axis.legend()
    figure.suptitle(
        "Selected reconstruction trajectories (zero initialization at t = 0)"
    )
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def comparison_figure(
    volumes: list[tuple[str, np.ndarray, float | None]],
    destination: Path,
    title: str,
    colorbar_label: str,
    percentile: float = 99.5,
    nonnegative: bool = False,
) -> None:
    center = tuple(size // 2 for size in volumes[0][1].shape)
    views = (
        ("Axial", lambda x: x[center[0]]),
        ("Coronal", lambda x: x[:, center[1]]),
        ("Sagittal", lambda x: x[:, :, center[2]]),
    )
    limit = max(
        float(
            np.percentile(
                np.abs(
                    np.concatenate([volume.ravel() for _, volume, _rmse in volumes])
                ),
                percentile,
            )
        ),
        np.finfo(np.float32).eps,
    )
    figure, axes = plt.subplots(
        len(views),
        len(volumes),
        figsize=(4 * len(volumes), 10),
        constrained_layout=True,
        squeeze=False,
    )
    figure.suptitle(title)
    image = None
    for column, (label, volume, nrmse_value) in enumerate(volumes):
        for row, (view, select) in enumerate(views):
            axis = axes[row, column]
            image_slice = np.rot90(select(volume))
            if row == 2:
                image_slice = np.rot90(image_slice, 2)
            image = axis.imshow(
                image_slice,
                cmap="gray",
                vmin=0.0 if nonnegative else -limit,
                vmax=limit,
            )
            axis.axis("off")
            if row == 0:
                axis.set_title(label)
            if column == 0:
                axis.text(
                    -0.04,
                    0.5,
                    view,
                    transform=axis.transAxes,
                    rotation=90,
                    ha="right",
                    va="center",
                )
            if nrmse_value is not None and row == 0:
                axis.text(
                    0.03,
                    0.96,
                    f"NRMSE = {nrmse_value:.4g}",
                    transform=axis.transAxes,
                    color="white",
                    ha="left",
                    va="top",
                    bbox={"facecolor": "black", "alpha": 0.65, "pad": 2},
                )
    if image is not None:
        figure.colorbar(
            image,
            ax=axes.ravel().tolist(),
            label=colorbar_label,
            shrink=0.85,
            pad=0.02,
        )
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def masked_array_nrmse(
    reconstruction: np.ndarray, reference: np.ndarray, mask: np.ndarray
) -> float:
    selected = mask.astype(bool)
    if not np.any(selected):
        raise ValueError("NRMSE mask is empty")
    difference_norm = np.linalg.norm(reconstruction[selected] - reference[selected])
    reference_norm = np.linalg.norm(reference[selected])
    return float(difference_norm / max(reference_norm, np.finfo(np.float32).eps))


def laplacian_metrics(laplacian: Tensor, phi: Tensor, mask: Tensor) -> dict[str, float]:
    selected = mask.bool()
    values, phi_values = laplacian[selected], phi[selected]
    return {
        "l2_norm": float(torch.linalg.vector_norm(values)),
        "rms": float(values.square().mean().sqrt()),
        "relative_l2_to_phi": float(
            torch.linalg.vector_norm(values)
            / torch.linalg.vector_norm(phi_values).clamp_min(1e-12)
        ),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu")
    if args.bf16 and device.type != "cuda":
        raise ValueError("Use --no-bf16 on CPU")
    wh_tv_run = args.wh_tv_run or find_wh_tv_run()
    config = json.loads((wh_tv_run / "config.json").read_text())
    best = json.loads((wh_tv_run / "best.json").read_text())
    data = torch.load(
        wh_tv_run / "simulation.pt", map_location=device, weights_only=True
    )
    if config.get("snr") != 100 or config.get("bg") != 10:
        raise ValueError("WH-TV run must use the COSMOS SNR=100/BG=10 simulation")
    wh_tv_iterations = args.wh_tv_iterations or int(config["iterations"])
    parameters = {name: float(best[name]) for name in PARAMETERS}
    phase, mask, dipole = data["phase"], data["mask"], data["dipole"]
    chi_reference, phi_reference = data["chi"], data["background"]

    model = ADMMUnrolledNet(ProximalNetwork(), ProximalNetwork(), args.whnet_iterations)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    model.eval().requires_grad_(False).to(device)
    if args.channels_last:
        model.to(memory_format=torch.channels_last_3d)
        phase = phase.contiguous(memory_format=torch.channels_last_3d)
        mask = mask.contiguous(memory_format=torch.channels_last_3d)
    autocast = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if args.bf16
        else nullcontext
    )

    # Stream scalar diagnostics and retain only previous/best paired volumes.
    # Synchronize before timestamps, then subtract measured diagnostic callback time.
    zero = torch.zeros_like(phase)
    rows = []
    whnet_iteration_seconds: list[float] = []
    previous_whnet = [zero, zero]
    best_whnet_state = [zero, zero]
    minimum_whnet_nrmse = float("inf")
    minimum_whnet_position = 0
    diagnostic_seconds = 0.0

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    synchronize()
    whnet_started = time.perf_counter()

    def track_whnet(iteration: int, chi: Tensor, phi: Tensor) -> None:
        nonlocal diagnostic_seconds, minimum_whnet_nrmse, minimum_whnet_position
        synchronize()
        callback_started = time.perf_counter()
        elapsed = callback_started - whnet_started - diagnostic_seconds
        row = trajectory_row(
            "WHNet",
            iteration,
            chi,
            phi,
            previous_whnet[0],
            previous_whnet[1],
            chi_reference,
            phi_reference,
            mask,
            elapsed,
        )
        values = torch.stack(row[2:6]).float().cpu()
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError(
                f"Nonfinite WHNet metrics at iteration {iteration}"
            )
        rows.append(row)
        whnet_iteration_seconds.append(elapsed)
        score = float(values[2])
        if score < minimum_whnet_nrmse:
            minimum_whnet_nrmse = score
            minimum_whnet_position = iteration - 1
            if args.selection == "matched":
                best_whnet_state[0], best_whnet_state[1] = chi.clone(), phi.clone()
        previous_whnet[0], previous_whnet[1] = chi, phi
        if iteration % 100 == 0 or iteration == args.whnet_iterations:
            print(
                f"WHNet {iteration}/{args.whnet_iterations}: chi NRMSE={score:.6g}, "
                f"solver time={elapsed:.2f}s",
                flush=True,
            )
        synchronize()
        diagnostic_seconds += time.perf_counter() - callback_started

    with torch.inference_mode(), autocast():
        final_whnet_chi, final_whnet_phi = model(
            phase,
            mask,
            dipole,
            mask,
            return_iterates=False,
            iteration_callback=track_whnet,
        )
    synchronize()
    whnet_wall_seconds = time.perf_counter() - whnet_started
    whnet_seconds = whnet_wall_seconds - diagnostic_seconds
    whnet_iteration_seconds[-1] = whnet_seconds
    rows[-1] = rows[-1][:-1] + (whnet_seconds,)
    best_whnet_position = (
        args.whnet_iterations - 1
        if args.selection == "final"
        else minimum_whnet_position
    )
    best_whnet_iteration = best_whnet_position + 1
    whnet_chi, whnet_phi = (
        (final_whnet_chi, final_whnet_phi)
        if args.selection == "final"
        else best_whnet_state
    )

    wh_tv_rows = []
    previous = [zero, zero]

    def track_wh_tv(iteration: int, chi: Tensor, phi: Tensor, _update: float) -> None:
        wh_tv_rows.append(
            trajectory_row(
                "WH-TV",
                iteration,
                chi,
                phi,
                previous[0],
                previous[1],
                chi_reference,
                phi_reference,
                mask,
                0.0,
            )
        )
        previous[0], previous[1] = chi.clone(), phi.clone()

    wh_tv_result = wh_tv(
        phase,
        dipole,
        mask,
        weight=mask,
        max_iterations=wh_tv_iterations,
        tol_update=None,
        check_every=1,
        callback=track_wh_tv,
        **parameters,
    )

    wh_tv_iteration_seconds: list[float] = []
    wh_tv_events: list[torch.cuda.Event] = []
    wh_tv_start_event = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        wh_tv_start_event = torch.cuda.Event(enable_timing=True)
        wh_tv_start_event.record()
    else:
        wh_tv_started = time.perf_counter()

    def track_wh_tv_time(
        _iteration: int, _chi: Tensor, _phi: Tensor, _update: float
    ) -> None:
        if device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            wh_tv_events.append(event)
        else:
            wh_tv_iteration_seconds.append(time.perf_counter() - wh_tv_started)

    wh_tv(
        phase,
        dipole,
        mask,
        weight=mask,
        max_iterations=wh_tv_iterations,
        tol_update=None,
        check_every=1,
        callback=track_wh_tv_time,
        **parameters,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        if wh_tv_start_event is None:
            raise RuntimeError("Missing WH-TV CUDA timing event")
        wh_tv_iteration_seconds = [
            wh_tv_start_event.elapsed_time(event) / 1000.0 for event in wh_tv_events
        ]
    wh_tv_seconds = wh_tv_iteration_seconds[-1]
    if len(wh_tv_rows) != len(wh_tv_iteration_seconds):
        raise RuntimeError("WH-TV trajectory and timing lengths differ")
    wh_tv_rows = [
        row[:-1] + (elapsed,)
        for row, elapsed in zip(wh_tv_rows, wh_tv_iteration_seconds, strict=True)
    ]
    rows.extend(wh_tv_rows)
    cpu_rows = rows_to_cpu(rows)

    target_chi_nrmse = float(rows[best_whnet_position][4])
    wh_tv_scores = [row["chi_nrmse"] for row in cpu_rows if row["method"] == "WH-TV"]
    matching_positions = [
        position
        for position, score in enumerate(wh_tv_scores)
        if score <= target_chi_nrmse
    ]
    target_reached = bool(matching_positions)
    selected_wh_tv_position = (
        matching_positions[0]
        if target_reached
        else min(
            range(len(wh_tv_scores)),
            key=lambda i: abs(wh_tv_scores[i] - target_chi_nrmse),
        )
    )
    if args.selection == "final":
        selected_wh_tv_position = wh_tv_iterations - 1
    selected_wh_tv_iteration = selected_wh_tv_position + 1
    if not target_reached and args.selection == "matched":
        print(
            "WH-TV did not reach WHNet's chi NRMSE; selecting the closest available iterate."
        )
    # Replay to the selected stopping point to recover its paired chi/phi state.
    # Trajectories and timings above continue to represent the complete runs.
    if selected_wh_tv_iteration != wh_tv_iterations:
        wh_tv_result = wh_tv(
            phase,
            dipole,
            mask,
            weight=mask,
            max_iterations=selected_wh_tv_iteration,
            tol_update=None,
            check_every=1,
            **parameters,
        )

    factor, pad = float(config["factor"]), int(config.get("pad", 0))
    interior = interior_mask(mask)
    masked_whnet_phi = whnet_phi * mask
    masked_wh_tv_chi = wh_tv_result.chi * mask
    masked_wh_tv_phi = wh_tv_result.background * mask
    reference_lap = positive_laplacian(phi_reference * factor)
    whnet_lap = positive_laplacian(masked_whnet_phi * factor)
    wh_tv_lap = positive_laplacian(masked_wh_tv_phi * factor)
    volumes = {
        "chi_reference": chi_reference * factor,
        "whnet_chi": whnet_chi * factor,
        "wh_tv_chi": masked_wh_tv_chi * factor,
        "phi_reference": phi_reference * factor,
        "whnet_phi": masked_whnet_phi * factor,
        "wh_tv_phi": masked_wh_tv_phi * factor,
        "whnet_chi_error": (whnet_chi - chi_reference) * mask * factor,
        "wh_tv_chi_error": (masked_wh_tv_chi - chi_reference) * mask * factor,
        "whnet_phi_error": (whnet_phi - phi_reference) * mask * factor,
        "wh_tv_phi_error": (masked_wh_tv_phi - phi_reference) * mask * factor,
        "laplacian_phi_reference": reference_lap * interior,
        "laplacian_whnet_phi": whnet_lap * interior,
        "laplacian_wh_tv_phi": wh_tv_lap * interior,
        "mask": mask,
    }
    arrays = {name: crop(volume, pad) for name, volume in volumes.items()}
    output = args.output_dir or RUNS_DIR / datetime.now(timezone.utc).strftime(
        "whnet_vs_wh_tv_%Y%m%d_%H%M%S_%f"
    )
    output.mkdir(parents=True, exist_ok=False)
    for name, array in arrays.items():
        save_nifti(output / f"{name}.nii.gz", array)
    with (output / "convergence.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cpu_rows[0]))
        writer.writeheader()
        writer.writerows(cpu_rows)
    selected_iterations = {
        "WHNet": best_whnet_iteration,
        "WH-TV": selected_wh_tv_iteration,
    }
    plot_rows = [
        row
        for row in cpu_rows
        if row["iteration"] <= selected_iterations[row["method"]]
    ]
    convergence_figure(plot_rows, output / "convergence.png")
    time_convergence_figure(plot_rows, output / "convergence_over_time.png")
    comparison_mask = arrays["mask"].astype(bool)
    laplacian_mask = crop(interior, pad).astype(bool)
    comparison_figure(
        [
            (
                f"WH-TV (iteration {selected_wh_tv_iteration})",
                arrays["wh_tv_phi"],
                masked_array_nrmse(
                    arrays["wh_tv_phi"], arrays["phi_reference"], comparison_mask
                ),
            ),
            (
                f"WHNet (iteration {best_whnet_iteration})",
                arrays["whnet_phi"],
                masked_array_nrmse(
                    arrays["whnet_phi"], arrays["phi_reference"], comparison_mask
                ),
            ),
            ("Ground Truth", arrays["phi_reference"], None),
        ],
        output / "phi_comparison.png",
        "Final harmonic-background reconstructions",
        "Harmonic background (ppm)",
    )
    comparison_figure(
        [
            (
                f"WH-TV (iteration {selected_wh_tv_iteration})",
                arrays["laplacian_wh_tv_phi"],
                masked_array_nrmse(
                    arrays["laplacian_wh_tv_phi"],
                    arrays["laplacian_phi_reference"],
                    laplacian_mask,
                ),
            ),
            (
                f"WHNet (iteration {best_whnet_iteration})",
                arrays["laplacian_whnet_phi"],
                masked_array_nrmse(
                    arrays["laplacian_whnet_phi"],
                    arrays["laplacian_phi_reference"],
                    laplacian_mask,
                ),
            ),
            ("Ground Truth", arrays["laplacian_phi_reference"], None),
        ],
        output / "laplacian_phi_comparison.png",
        "Interior Laplacian of harmonic background",
        "Laplacian (ppm/voxel²)",
        percentile=99.0,
    )
    comparison_figure(
        [
            (
                f"WH-TV (iteration {selected_wh_tv_iteration})",
                arrays["wh_tv_chi"],
                masked_array_nrmse(
                    arrays["wh_tv_chi"], arrays["chi_reference"], comparison_mask
                ),
            ),
            (
                f"WHNet (iteration {best_whnet_iteration})",
                arrays["whnet_chi"],
                masked_array_nrmse(
                    arrays["whnet_chi"], arrays["chi_reference"], comparison_mask
                ),
            ),
            ("Ground Truth", arrays["chi_reference"], None),
        ],
        output / "chi_comparison.png",
        "Final susceptibility reconstructions",
        "Susceptibility (ppm)",
    )
    for quantity, symbol in (("chi", "χ"), ("phi", "φₕ")):
        comparison_figure(
            [
                (
                    f"WH-TV (iteration {selected_wh_tv_iteration})",
                    arrays[f"wh_tv_{quantity}_error"],
                    None,
                ),
                (
                    f"WHNet (iteration {best_whnet_iteration})",
                    arrays[f"whnet_{quantity}_error"],
                    None,
                ),
            ],
            output / f"{quantity}_signed_residual_comparison.png",
            f"Signed {symbol} residual against ground truth (inside mask)",
            "Prediction - ground truth (ppm)",
            percentile=95.0,
            nonnegative=False,
        )
    metrics = {
        "selection": args.selection,
        "residual_definition": "reconstruction - reference inside mask",
        "residual_scale": "symmetric pooled 95th percentile of abs(residual); shared across methods per quantity",
        "checkpoint": str(args.checkpoint.resolve()),
        "wh_tv_run": str(wh_tv_run.resolve()),
        "simulation": {"snr": 100, "bg": 10},
        "whnet": {
            "iterations": args.whnet_iterations,
            "selected_iteration": best_whnet_iteration,
            "selection_rule": (
                "final iteration"
                if args.selection == "final"
                else "minimum masked chi NRMSE against ground truth"
            ),
            "minimum_chi_nrmse": minimum_whnet_nrmse,
            "minimum_chi_nrmse_iteration": minimum_whnet_position + 1,
            "instrumented_wall_seconds": whnet_wall_seconds,
            "diagnostic_seconds": diagnostic_seconds,
            "timing_method": "synchronized wall-clock minus measured diagnostic callback time",
            "configured_iterations": args.whnet_iterations,
            "seconds": whnet_seconds,
            "selected_iteration_seconds": whnet_iteration_seconds[best_whnet_position],
            "full_trajectory_seconds": whnet_seconds,
            "chi_nrmse": float(nrmse(whnet_chi, chi_reference, mask)),
            "phi_nrmse": float(nrmse(whnet_phi, phi_reference, mask)),
            "laplacian": laplacian_metrics(
                whnet_lap, masked_whnet_phi * factor, interior
            ),
        },
        "wh_tv": {
            "iterations": wh_tv_iterations,
            "seconds": wh_tv_seconds,
            "selected_iteration": selected_wh_tv_iteration,
            "selected_iteration_seconds": wh_tv_iteration_seconds[
                selected_wh_tv_position
            ],
            "target_chi_nrmse": target_chi_nrmse,
            "target_reached": target_reached,
            "selection_rule": (
                "final iteration"
                if args.selection == "final"
                else "first chi NRMSE <= WHNet minimum; closest if never reached"
            ),
            "full_trajectory_chi_nrmse": wh_tv_scores[-1],
            "parameters": parameters,
            "chi_nrmse": float(nrmse(masked_wh_tv_chi, chi_reference, mask)),
            "phi_nrmse": float(nrmse(masked_wh_tv_phi, phi_reference, mask)),
            "laplacian": laplacian_metrics(
                wh_tv_lap, masked_wh_tv_phi * factor, interior
            ),
        },
        "reference_laplacian": laplacian_metrics(
            reference_lap, phi_reference * factor, interior
        ),
        "laplacian_mask": "one-voxel six-neighbor interior",
        "laplacian_units": "ppm/voxel^2",
        "error_definition": "reconstruction - reference",
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Saved benchmark to {output}")


if __name__ == "__main__":
    main()

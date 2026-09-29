"""Create camera-ready reconstruction and runtime figures from a benchmark run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

TEXT_COLOR = "#0F2A70"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def load_volume(run_dir: Path, name: str) -> np.ndarray:
    return np.asarray(nib.load(run_dir / f"{name}.nii.gz").dataobj, dtype=np.float32)


def display_slice(volume: np.ndarray, bounds: tuple[slice, slice, int]) -> np.ndarray:
    rows, columns, axial = bounds
    return np.rot90(volume[axial, rows, columns])


def axial_bounds(mask: np.ndarray, margin: int = 2) -> tuple[slice, slice, int]:
    coordinates = np.argwhere(mask)
    if not coordinates.size:
        raise ValueError("Mask is empty")
    axial = int(np.round(np.median(coordinates[:, 0])))
    support = mask[axial]
    if not support.any():
        axial = int(coordinates[len(coordinates) // 2, 0])
        support = mask[axial]
    rows, columns = np.where(support)
    row_slice = slice(
        max(int(rows.min()) - margin, 0),
        min(int(rows.max()) + margin + 1, mask.shape[1]),
    )
    column_slice = slice(
        max(int(columns.min()) - margin, 0),
        min(int(columns.max()) + margin + 1, mask.shape[2]),
    )
    return row_slice, column_slice, axial


def pooled_limit(volumes: list[np.ndarray], quantile: float) -> float:
    values = np.concatenate([np.abs(volume).ravel() for volume in volumes])
    return max(float(np.quantile(values, quantile)), np.finfo(np.float32).eps)


def image_figure(run_dir: Path, output_dir: Path, dpi: int) -> dict[str, float]:
    metrics = json.loads((run_dir / "metrics.json").read_text())
    mask = load_volume(run_dir, "mask").astype(bool)
    bounds = axial_bounds(mask)
    methods = (
        ("wh_tv", f"WH-TV · iter. {metrics['wh_tv']['selected_iteration']:,}"),
        ("whnet", f"WHNet · iter. {metrics['whnet']['selected_iteration']:,}"),
        ("reference", "Ground truth"),
    )
    data = {
        "chi": {
            name: load_volume(
                run_dir, f"{name}_chi" if name != "reference" else "chi_reference"
            )
            for name, _ in methods
        },
        "phi": {
            name: load_volume(
                run_dir, f"{name}_phi" if name != "reference" else "phi_reference"
            )
            for name, _ in methods
        },
        "laplacian": {
            name: load_volume(
                run_dir,
                f"laplacian_{name}_phi"
                if name != "reference"
                else "laplacian_phi_reference",
            )
            for name, _ in methods
        },
    }
    zero = np.zeros_like(data["chi"]["reference"])
    data["chi_residual"] = {
        "wh_tv": load_volume(run_dir, "wh_tv_chi_error"),
        "whnet": load_volume(run_dir, "whnet_chi_error"),
        "reference": zero,
    }
    data["phi_residual"] = {
        "wh_tv": load_volume(run_dir, "wh_tv_phi_error"),
        "whnet": load_volume(run_dir, "whnet_phi_error"),
        "reference": zero,
    }
    rows = (
        ("chi", "Susceptibility χ", "χ (ppm)", 0.995),
        ("chi_residual", "χ residual", "χ pred. − GT (ppm)", 0.95),
        ("phi", "Harmonic field φₕ", "φₕ (ppm)", 0.995),
        ("phi_residual", "φₕ residual", "φₕ pred. − GT (ppm)", 0.95),
        ("laplacian", "Laplacian of φₕ", "∇²φₕ (ppm/voxel²)", 0.99),
    )
    limits = {}
    for name, _, _, quantile in rows:
        scale_methods = methods[:2] if name.endswith("residual") else methods
        limits[name] = pooled_limit(
            [data[name][method] for method, _ in scale_methods], quantile
        )

    figure = plt.figure(figsize=(7.2, 9.0))
    grid = figure.add_gridspec(
        5, 4, width_ratios=(1, 1, 1, 0.045), wspace=0.015, hspace=0.035
    )
    for row_index, (quantity, row_label, colorbar_label, _quantile) in enumerate(rows):
        limit = limits[quantity]
        image = None
        for column, (method, title) in enumerate(methods):
            axis = figure.add_subplot(grid[row_index, column])
            image = axis.imshow(
                display_slice(data[quantity][method], bounds),
                cmap="gray",
                vmin=-limit,
                vmax=limit,
            )
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if row_index == 0:
                axis.set_title(title, fontsize=10, pad=3, weight="semibold")
            if column == 0:
                axis.set_ylabel(row_label, fontsize=9, labelpad=3)
            if quantity in ("chi", "phi") and method != "reference":
                score = metrics[method][f"{quantity}_nrmse"]
                axis.text(
                    0.025,
                    0.965,
                    f"NRMSE {score:.4f}",
                    transform=axis.transAxes,
                    ha="left",
                    va="top",
                    color="white",
                    fontsize=7.5,
                    bbox={
                        "facecolor": "black",
                        "edgecolor": "none",
                        "alpha": 0.62,
                        "pad": 1.5,
                    },
                )
        color_axis = figure.add_subplot(grid[row_index, 3])
        figure.colorbar(image, cax=color_axis, label=colorbar_label)
        color_axis.tick_params(labelsize=7, length=2, pad=1)
        color_axis.yaxis.label.set_size(8)
    figure.subplots_adjust(left=0.065, right=0.985, bottom=0.01, top=0.965)
    save_figure(figure, output_dir / "poster_reconstructions", dpi)
    return limits


def read_trajectory(run_dir: Path, metrics: dict) -> dict[str, list[dict[str, float]]]:
    limits = {
        "WHNet": metrics["whnet"]["selected_iteration"],
        "WH-TV": metrics["wh_tv"]["selected_iteration"],
    }
    trajectories = {"WHNet": [], "WH-TV": []}
    with (run_dir / "convergence.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            method, iteration = row["method"], int(row["iteration"])
            if iteration <= limits[method]:
                trajectories[method].append(
                    {
                        "iteration": float(iteration),
                        "elapsed_seconds": float(row["elapsed_seconds"]),
                        "chi_nrmse": float(row["chi_nrmse"]),
                        "phi_nrmse": float(row["phi_nrmse"]),
                    }
                )
    for method, rows in trajectories.items():
        if len(rows) != limits[method]:
            raise ValueError(f"Incomplete selected trajectory for {method}")
    return trajectories


def runtime_figure(run_dir: Path, output_dir: Path, dpi: int) -> None:
    metrics = json.loads((run_dir / "metrics.json").read_text())
    trajectories = read_trajectory(run_dir, metrics)
    colors = {"WH-TV": "#0072B2", "WHNet": "#D55E00"}
    display_names = {"WH-TV": "WH-QSM", "WHNet": "WHNet"}
    quantities = (("chi_nrmse", "χ", "-"), ("phi_nrmse", "φₕ", "--"))
    plt.rcParams.update(
        {
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.edgecolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.75))
    handles = []
    for method in ("WH-TV", "WHNet"):
        rows = trajectories[method]
        iteration = np.array([0.0, *[row["iteration"] for row in rows]])
        elapsed = np.array([0.0, *[row["elapsed_seconds"] for row in rows]])
        for field, symbol, linestyle in quantities:
            values = np.array([1.0, *[row[field] for row in rows]])
            label = f"{display_names[method]} {symbol}"
            line = axes[0].plot(
                iteration,
                values,
                color=colors[method],
                linestyle=linestyle,
                linewidth=1.45,
                label=label,
            )[0]
            axes[1].plot(
                elapsed,
                values,
                color=colors[method],
                linestyle=linestyle,
                linewidth=1.45,
            )
            axes[0].scatter(
                iteration[-1], values[-1], color=colors[method], s=14, zorder=3
            )
            axes[1].scatter(
                elapsed[-1], values[-1], color=colors[method], s=14, zorder=3
            )
            handles.append(line)
    axes[0].set_xlabel("Iteration")
    axes[1].set_xlabel("Wall-clock time (s)")
    axes[0].set_ylabel("NRMSE")
    axes[0].set_xscale("symlog", linthresh=1.0)
    axes[0].set_xlim(left=0)
    axes[1].set_xlim(
        0, max(rows[-1]["elapsed_seconds"] for rows in trajectories.values()) * 1.02
    )
    for axis in axes:
        axis.set_yscale("log")
        axis.grid(True, which="both", linewidth=0.35, alpha=0.35)
        axis.tick_params(labelsize=8, colors=TEXT_COLOR, pad=2)
        axis.xaxis.label.set_size(9)
        axis.yaxis.label.set_size(9)
    figure.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, 0.995),
    )
    figure.subplots_adjust(left=0.075, right=0.995, bottom=0.17, top=0.82, wspace=0.16)
    save_figure(figure, output_dir / "poster_runtime", dpi)


def save_figure(figure: plt.Figure, stem: Path, dpi: int) -> None:
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            stem.with_suffix(f".{suffix}"),
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.01,
        )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.run_dir / "poster_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    limits = image_figure(args.run_dir, output_dir, args.dpi)
    runtime_figure(args.run_dir, output_dir, args.dpi)
    (output_dir / "figure_metadata.json").write_text(
        json.dumps(
            {"source_run": str(args.run_dir.resolve()), "display_limits": limits},
            indent=2,
        )
        + "\n"
    )
    print(f"Saved poster figures to {output_dir.resolve()}")


if __name__ == "__main__":
    main()

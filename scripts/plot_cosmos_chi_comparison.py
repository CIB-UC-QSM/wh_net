"""Create a publication-ready COSMOS chi/residual comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import nibabel as nib
import numpy as np

from whnet.paths import EVALUATION_VOLUMES_DIR, EXPORTS_DIR, RUNS_DIR

TEXT_COLOR = "#0F2A70"
DEFAULT_RUN = RUNS_DIR / "whnet_vs_wh_tv_20260914_164004_514282"
DEFAULT_IQSM = EVALUATION_VOLUMES_DIR / "iQSM.nii.gz"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--wh-tv", type=Path)
    parser.add_argument(
        "--wh-qsm",
        type=Path,
        help="WH-QSM volume; defaults to the run's whnet_chi.nii.gz output.",
    )
    parser.add_argument("--iqsm", type=Path, default=DEFAULT_IQSM)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--chi-quantile", type=float, default=0.995)
    parser.add_argument("--difference-quantile", type=float, default=0.995)
    parser.add_argument(
        "--output", type=Path, default=EXPORTS_DIR / "cosmos_method_comparison"
    )
    args = parser.parse_args(argv)
    if not 0 < args.chi_quantile <= 1 or not 0 < args.difference_quantile <= 1:
        parser.error("display quantiles must be in (0, 1]")
    run = args.run
    args.wh_tv = args.wh_tv or run / "wh_tv_chi.nii.gz"
    args.wh_qsm = args.wh_qsm or run / "whnet_chi.nii.gz"
    args.ground_truth = args.ground_truth or run / "chi_reference.nii.gz"
    args.mask = args.mask or run / "mask.nii.gz"
    return args


def load_volume(path: Path) -> np.ndarray:
    volume = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    if volume.ndim != 3 or not np.isfinite(volume).all():
        raise ValueError(f"{path} must contain one finite 3D volume")
    return volume


def rounded_limit(values: np.ndarray, quantile: float) -> float:
    raw = float(np.quantile(np.abs(values), quantile))
    if not np.isfinite(raw) or raw <= 0:
        raise ValueError("Cannot determine a positive display limit")
    step = 0.05 if raw >= 0.2 else 0.01
    return float(step * np.ceil(raw / step))


def nrmse(prediction: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> float:
    selected = mask.astype(bool)
    denominator = np.linalg.norm(reference[selected])
    return float(
        np.linalg.norm((prediction - reference)[selected])
        / max(float(denominator), np.finfo(np.float32).eps)
    )


def create_figure(args: argparse.Namespace) -> dict:
    ground_truth = load_volume(args.ground_truth)
    mask = load_volume(args.mask) > 0.5
    if ground_truth.shape != mask.shape:
        raise ValueError("Ground-truth and mask shapes differ")
    method_paths = [
        ("WH-TV", args.wh_tv),
        ("WH-QSM", args.wh_qsm),
        ("iQSM", args.iqsm),
    ]
    methods = []
    for name, path in method_paths:
        prediction = load_volume(path)
        if prediction.shape != ground_truth.shape:
            raise ValueError(
                f"{name} shape {prediction.shape} differs from COSMOS {ground_truth.shape}"
            )
        prediction = np.where(mask, prediction, 0.0)
        methods.append((name, prediction, path))
    ground_truth = np.where(mask, ground_truth, 0.0)

    chi_values = np.concatenate(
        [ground_truth[mask]] + [prediction[mask] for _, prediction, _ in methods]
    )
    residuals = [prediction - ground_truth for _, prediction, _ in methods]
    residual_values = np.concatenate([residual[mask] for residual in residuals])
    chi_limit = rounded_limit(chi_values, args.chi_quantile)
    difference_limit = rounded_limit(residual_values, args.difference_quantile)
    chi_norm = Normalize(-chi_limit, chi_limit, clip=True)
    difference_norm = Normalize(-difference_limit, difference_limit, clip=True)
    chi_cmap = matplotlib.colormaps["gray"].copy()
    difference_cmap = matplotlib.colormaps["RdBu_r"].copy()

    views = [
        ("Axial", 2, ground_truth.shape[2] // 2),
        ("Sagittal", 0, ground_truth.shape[0] // 2),
        ("Coronal", 1, ground_truth.shape[1] // 2),
    ]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.edgecolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(14.0, 9.0))
    grid = figure.add_gridspec(
        3,
        6,
        left=0.035,
        right=0.995,
        bottom=0.13,
        top=0.87,
        hspace=0.025,
        wspace=0.012,
    )
    for row, (view_name, axis_index, slice_index) in enumerate(views):
        first_chi_axis = None
        for method_index, (name, prediction, _) in enumerate(methods):
            view = np.rot90(np.take(prediction, slice_index, axis=axis_index))
            difference = np.rot90(
                np.take(residuals[method_index], slice_index, axis=axis_index)
            )
            chi_axis = figure.add_subplot(grid[row, 2 * method_index])
            difference_axis = figure.add_subplot(grid[row, 2 * method_index + 1])
            chi_axis.imshow(
                view,
                cmap=chi_cmap,
                norm=chi_norm,
                interpolation="nearest",
                rasterized=True,
            )
            difference_axis.imshow(
                difference,
                cmap=difference_cmap,
                norm=difference_norm,
                interpolation="nearest",
                rasterized=True,
            )
            chi_axis.set_axis_off()
            difference_axis.set_axis_off()
            if method_index == 0:
                first_chi_axis = chi_axis
            if row == 0:
                chi_axis.set_title(f"{name}\nχ", pad=5, linespacing=1.2)
                difference_axis.set_title(f"{name}\nΔχ", pad=5, linespacing=1.2)
        first_chi_axis.text(
            -0.08,
            0.5,
            view_name,
            transform=chi_axis.transAxes,
            rotation=90,
            ha="right",
            va="center",
            color=TEXT_COLOR,
            fontsize=10,
            clip_on=False,
        )

    chi_bar_axis = figure.add_axes([0.12, 0.057, 0.32, 0.018])
    difference_bar_axis = figure.add_axes([0.56, 0.057, 0.32, 0.018])
    chi_bar = figure.colorbar(
        ScalarMappable(norm=chi_norm, cmap=chi_cmap),
        cax=chi_bar_axis,
        orientation="horizontal",
    )
    difference_bar = figure.colorbar(
        ScalarMappable(norm=difference_norm, cmap=difference_cmap),
        cax=difference_bar_axis,
        orientation="horizontal",
    )
    chi_bar.set_label("χ (ppm)", color=TEXT_COLOR, labelpad=3)
    difference_bar.set_label(
        "Δχ = prediction − COSMOS ground truth (ppm)", color=TEXT_COLOR, labelpad=3
    )
    for colorbar in (chi_bar, difference_bar):
        colorbar.set_ticks(np.linspace(-colorbar.norm.vmax, colorbar.norm.vmax, 5))
        colorbar.ax.tick_params(length=3, width=0.7, colors=TEXT_COLOR, pad=2)
        colorbar.outline.set_edgecolor(TEXT_COLOR)
        for tick in colorbar.ax.get_xticklabels():
            tick.set_color(TEXT_COLOR)

    figure.suptitle(
        "COSMOS χ reconstruction comparison",
        x=0.035,
        y=0.985,
        ha="left",
        color=TEXT_COLOR,
        fontsize=17,
        fontweight="bold",
    )
    figure.text(
        0.995,
        0.972,
        "SNR = 100 · BG = 10 · central orthogonal slices",
        ha="right",
        va="top",
        color=TEXT_COLOR,
        fontsize=9,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    png_path = args.output.with_suffix(".png")
    pdf_path = args.output.with_suffix(".pdf")
    svg_path = args.output.with_suffix(".svg")
    figure.savefig(png_path, dpi=400, bbox_inches="tight", pad_inches=0.02)
    figure.savefig(pdf_path, dpi=400, bbox_inches="tight", pad_inches=0.02)
    figure.savefig(svg_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)

    metadata = {
        "source_run": str(args.run.resolve()),
        "ground_truth": str(args.ground_truth.resolve()),
        "mask": str(args.mask.resolve()),
        "methods": {
            name: {
                "path": str(path.resolve()),
                "chi_nrmse": nrmse(prediction, ground_truth, mask),
            }
            for (name, prediction, path) in methods
        },
        "wh_qsm_mapping": "WH-QSM label uses the run's whnet_chi.nii.gz because no separate wh_qsm file is present.",
        "shape": list(ground_truth.shape),
        "voxel_size_mm": [
            float(value) for value in nib.load(args.ground_truth).header.get_zooms()[:3]
        ],
        "views": {
            name.lower(): {"axis": axis, "index": index} for name, axis, index in views
        },
        "display": {
            "text_color": TEXT_COLOR,
            "chi_colormap": "gray",
            "difference_colormap": "RdBu_r",
            "chi_limit_ppm": chi_limit,
            "difference_limit_ppm": difference_limit,
            "chi_quantile": args.chi_quantile,
            "difference_quantile": args.difference_quantile,
            "background_value_ppm": 0.0,
        },
        "outputs": {
            "png": str(png_path.resolve()),
            "pdf": str(pdf_path.resolve()),
            "svg": str(svg_path.resolve()),
        },
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main(argv: list[str] | None = None) -> None:
    metadata = create_figure(parse_args(argv))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

"""Create compact vertical COSMOS reconstruction and residual composites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

from whnet.paths import EVALUATION_VOLUMES_DIR, EXPORTS_DIR, RUNS_DIR

TEXT_COLOR = "#0F2A70"
DEFAULT_RUN = RUNS_DIR / "whnet_vs_wh_tv_20260914_164004_514282"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument(
        "--iqsm", type=Path, default=EVALUATION_VOLUMES_DIR / "iQSM.nii.gz"
    )
    parser.add_argument("--wh-qsm", type=Path)
    parser.add_argument("--wh-net", type=Path)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--chi-quantile", type=float, default=0.995)
    parser.add_argument("--difference-quantile", type=float, default=0.995)
    parser.add_argument(
        "--output", type=Path, default=EXPORTS_DIR / "cosmos_composite_comparison"
    )
    args = parser.parse_args(argv)
    if not 0 < args.chi_quantile <= 1 or not 0 < args.difference_quantile <= 1:
        parser.error("display quantiles must be in (0, 1]")
    args.wh_qsm = args.wh_qsm or args.run / "wh_tv_chi.nii.gz"
    args.wh_net = args.wh_net or args.run / "whnet_chi.nii.gz"
    args.ground_truth = args.ground_truth or args.run / "chi_reference.nii.gz"
    args.mask = args.mask or args.run / "mask.nii.gz"
    return args


def load_volume(path: Path) -> np.ndarray:
    volume = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    if volume.ndim != 3 or not np.isfinite(volume).all():
        raise ValueError(f"{path} must contain one finite 3D volume")
    return volume


def nrmse(prediction: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> float:
    difference = prediction[mask] - reference[mask]
    denominator = np.linalg.norm(reference[mask])
    return float(
        np.linalg.norm(difference) / max(float(denominator), np.finfo(np.float32).eps)
    )


def rounded_limit(values: np.ndarray, quantile: float) -> float:
    raw = float(np.quantile(np.abs(values), quantile))
    if not np.isfinite(raw) or raw <= 0:
        raise ValueError("Cannot determine a positive display limit")
    step = 0.05 if raw >= 0.2 else 0.01
    return float(step * np.ceil(raw / step))


def cropped_views(
    volume: np.ndarray,
    mask: np.ndarray,
    *,
    rotate_axial_180: bool,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    definitions = (
        ("Axial", 2, volume.shape[2] // 2),
        ("Sagittal", 0, volume.shape[0] // 2),
    )
    output = []
    for name, axis, index in definitions:
        image = np.rot90(np.take(volume, index, axis=axis))
        support = np.rot90(np.take(mask, index, axis=axis))
        if name == "Axial" and rotate_axial_180:
            image = np.rot90(image, 2)
            support = np.rot90(support, 2)
        coordinates = np.argwhere(support)
        if not coordinates.size:
            raise ValueError(f"Central {name.lower()} mask slice is empty")
        lower, upper = coordinates.min(0), coordinates.max(0) + 1
        crop = tuple(
            slice(int(lo), int(hi)) for lo, hi in zip(lower, upper, strict=True)
        )
        output.append((name, image[crop], support[crop]))
    return output


def resize_view(
    image: np.ndarray, support: np.ndarray, target_width: int
) -> tuple[np.ndarray, np.ndarray]:
    scale = target_width / image.shape[1]
    resized = zoom(image, (scale, scale), order=1, mode="nearest", prefilter=False)
    resized_support = zoom(
        support.astype(np.uint8),
        (scale, scale),
        order=0,
        mode="nearest",
        prefilter=False,
    ).astype(bool)
    if resized.shape[1] != target_width:
        raise RuntimeError("Unexpected resized view width")
    return np.asarray(resized, dtype=np.float32), resized_support


def composite(
    volume: np.ndarray,
    mask: np.ndarray,
    *,
    rotate_axial_180: bool,
    outside: float,
) -> tuple[np.ndarray, list[dict[str, int | str]]]:
    views = cropped_views(volume, mask, rotate_axial_180=rotate_axial_180)
    target_width = max(image.shape[1] for _, image, _ in views)
    images = []
    metadata = []
    offset = 0
    for name, image, support in views:
        image, support = resize_view(image, support, target_width)
        image = np.where(support, image, outside)
        images.append(image)
        metadata.append(
            {
                "orientation": name,
                "start_row": offset,
                "end_row": offset + image.shape[0],
                "width": image.shape[1],
            }
        )
        offset += image.shape[0]
    return np.concatenate(images, axis=0), metadata


def style_colorbar(colorbar: matplotlib.colorbar.Colorbar, label: str) -> None:
    colorbar.set_label(label, color=TEXT_COLOR, labelpad=7)
    colorbar.set_ticks(np.linspace(-colorbar.norm.vmax, colorbar.norm.vmax, 5))
    colorbar.ax.tick_params(length=3, width=0.7, colors=TEXT_COLOR, pad=3)
    colorbar.outline.set_edgecolor(TEXT_COLOR)
    for tick in colorbar.ax.get_yticklabels():
        tick.set_color(TEXT_COLOR)


def create_figure(args: argparse.Namespace) -> dict:
    reference = load_volume(args.ground_truth)
    mask = load_volume(args.mask) > 0.5
    if reference.shape != mask.shape:
        raise ValueError("Ground-truth and mask shapes differ")
    sources = [
        ("iQSM", args.iqsm),
        ("WH-QSM", args.wh_qsm),
        ("WH-Net", args.wh_net),
    ]
    methods = []
    for name, path in sources:
        prediction = load_volume(path)
        if prediction.shape != reference.shape:
            raise ValueError(
                f"{name} shape {prediction.shape} differs from COSMOS {reference.shape}"
            )
        methods.append((name, np.where(mask, prediction, 0.0), path))
    reference = np.where(mask, reference, 0.0)
    residuals = [prediction - reference for _, prediction, _ in methods]
    chi_values = np.concatenate([prediction[mask] for _, prediction, _ in methods])
    difference_values = np.concatenate([residual[mask] for residual in residuals])
    chi_limit = rounded_limit(chi_values, args.chi_quantile)
    difference_limit = rounded_limit(difference_values, args.difference_quantile)
    chi_norm = Normalize(-chi_limit, chi_limit, clip=True)
    difference_norm = Normalize(-difference_limit, difference_limit, clip=True)
    chi_cmap = matplotlib.colormaps["gray"].copy()
    difference_cmap = matplotlib.colormaps["RdBu_r"].copy()
    difference_cmap.set_bad("#D9DEE8")

    reconstruction_composites = []
    difference_composites = []
    view_metadata = None
    for (_, prediction, _), residual in zip(methods, residuals, strict=True):
        reconstruction, metadata = composite(
            prediction,
            mask,
            rotate_axial_180=True,
            outside=0.0,
        )
        difference, _ = composite(
            residual,
            mask,
            rotate_axial_180=True,
            outside=np.nan,
        )
        reconstruction_composites.append(reconstruction)
        difference_composites.append(difference)
        view_metadata = metadata
    shapes = {
        array.shape for array in reconstruction_composites + difference_composites
    }
    if len(shapes) != 1:
        raise RuntimeError(f"Composite shapes differ: {sorted(shapes)}")

    scores = {
        name: nrmse(prediction, reference, mask) for name, prediction, _ in methods
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
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
    figure = plt.figure(figsize=(10.2, 12.6))
    grid = figure.add_gridspec(
        2,
        4,
        width_ratios=(1, 1, 1, 0.055),
        left=0.075,
        right=0.965,
        bottom=0.025,
        top=0.955,
        hspace=0.018,
        wspace=0.012,
    )
    for column, ((name, _, _), reconstruction, difference) in enumerate(
        zip(methods, reconstruction_composites, difference_composites, strict=True)
    ):
        reconstruction_axis = figure.add_subplot(grid[0, column])
        difference_axis = figure.add_subplot(grid[1, column])
        reconstruction_axis.imshow(
            reconstruction,
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
        for axis in (reconstruction_axis, difference_axis):
            axis.set_axis_off()
        reconstruction_axis.set_title(
            f"{name}\nNRMSE: {scores[name]:.2f}",
            color=TEXT_COLOR,
            pad=7,
            linespacing=1.25,
        )

    reconstruction_bar = figure.colorbar(
        ScalarMappable(norm=chi_norm, cmap=chi_cmap),
        cax=figure.add_subplot(grid[0, 3]),
        orientation="vertical",
    )
    difference_bar = figure.colorbar(
        ScalarMappable(norm=difference_norm, cmap=difference_cmap),
        cax=figure.add_subplot(grid[1, 3]),
        orientation="vertical",
    )
    style_colorbar(reconstruction_bar, "Magnetic susceptibility, χ (ppm)")
    style_colorbar(difference_bar, "Difference, prediction − COSMOS (ppm)")
    figure.text(
        0.022,
        0.735,
        "QSM Reconstruction",
        rotation=90,
        ha="center",
        va="center",
        color=TEXT_COLOR,
        fontsize=14,
        fontweight="semibold",
    )
    figure.text(
        0.022,
        0.265,
        "Difference Map",
        rotation=90,
        ha="center",
        va="center",
        color=TEXT_COLOR,
        fontsize=14,
        fontweight="semibold",
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
                "chi_nrmse": scores[name],
            }
            for name, _, path in methods
        },
        "method_mapping": {
            "WH-QSM": "The source run's classical weak-harmonic wh_tv_chi.nii.gz output.",
            "WH-Net": "The source run's learned whnet_chi.nii.gz output.",
        },
        "residual_definition": "prediction - COSMOS ground truth inside the mask",
        "shape": list(reference.shape),
        "voxel_size_mm": [
            float(value) for value in nib.load(args.ground_truth).header.get_zooms()[:3]
        ],
        "composite_shape": list(reconstruction_composites[0].shape),
        "composite_views": view_metadata,
        "axial_rotation": {
            "qsm_reconstruction": 180,
            "difference_map": 180,
        },
        "display": {
            "text_color": TEXT_COLOR,
            "chi_colormap": "gray",
            "difference_colormap": "RdBu_r",
            "chi_limit_ppm": chi_limit,
            "difference_limit_ppm": difference_limit,
            "chi_quantile": args.chi_quantile,
            "difference_quantile": args.difference_quantile,
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

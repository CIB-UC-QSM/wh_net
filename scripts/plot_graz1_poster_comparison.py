"""Create a poster-ready Graz comparison of WHNet, WH-TV, and iQSM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import nibabel as nib
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import zoom

from whnet.paths import (
    EVALUATION_DATA_DIR,
    EVALUATION_VOLUMES_DIR,
    EXPORTS_DIR,
    RUNS_DIR,
)

DEFAULT_RUN = RUNS_DIR / "graz_whnet_wh_tv_20260915_153321_072659"
TEXT_COLOR = "#0F2A70"
SOURCES = {
    "graz1": EVALUATION_DATA_DIR / "data_blend_1_graz.mat",
    "graz2": EVALUATION_DATA_DIR / "data_blending2.mat",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(SOURCES), default="graz1")
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--whnet", type=Path)
    parser.add_argument("--wh-tv", type=Path)
    parser.add_argument("--wh-tv-subtitle", default="L-curve tuned")
    parser.add_argument(
        "--iqsm", type=Path, default=EVALUATION_VOLUMES_DIR / "iQSM.nii.gz"
    )
    parser.add_argument("--source", type=Path)
    parser.add_argument("--pad", type=int, default=20)
    parser.add_argument("--vmin", type=float, default=-0.25)
    parser.add_argument("--vmax", type=float, default=0.25)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.pad < 0:
        parser.error("pad must be nonnegative")
    if args.vmin >= args.vmax:
        parser.error("vmin must be smaller than vmax")
    dataset_run = args.run / args.dataset
    args.whnet = args.whnet or dataset_run / "whnet_chi.nii.gz"
    args.wh_tv = args.wh_tv or dataset_run / "wh_tv_l_curve_chi.nii.gz"
    args.source = args.source or SOURCES[args.dataset]
    args.output = args.output or EXPORTS_DIR / f"{args.dataset}_method_comparison"
    return args


def pad_cube(array: np.ndarray, size: int) -> np.ndarray:
    padding = []
    for length in array.shape:
        total = size - length
        if total < 0:
            raise ValueError(f"Cannot pad shape {array.shape} to {size} cubed")
        padding.append((total // 2, total - total // 2))
    return np.pad(array, padding)


def graz_mask(source_path: Path, pad: int) -> tuple[np.ndarray, np.ndarray]:
    source = loadmat(source_path, variable_names=("mask", "voxel_size"))
    if "mask" not in source or "voxel_size" not in source:
        raise KeyError(f"{source_path} does not contain mask and voxel_size")
    voxel_size = np.asarray(source["voxel_size"], dtype=float).ravel()
    mask = zoom(
        np.asarray(source["mask"], dtype=np.uint8),
        voxel_size,
        order=0,
        mode="nearest",
    ).astype(bool)
    coordinates = np.argwhere(mask)
    if not coordinates.size:
        raise ValueError("Graz mask is empty")
    lower, upper = coordinates.min(0), coordinates.max(0) + 1
    crop = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper, strict=True))
    cropped = mask[crop]
    padded = pad_cube(cropped, max(cropped.shape) + 2 * pad)
    return cropped, padded


def load_volume(path: Path) -> np.ndarray:
    volume = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    if volume.ndim != 3 or not np.isfinite(volume).all():
        raise ValueError(f"{path} must contain one finite 3D volume")
    return volume


def crop_to_mask(
    volume: np.ndarray, cropped_mask: np.ndarray, padded_mask: np.ndarray
) -> np.ndarray:
    if volume.shape == cropped_mask.shape:
        return volume
    if volume.shape != padded_mask.shape:
        raise ValueError(
            f"Volume shape {volume.shape} matches neither cropped "
            f"{cropped_mask.shape} nor padded {padded_mask.shape} Graz geometry"
        )
    coordinates = np.argwhere(padded_mask)
    lower, upper = coordinates.min(0), coordinates.max(0) + 1
    crop = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper, strict=True))
    return volume[crop]


def create_figure(args: argparse.Namespace) -> dict:
    cropped_mask, padded_mask = graz_mask(args.source, args.pad)
    methods = [
        ("iQSM", "", args.iqsm),
        ("WH-QSM", args.wh_tv_subtitle, args.wh_tv),
        ("WH-Net", "48 iterations", args.whnet),
    ]
    arrays = []
    for name, subtitle, path in methods:
        volume = crop_to_mask(load_volume(path), cropped_mask, padded_mask)
        if volume.shape != cropped_mask.shape:
            raise AssertionError("Internal crop mismatch")
        arrays.append((name, subtitle, np.where(cropped_mask, volume, 0.0), path))

    normalization = Normalize(vmin=args.vmin, vmax=args.vmax, clip=True)
    cmap = matplotlib.colormaps["gray"].copy()

    views = [
        ("Axial", 2, cropped_mask.shape[2] // 2),
        ("Sagittal", 0, cropped_mask.shape[0] // 2),
        ("Coronal", 1, cropped_mask.shape[1] // 2),
    ]
    view_aspects = [
        np.take(arrays[0][2], index, axis=axis).shape[1]
        / np.take(arrays[0][2], index, axis=axis).shape[0]
        for _, axis, index in views
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
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
    figure = plt.figure(figsize=(9.1, 8.8))
    grid = figure.add_gridspec(
        4,
        3,
        height_ratios=(*view_aspects, 0.075),
        left=0.01,
        right=0.995,
        bottom=0.075,
        top=0.94,
        hspace=0.018,
        wspace=0.018,
    )
    image = None
    for row, (_, axis_index, slice_index) in enumerate(views):
        for column, (name, subtitle, volume, _) in enumerate(arrays):
            axis = figure.add_subplot(grid[row, column])
            view = np.rot90(np.take(volume, slice_index, axis=axis_index))
            image = axis.imshow(
                view,
                cmap=cmap,
                norm=normalization,
                interpolation="nearest",
                rasterized=True,
            )
            axis.set_axis_off()
            if row == 0:
                title = name if not subtitle else f"{name}\n{subtitle}"
                axis.set_title(
                    title,
                    color=TEXT_COLOR,
                    pad=6,
                    linespacing=1.25,
                )
    colorbar_axis = figure.add_subplot(grid[3, :])
    colorbar = figure.colorbar(image, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label("Magnetic susceptibility, χ (ppm)", labelpad=4)
    colorbar.ax.xaxis.label.set_color(TEXT_COLOR)
    colorbar.set_ticks(np.linspace(args.vmin, args.vmax, 5))
    colorbar.outline.set_linewidth(0.7)
    colorbar.outline.set_edgecolor(TEXT_COLOR)
    colorbar.ax.tick_params(length=3, width=0.7, colors=TEXT_COLOR, pad=2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    png_path = args.output.with_suffix(".png")
    pdf_path = args.output.with_suffix(".pdf")
    figure.savefig(png_path, dpi=400, bbox_inches="tight", pad_inches=0.02)
    figure.savefig(pdf_path, dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)

    metadata = {
        "dataset": args.dataset,
        "inputs": {name: str(path.resolve()) for name, _, _, path in arrays},
        "source": str(args.source.resolve()),
        "shape": list(cropped_mask.shape),
        "voxel_size_mm": [1.0, 1.0, 1.0],
        "slices": {
            name.lower(): {"axis": axis, "index": index} for name, axis, index in views
        },
        "display": {
            "colormap": "gray",
            "vmin_ppm": args.vmin,
            "vmax_ppm": args.vmax,
            "center_ppm": 0.0,
            "text_color": TEXT_COLOR,
            "shared_across_methods_and_orientations": True,
        },
        "column_order": [name for name, _, _, _ in arrays],
        "outputs": {"png": str(png_path.resolve()), "pdf": str(pdf_path.resolve())},
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main(argv: list[str] | None = None) -> None:
    metadata = create_figure(parse_args(argv))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

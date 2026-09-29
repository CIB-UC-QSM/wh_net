"""Create a publication-ready phi_h and Laplacian comparison from a benchmark run."""

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

from whnet.paths import EXPORTS_DIR, RUNS_DIR

TEXT_COLOR = "#0F2A70"
DEFAULT_RUN = RUNS_DIR / "whnet_vs_wh_tv_20260914_164004_514282"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--phi-quantile", type=float, default=0.995)
    parser.add_argument("--laplacian-quantile", type=float, default=0.99)
    parser.add_argument(
        "--output", type=Path, default=EXPORTS_DIR / "cosmos_phih_comparison"
    )
    args = parser.parse_args(argv)
    for name in ("phi_quantile", "laplacian_quantile"):
        if not 0 < getattr(args, name) <= 1:
            parser.error(f"--{name.replace('_', '-')} must be in (0, 1]")
    return args


def load_volume(path: Path) -> np.ndarray:
    volume = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    if volume.ndim != 3 or not np.isfinite(volume).all():
        raise ValueError(f"{path} must contain one finite 3D volume")
    return volume


def nrmse(prediction: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> float:
    selected = mask.astype(bool)
    denominator = np.linalg.norm(reference[selected])
    return float(
        np.linalg.norm(prediction[selected] - reference[selected])
        / max(float(denominator), np.finfo(np.float32).eps)
    )


def interior_mask(mask: np.ndarray) -> np.ndarray:
    inside = mask.astype(bool).copy()
    for axis in range(3):
        inside &= np.roll(mask, -1, axis=axis) & np.roll(mask, 1, axis=axis)
    return inside


def nice_limit(volumes: list[np.ndarray], mask: np.ndarray, quantile: float) -> float:
    values = np.concatenate([np.abs(volume[mask]) for volume in volumes])
    raw = float(np.quantile(values, quantile))
    if not np.isfinite(raw) or raw <= 0:
        raise ValueError("Cannot determine a positive display limit")
    if raw >= 0.2:
        step = 0.05
    elif raw >= 0.02:
        step = 0.01
    elif raw >= 0.002:
        step = 0.001
    else:
        step = 10 ** (np.floor(np.log10(raw)) - 1)
    return float(step * np.ceil(raw / step))


def cropped_views(
    volume: np.ndarray, mask: np.ndarray
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    definitions = (
        ("Axial", 0, volume.shape[0] // 2),
        ("Coronal", 1, volume.shape[1] // 2),
        ("Sagittal", 2, volume.shape[2] // 2),
    )
    output = []
    for name, axis, index in definitions:
        image = np.rot90(np.take(volume, index, axis=axis))
        support = np.rot90(np.take(mask, index, axis=axis))
        if name == "Sagittal":
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
    return np.asarray(resized, dtype=np.float32), resized_support


def composite(
    volume: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, list[dict[str, int | str]]]:
    views = cropped_views(volume, mask)
    target_width = max(image.shape[1] for _, image, _ in views)
    images = []
    metadata = []
    offset = 0
    for name, image, support in views:
        image, support = resize_view(image, support, target_width)
        image = np.where(support, image, 0.0)
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


def create_figure(args: argparse.Namespace) -> dict:
    paths = {
        "WH-QSM": {
            "phi": args.run / "wh_tv_phi.nii.gz",
            "laplacian": args.run / "laplacian_wh_tv_phi.nii.gz",
        },
        "WH-Net": {
            "phi": args.run / "whnet_phi.nii.gz",
            "laplacian": args.run / "laplacian_whnet_phi.nii.gz",
        },
        "Ground Truth": {
            "phi": args.run / "phi_reference.nii.gz",
            "laplacian": args.run / "laplacian_phi_reference.nii.gz",
        },
    }
    mask = load_volume(args.run / "mask.nii.gz") > 0.5
    laplacian_mask = interior_mask(mask)
    methods = []
    for name, method_paths in paths.items():
        phi = load_volume(method_paths["phi"])
        laplacian = load_volume(method_paths["laplacian"])
        if phi.shape != mask.shape or laplacian.shape != mask.shape:
            raise ValueError(f"{name} volume shape differs from mask {mask.shape}")
        methods.append(
            {
                "name": name,
                "phi": np.where(mask, phi, 0.0),
                "laplacian": np.where(laplacian_mask, laplacian, 0.0),
            }
        )

    phi_reference = methods[-1]["phi"]
    for method in methods[:2]:
        method["phi_nrmse"] = nrmse(method["phi"], phi_reference, mask)

    phi_limit = nice_limit(
        [method["phi"] for method in methods], mask, args.phi_quantile
    )
    laplacian_limit = nice_limit(
        [method["laplacian"] for method in methods],
        laplacian_mask,
        args.laplacian_quantile,
    )
    quantities = (
        (
            "phi",
            "Harmonic field, φₕ",
            "φₕ (ppm)",
            matplotlib.colormaps["gray"].copy(),
            Normalize(-phi_limit, phi_limit, clip=True),
        ),
        (
            "laplacian",
            "Laplacian of φₕ",
            "∇²φₕ (ppm/voxel²)",
            matplotlib.colormaps["RdBu_r"].copy(),
            Normalize(-laplacian_limit, laplacian_limit, clip=True),
        ),
    )
    composites: dict[str, list[np.ndarray]] = {"phi": [], "laplacian": []}
    view_metadata = None
    for quantity, *_ in quantities:
        for method in methods:
            image, metadata = composite(method[quantity], mask)
            composites[quantity].append(image)
            view_metadata = metadata

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
    figure = plt.figure(figsize=(10.2, 17.2))
    grid = figure.add_gridspec(
        2,
        4,
        width_ratios=(1, 1, 1, 0.055),
        left=0.075,
        right=0.965,
        bottom=0.02,
        top=0.955,
        hspace=0.018,
        wspace=0.012,
    )
    for row, (quantity, row_label, colorbar_label, cmap, norm) in enumerate(quantities):
        for column, method in enumerate(methods):
            axis = figure.add_subplot(grid[row, column])
            axis.imshow(
                composites[quantity][column],
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
                rasterized=True,
            )
            axis.set_axis_off()
            if row == 0:
                axis.set_title(method["name"], color=TEXT_COLOR, pad=7)
            if quantity == "phi" and column < 2:
                axis.text(
                    0.035,
                    0.975,
                    f"NRMSE: {method['phi_nrmse']:.4f}",
                    transform=axis.transAxes,
                    ha="left",
                    va="top",
                    color=TEXT_COLOR,
                    fontsize=9,
                    fontweight="semibold",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": TEXT_COLOR,
                        "linewidth": 0.5,
                        "alpha": 0.82,
                        "pad": 2.0,
                    },
                )
        colorbar = figure.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            cax=figure.add_subplot(grid[row, 3]),
            orientation="vertical",
        )
        style_colorbar(colorbar, colorbar_label)
        figure.text(
            0.052,
            0.735 if row == 0 else 0.265,
            row_label,
            rotation=90,
            ha="center",
            va="center",
            color=TEXT_COLOR,
            fontsize=14,
            fontweight="semibold",
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for suffix in ("png", "pdf", "svg"):
        path = args.output.with_suffix(f".{suffix}")
        figure.savefig(path, dpi=400, bbox_inches="tight", pad_inches=0.02)
        outputs[suffix] = str(path.resolve())
    plt.close(figure)

    metadata = {
        "source_run": str(args.run.resolve()),
        "method_mapping": {
            "WH-QSM": "wh_tv_phi.nii.gz and laplacian_wh_tv_phi.nii.gz",
            "WH-Net": "whnet_phi.nii.gz and laplacian_whnet_phi.nii.gz",
            "Ground Truth": "phi_reference.nii.gz and laplacian_phi_reference.nii.gz",
        },
        "nrmse": {
            method["name"]: {"phi_h": method["phi_nrmse"]} for method in methods[:2]
        },
        "nrmse_mask": "brain mask; predictions only",
        "shape": list(mask.shape),
        "composite_views": view_metadata,
        "display": {
            "text_color": TEXT_COLOR,
            "phi_h_colormap": "gray",
            "laplacian_colormap": "RdBu_r",
            "phi_h_limit_ppm": phi_limit,
            "laplacian_limit_ppm_per_voxel_squared": laplacian_limit,
            "phi_h_quantile": args.phi_quantile,
            "laplacian_quantile": args.laplacian_quantile,
        },
        "outputs": outputs,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(create_figure(parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()

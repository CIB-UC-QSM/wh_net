"""Summarize local QSM susceptibility volumes and draw report figures.

Run from the repository root after generate_geometric_data.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from whnet.adaptive_wh_tv_validation import load_cosmos_case, load_sim2_case
from whnet.paths import PROJECT_ROOT, TRAINING_DATA_DIR

OUTPUT_DIR = PROJECT_ROOT / "docs" / "figures" / "synthetic_data"
EDGES = np.linspace(-5.0, 5.0, 20001)
EXAMPLE_INDICES = (651, 684, 999, 1301)
COLORS = {
    "Synthetic": "#2769a7",
    "Original": "#657080",
    "Combined": "#703f95",
    "COSMOS": "#18865f",
    "sim2": "#dc6844",
}


class Distribution:
    def __init__(self) -> None:
        self.histogram = np.zeros(len(EDGES) - 1, dtype=np.int64)
        self.voxels = 0
        self.volumes = 0
        self.total = 0.0
        self.total_squared = 0.0
        self.minimum = float("inf")
        self.maximum = float("-inf")
        self.positive = 0
        self.negative = 0
        self.near_zero = 0
        self.above_005 = 0
        self.above_010 = 0
        self.above_020 = 0
        self.occupancies: list[float] = []

    def add(self, chi: np.ndarray, mask: np.ndarray) -> None:
        if chi.shape != mask.shape or not mask.any():
            raise ValueError("Expected matching nonempty chi and mask")
        values = np.asarray(chi[mask], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite susceptibility")
        absolute = np.abs(values)
        self.histogram += np.histogram(values, EDGES)[0]
        self.voxels += values.size
        self.volumes += 1
        self.total += values.sum()
        self.total_squared += np.dot(values, values)
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))
        self.positive += int(np.count_nonzero(values > 0))
        self.negative += int(np.count_nonzero(values < 0))
        self.near_zero += int(np.count_nonzero(absolute < 0.005))
        self.above_005 += int(np.count_nonzero(absolute > 0.05))
        self.above_010 += int(np.count_nonzero(absolute > 0.10))
        self.above_020 += int(np.count_nonzero(absolute > 0.20))
        self.occupancies.append(float(mask.mean()))

    def quantile(self, fraction: float) -> float:
        cumulative = np.cumsum(self.histogram)
        index = int(np.searchsorted(cumulative, fraction * self.voxels))
        if index >= len(self.histogram):
            raise ValueError("Histogram does not cover susceptibility values")
        preceding = cumulative[index - 1] if index else 0
        proportion = (fraction * self.voxels - preceding) / max(
            int(self.histogram[index]), 1
        )
        return float(EDGES[index] + proportion * (EDGES[index + 1] - EDGES[index]))

    def summary(self) -> dict[str, float | int]:
        if int(self.histogram.sum()) != self.voxels:
            raise ValueError("Histogram range does not cover susceptibility values")
        mean = self.total / self.voxels
        return {
            "volumes": self.volumes,
            "masked_voxels": self.voxels,
            "mask_fraction_mean": float(np.mean(self.occupancies)),
            "mask_fraction_min": min(self.occupancies),
            "mask_fraction_max": max(self.occupancies),
            "chi_mean_ppm": mean,
            "chi_std_ppm": float(np.sqrt(self.total_squared / self.voxels - mean**2)),
            "chi_min_ppm": self.minimum,
            "chi_p01_ppm": self.quantile(0.01),
            "chi_p05_ppm": self.quantile(0.05),
            "chi_p25_ppm": self.quantile(0.25),
            "chi_median_ppm": self.quantile(0.5),
            "chi_p75_ppm": self.quantile(0.75),
            "chi_p95_ppm": self.quantile(0.95),
            "chi_p99_ppm": self.quantile(0.99),
            "chi_max_ppm": self.maximum,
            "positive_fraction": self.positive / self.voxels,
            "negative_fraction": self.negative / self.voxels,
            "abs_below_0005_fraction": self.near_zero / self.voxels,
            "abs_above_005_fraction": self.above_005 / self.voxels,
            "abs_above_010_fraction": self.above_010 / self.voxels,
            "abs_above_020_fraction": self.above_020 / self.voxels,
        }


def load_training_distribution(start: int, stop: int) -> Distribution:
    distribution = Distribution()
    for index in range(start, stop):
        image = TRAINING_DATA_DIR / f"image_{index:04d}.pt"
        mask_path = TRAINING_DATA_DIR / f"mask_{index:04d}.pt"
        chi = (
            torch.load(image, weights_only=True, map_location="cpu").squeeze(0).numpy()
        )
        mask = (
            torch.load(mask_path, weights_only=True, map_location="cpu")
            .squeeze(0)
            .numpy()
            .astype(bool)
        )
        distribution.add(chi, mask)
        if (index - start + 1) % 100 == 0:
            print(f"Scanned {index - start + 1}/{stop - start} volumes", flush=True)
    return distribution


def load_external_distribution(case) -> Distribution:
    distribution = Distribution()
    chi = (case.chi * case.factor).squeeze().numpy()
    mask = case.mask.squeeze().numpy().astype(bool)
    distribution.add(chi, mask)
    return distribution


def draw_examples() -> None:
    fig, axes = plt.subplots(len(EXAMPLE_INDICES), 4, figsize=(13.5, 11))
    image = None
    for row, index in enumerate(EXAMPLE_INDICES):
        chi = (
            torch.load(TRAINING_DATA_DIR / f"image_{index:04d}.pt", weights_only=True)
            .squeeze(0)
            .numpy()
        )
        mask = (
            torch.load(TRAINING_DATA_DIR / f"mask_{index:04d}.pt", weights_only=True)
            .squeeze(0)
            .numpy()
        )
        depth = int(np.argmax(mask.sum(axis=(1, 2))))
        height = int(np.argmax(mask.sum(axis=(0, 2))))
        width = int(np.argmax(mask.sum(axis=(0, 1))))
        axes[row, 0].imshow(mask[depth], cmap="gray", vmin=0, vmax=1)
        for column, slice_ in enumerate(
            (chi[depth], chi[:, height, :], chi[:, :, width]), start=1
        ):
            image = axes[row, column].imshow(
                slice_, cmap="RdBu_r", vmin=-0.15, vmax=0.15
            )
        axes[row, 0].set_ylabel(
            f"image_{index:04d}\n{mask.mean():.1%} mask", fontsize=10
        )
    for axis, title in zip(
        axes[0], ("Axial mask", "Axial χ", "Coronal χ", "Sagittal χ")
    ):
        axis.set_title(title)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    assert image is not None
    fig.suptitle("Examples of generated geometric QSM volumes", fontsize=15)
    fig.subplots_adjust(
        left=0.08, right=0.88, top=0.91, bottom=0.04, wspace=0.06, hspace=0.14
    )
    colorbar_axis = fig.add_axes([0.90, 0.22, 0.017, 0.55])
    fig.colorbar(image, cax=colorbar_axis).set_label("Susceptibility χ (ppm)")
    fig.savefig(OUTPUT_DIR / "examples.png", dpi=170)
    plt.close(fig)


def draw_references(cases: tuple) -> None:
    fig, axes = plt.subplots(len(cases), 3, figsize=(10.5, 6.4))
    image = None
    for row, case in enumerate(cases):
        chi = (case.chi * case.factor).squeeze().numpy()
        mask = case.mask.squeeze().numpy().astype(bool)
        depth = int(np.argmax(mask.sum(axis=(1, 2))))
        height = int(np.argmax(mask.sum(axis=(0, 2))))
        axes[row, 0].imshow(mask[depth], cmap="gray", vmin=0, vmax=1)
        image = axes[row, 1].imshow(chi[depth], cmap="RdBu_r", vmin=-0.15, vmax=0.15)
        axes[row, 2].imshow(chi[:, height, :], cmap="RdBu_r", vmin=-0.15, vmax=0.15)
        axes[row, 0].set_ylabel(case.name.upper() if case.name == "cosmos" else "sim2")
    for axis, title in zip(axes[0], ("Axial mask", "Axial χ", "Coronal χ")):
        axis.set_title(title)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    assert image is not None
    fig.suptitle("External reference volumes in physical ppm", fontsize=15)
    fig.subplots_adjust(
        left=0.08, right=0.87, top=0.87, bottom=0.04, wspace=0.06, hspace=0.11
    )
    colorbar_axis = fig.add_axes([0.90, 0.2, 0.02, 0.6])
    fig.colorbar(image, cax=colorbar_axis).set_label("Susceptibility χ (ppm)")
    fig.savefig(OUTPUT_DIR / "references.png", dpi=170)
    plt.close(fig)


def draw_distribution(distributions: dict[str, Distribution]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    centers = (EDGES[:-1] + EDGES[1:]) / 2
    width = EDGES[1] - EDGES[0]
    for name, distribution in distributions.items():
        histogram = distribution.histogram
        density = histogram / (distribution.voxels * width)
        # Smooth only the display; summaries use unsmoothed full-resolution bins.
        window = np.ones(21) / 21
        density = np.convolve(density, window, mode="same")
        style = "--" if name == "Combined" else "-"
        axes[0].plot(centers, density, label=name, color=COLORS[name], linestyle=style)
        thresholds = np.linspace(0, 0.4, 401)
        absolute_tail = (
            np.array(
                [
                    histogram[np.abs(centers) >= threshold].sum()
                    for threshold in thresholds
                ]
            )
            / distribution.voxels
        )
        axes[1].plot(
            thresholds, absolute_tail, label=name, color=COLORS[name], linestyle=style
        )
    axes[0].set(xlim=(-0.2, 0.2), xlabel="χ (ppm)", ylabel="Masked-voxel density")
    axes[1].set(
        xlim=(0, 0.4),
        ylim=(1e-5, 1),
        yscale="log",
        xlabel="Absolute χ threshold (ppm)",
        ylabel="Fraction of masked voxels above threshold",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, ncol=2)
    fig.suptitle("Masked susceptibility distributions in physical ppm", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "distributions.png", dpi=170)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    original = load_training_distribution(0, 651)
    synthetic = load_training_distribution(651, 1302)
    combined = Distribution()
    for attribute in (
        "histogram",
        "voxels",
        "volumes",
        "total",
        "total_squared",
        "positive",
        "negative",
        "near_zero",
        "above_005",
        "above_010",
        "above_020",
    ):
        setattr(
            combined,
            attribute,
            getattr(original, attribute) + getattr(synthetic, attribute),
        )
    combined.minimum = min(original.minimum, synthetic.minimum)
    combined.maximum = max(original.maximum, synthetic.maximum)
    combined.occupancies = original.occupancies + synthetic.occupancies
    cosmos_case = load_cosmos_case()
    sim2_case = load_sim2_case()
    distributions = {
        "Synthetic": synthetic,
        "Original": original,
        "Combined": combined,
        "COSMOS": load_external_distribution(cosmos_case),
        "sim2": load_external_distribution(sim2_case),
    }
    summary = {
        "method": {
            "units": "ppm",
            "sample_selection": "all voxels inside each saved or reference mask",
            "training_scale": "saved chi before random 0.5–3 training augmentation",
            "quantile_resolution_ppm": float(EDGES[1] - EDGES[0]),
            "cosmos_source": str(
                Path(cosmos_case.metadata["simulation"]).relative_to(PROJECT_ROOT)
            ),
            "cosmos_factor": cosmos_case.factor,
            "sim2_source": str(
                Path(sim2_case.metadata["chi_reference"]).relative_to(PROJECT_ROOT)
            ),
            "sim2_factor": sim2_case.factor,
        },
        "distributions": {name: data.summary() for name, data in distributions.items()},
    }
    (OUTPUT_DIR / "statistics.json").write_text(json.dumps(summary, indent=2) + "\n")
    draw_examples()
    draw_references((cosmos_case, sim2_case))
    draw_distribution(distributions)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

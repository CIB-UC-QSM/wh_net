"""Export the WHNet conference figure as SVG, PDF and a 3840 x 2160 PNG.

Usage: uv run python scripts/draw_conference_architecture.py

This is a conceptual view of EfficientADMMUnrolledNet. The state bus includes
the current estimates, dual variables and conditioning statistics. The learned
branches have independent weights, reused at each iteration. Normalization and
rescaling are implicit. The detailed companion figure documents the equations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
INK = "#183044"
MUTED = "#637787"
BLUE = "#237EA0"
PURPLE = "#8056AF"
STATE = "#7C8D9C"
PALE_BLUE = "#ECF6FA"
PALE_PURPLE = "#F4EFFA"


def text(
    ax: Axes,
    x: float,
    y: float,
    value: str,
    size: float = 13,
    color: str = INK,
    weight: str = "normal",
    align: str = "center",
    **kwargs: object,
) -> None:
    ax.text(
        x,
        y,
        value,
        fontsize=size,
        color=color,
        weight=weight,
        ha=align,
        va="center",
        **kwargs,
    )


def card(
    ax: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    fill: str,
    edge: str = "none",
    radius: float = 0.14,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            linewidth=1.1,
            facecolor=fill,
            edgecolor=edge,
            zorder=2,
        )
    )


def route(
    ax: Axes,
    points: list[tuple[float, float]],
    color: str = STATE,
    arrow: bool = True,
    dashed: bool = False,
) -> None:
    """Draw an orthogonal connection, with an optional final arrowhead."""
    xs, ys = zip(*points)
    ax.plot(
        xs,
        ys,
        color=color,
        linewidth=1.45,
        linestyle=(0, (4, 3)) if dashed else "-",
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=1,
    )
    if arrow:
        ax.add_patch(
            FancyArrowPatch(
                points[-2],
                points[-1],
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.45,
                color=color,
                shrinkA=0,
                shrinkB=1.5,
                zorder=3,
            )
        )


def unet_icon(ax: Axes, x: float, y: float) -> None:
    """A schematic encoder/decoder glyph, not a layer-by-layer network plot."""
    points = [
        (x, y + 0.49),
        (x + 0.2, y + 0.27),
        (x + 0.4, y + 0.05),
        (x + 0.6, y + 0.27),
        (x + 0.8, y + 0.49),
    ]
    ax.plot(*zip(*points), color=PURPLE, linewidth=1.4, zorder=3)
    for left, right in [(points[0], points[-1]), (points[1], points[-2])]:
        ax.plot(
            [left[0], right[0]],
            [left[1], right[1]],
            color=PURPLE,
            alpha=0.45,
            linewidth=1,
            linestyle=(0, (2, 2)),
            zorder=3,
        )
    for px, py in points:
        card(ax, px - 0.045, py - 0.09, 0.09, 0.18, PURPLE, radius=0.025)


def draw_conference_architecture(output_dir: Path) -> tuple[Path, Path, Path]:
    """Render the English conference diagram without model or dataset access."""
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "svg.fonttype": "none",
            "svg.hashsalt": "whnet-conference-architecture",
            "pdf.fonttype": 42,
        }
    ):
        fig, ax = plt.subplots(figsize=(16, 9))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig.patch.set_facecolor("white")
        ax.set(xlim=(0, 16), ylim=(0, 9))
        ax.axis("off")

        text(ax, 0.6, 8.35, "WHNet", 30, weight="bold", align="left")
        text(
            ax,
            0.62,
            7.78,
            "Joint susceptibility and background field estimation",
            19,
            align="left",
        )
        text(ax, 14.0, 8.18, r"$\varphi\approx A\chi+\varphi_h$", 23, BLUE)
        text(ax, 14.0, 7.72, "Dipole forward model", 12, MUTED)

        # Light container for one shared-weight iteration, expanded in place.
        card(ax, 3.13, 0.98, 9.52, 6.14, "#F8FAFC", "#E1E8EE", 0.2)
        ax.patches[-1].set_zorder(0)
        text(ax, 3.48, 6.78, "Unrolled ADMM", 18, weight="bold", align="left")
        card(ax, 10.53, 6.53, 1.77, 0.49, "#E7EDF3", radius=0.15)
        text(ax, 11.415, 6.775, "K iterations", 12, weight="bold")

        # Measured field enters data consistency directly; the state bus is
        # distinct, so no arrow suggests that the measured field feeds a U-Net.
        card(ax, 0.6, 4.94, 2.04, 1.15, "#EDF2F6")
        text(ax, 1.62, 5.72, "Measured field", 14, weight="bold")
        text(ax, 1.62, 5.25, r"$\varphi$", 25)
        route(ax, [(1.62, 6.09), (1.62, 6.32), (5.45, 6.32), (5.45, 6.0)], BLUE)
        text(ax, 0.65, 4.28, "Mask · Dipole kernel", 12, MUTED, align="left")
        text(ax, 0.65, 3.93, "Data weights", 12, MUTED, align="left")
        text(ax, 0.65, 3.5, "Known inputs", 11, MUTED, align="left")

        # A single state fans out to the three components; neural branches are
        # independent. The residual conditioning is summarized in the footer.
        route(ax, [(3.75, 1.4), (3.75, 5.5)], arrow=False)
        for y in (5.5, 4.0, 2.6):
            route(ax, [(3.75, y), (4.3, y)])
            ax.plot(3.75, y, "o", markersize=3.3, color=STATE, zorder=3)
        text(ax, 3.43, 3.82, "Current state", 11.5, MUTED, rotation=90)

        card(ax, 4.3, 5.0, 3.8, 1.0, PALE_BLUE)
        text(ax, 6.2, 5.66, "Data consistency", 16, BLUE, "bold")
        text(ax, 6.2, 5.28, "Match the measured field", 12, MUTED)

        for y, title, symbol in [
            (3.35, "Susceptibility prior", r"$P_\chi$"),
            (1.95, "Background field prior", r"$P_h$"),
        ]:
            card(ax, 4.3, y, 3.8, 1.3, PALE_PURPLE)
            unet_icon(ax, 4.62, y + 0.33)
            text(ax, 5.72, y + 0.86, title, 12.8, PURPLE, "bold", "left")
            text(ax, 5.72, y + 0.43, f"3D U-Net   {symbol}", 12, MUTED, align="left")

        for y, color, name in [
            (5.5, BLUE, "y"),
            (4.0, PURPLE, "u"),
            (2.6, PURPLE, "v"),
        ]:
            route(ax, [(8.1, y), (9.15, y)], color)
            text(ax, 8.61, y + 0.23, f"${name}$", 12, color)

        # All current state / known input dependencies of these grouped
        # algebraic updates are implicit within the ADMM iteration container.
        card(ax, 9.15, 1.95, 2.96, 4.05, PALE_BLUE, "#CDE4EE")
        text(ax, 10.63, 5.58, "Physics-based", 16, BLUE, "bold")
        text(ax, 10.63, 5.19, "updates", 16, BLUE, "bold")
        for y, value in [
            (4.51, r"Update $\chi$ · apply mask"),
            (3.62, "Recompute dipole field"),
            (2.73, r"Update $\varphi_h$"),
        ]:
            card(ax, 9.38, y - 0.26, 2.5, 0.52, "white", radius=0.08)
            text(ax, 10.63, y, value, 11.6, INK)
        route(ax, [(10.63, 4.25), (10.63, 3.91)], BLUE)
        route(ax, [(10.63, 3.36), (10.63, 3.02)], BLUE)

        # Feedback represents carrying estimates, duals and statistics to k+1.
        route(ax, [(10.63, 1.95), (10.63, 1.4), (3.75, 1.4), (3.75, 1.79)], dashed=True)
        text(
            ax,
            7.0,
            1.4,
            "State & dual update",
            12,
            MUTED,
            bbox={"facecolor": "#F8FAFC", "edgecolor": "none", "pad": 5},
        )

        # Final estimates leave the repeated module together and then split.
        route(ax, [(12.11, 4.0), (13.1, 4.0)], arrow=False)
        route(ax, [(13.1, 4.0), (13.1, 5.38), (13.58, 5.38)])
        route(ax, [(13.1, 4.0), (13.1, 3.15), (13.58, 3.15)])
        ax.plot(13.1, 4.0, "o", markersize=3.3, color=STATE, zorder=3)
        text(ax, 14.46, 6.32, "After K iterations", 12, MUTED)
        for y, title, symbol in [
            (4.77, "Susceptibility", r"$\chi$"),
            (2.54, "Background field", r"$\varphi_h$"),
        ]:
            card(ax, 13.58, y, 1.83, 1.22, "#EDF2F6")
            text(ax, 14.495, y + 0.88, title, 12, weight="bold")
            text(ax, 14.495, y + 0.38, symbol, 25)

        text(
            ax, 3.3, 0.52, "Weights shared across iterations", 12, PURPLE, align="left"
        )
        text(
            ax,
            8.22,
            0.52,
            "Iteration & residual conditioning",
            12,
            PURPLE,
            align="left",
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        paths = tuple(
            output_dir / f"model_architecture_conference.{ext}"
            for ext in ("svg", "pdf", "png")
        )
        fig.savefig(paths[0], metadata={"Date": None})
        fig.savefig(paths[1], metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(paths[2], dpi=240)
        plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "figures")
    args = parser.parse_args()
    for path in draw_conference_architecture(args.output_dir):
        print(path)


if __name__ == "__main__":
    main()

"""Draw the current WHNet architecture without loading data or the model.

Run from the repository root:
    uv run python scripts/draw_model_architecture.py

The SVG and PDF are written to docs/figures by default. Arrows in panel B
denote execution order; each equation specifies its actual data dependencies.
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
INK = "#172D40"
MUTED = "#526879"
COLORS = {
    "physics": ("#E8F3F8", "#287DA3"),
    "learned": ("#F0EAF8", "#8055A5"),
    "algebra": ("#EDF5EF", "#508565"),
    "state": ("#F1F4F7", "#738597"),
}


def label(
    ax: Axes,
    x: float,
    y: float,
    text: str,
    size: float = 12,
    color: str = INK,
    weight: str = "normal",
    ha: str = "center",
) -> None:
    ax.text(x, y, text, fontsize=size, color=color, weight=weight, ha=ha, va="center")


def box(
    ax: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    lines: list[str],
    kind: str = "algebra",
    size: float = 12,
) -> None:
    fill, edge = COLORS[kind]
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.025,rounding_size=0.12",
            linewidth=1.3,
            facecolor=fill,
            edgecolor=edge,
        )
    )
    label(ax, x + width / 2, y + height - 0.32, title, 12.5, edge, "bold")
    available = height - 0.66
    for i, line in enumerate(lines):
        position = y + available * (1 - (i + 0.5) / len(lines))
        label(ax, x + width / 2, position, line, size)


def arrow(ax: Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.5,
            color=MUTED,
            shrinkA=5,
            shrinkB=5,
        )
    )


def draw_architecture(output_dir: Path) -> tuple[Path, Path]:
    """Export a two-panel architecture figure as editable SVG and vector PDF."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "svg.fonttype": "none",
            "svg.hashsalt": "whnet-architecture",
            "pdf.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(22, 14))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.patch.set_facecolor("white")
    ax.set(xlim=(0, 22), ylim=(0, 14))
    ax.axis("off")

    label(
        ax, 0.65, 13.5, "WHNet · arquitectura del modelo", 25, weight="bold", ha="left"
    )
    label(
        ax,
        0.65,
        13.02,
        "EfficientADMMUnrolledNet · ADMM aprendido con física dipolar",
        13,
        MUTED,
        ha="left",
    )
    label(ax, 0.65, 12.38, "A   FLUJO GENERAL", 13, weight="bold", ha="left")

    overview = [
        (
            0.65,
            2.7,
            "Entradas",
            [r"$\varphi,\ M,\ D,\ W$", r"$W=M$ si se omite"],
            "state",
        ),
        (
            3.8,
            3.0,
            "Normalización",
            [r"$s=\max(\mathrm{RMS}_M(\varphi),10^{-3})$", r"$p=\varphi/s$"],
            "algebra",
        ),
        (
            7.25,
            2.7,
            "Inicialización",
            [r"$\chi^0=h^0=0$", r"$\eta_y^0=\eta_u^0=\eta_v^0=0$"],
            "state",
        ),
        (
            10.4,
            4.0,
            "K iteraciones ADMM",
            [
                r"$\mathcal{B}_0\ \longrightarrow\ \cdots\ \longrightarrow\ \mathcal{B}_{K-1}$",
                "Dos U-Nets 3D por iteración",
            ],
            "learned",
        ),
        (
            14.85,
            2.7,
            "Restaurar escala",
            [r"$\widehat{\chi}=s\chi^K$", r"$\widehat{\varphi}_h=s h^K$"],
            "algebra",
        ),
        (
            18.0,
            3.35,
            "Salidas",
            [
                r"Susceptibilidad $\widehat{\chi}$",
                r"Campo de fondo $\widehat{\varphi}_h$",
            ],
            "state",
        ),
    ]
    for x, width, title, lines, kind in overview:
        box(ax, x, 10.35, width, 1.65, title, lines, kind, size=11.5)
    for left, right in zip(overview, overview[1:]):
        arrow(ax, (left[0] + left[1], 11.17), (right[0], 11.17))
    label(
        ax,
        0.65,
        9.93,
        r"$p,M,D,W$ disponibles en cada iteración · K configurable · volúmenes [N, C, D, H, W]",
        11.5,
        MUTED,
        ha="left",
    )
    ax.plot([0.65, 21.35], [9.55, 9.55], color="#D9E1E7", linewidth=1)

    label(
        ax,
        0.65,
        9.13,
        r"B   UNA ITERACIÓN $\mathcal{B}_k$",
        13,
        weight="bold",
        ha="left",
    )
    label(
        ax,
        0.65,
        8.66,
        r"Estado entrante: $\chi^k,h^k,\eta_y^k,\eta_u^k,\eta_v^k$ y $\delta_k$ · $h\equiv\varphi_h$ en escala normalizada",
        12,
        MUTED,
        ha="left",
    )
    label(
        ax,
        21.35,
        9.13,
        "Flechas: orden de cálculo; las ecuaciones indican las dependencias.",
        10.5,
        MUTED,
        ha="right",
    )

    xs = [0.65, 6.02, 11.39, 16.76]
    width, top_y, height, bottom_y = 4.59, 5.9, 2.25, 2.87
    box(
        ax,
        xs[0],
        top_y,
        width,
        height,
        "1 · Física y residuo",
        [r"$f^k=A(\chi^k)$", r"$r_k=\mathrm{RMS}_M(f^k+h^k-p)$"],
        "physics",
        13,
    )
    box(
        ax,
        xs[1],
        top_y,
        width,
        height,
        "2 · Consistencia con los datos",
        [r"$y=\frac{Wp+f^k+h^k+\eta_y^k}{W+1}$"],
        size=17,
    )
    box(
        ax,
        xs[2],
        top_y,
        width,
        height,
        "3 · Redes proximales aprendidas",
        [
            r"$u=P_\chi(\chi^k+\eta_u^k;\ k,\delta_k)$",
            r"$v=P_h(h^k+\eta_v^k;\ k,r_k)$",
            "chinet / phinet · dos U-Nets 3D",
        ],
        "learned",
        12.5,
    )
    box(
        ax,
        xs[3],
        top_y,
        width,
        height,
        "4 · Actualizar susceptibilidad",
        [
            r"$q=\frac{D\,\mathcal{F}(y-\eta_y^k-h^k)+\mathcal{F}(u-\eta_u^k)}{D^2+1+\varepsilon}$",
            r"$\chi^{k+1}=M\,\mathcal{F}^{-1}(q)$",
        ],
        size=12.5,
    )

    box(
        ax,
        xs[3],
        bottom_y,
        width,
        height,
        "5 · Recalcular campo dipolar",
        [
            r"$f^{k+1}=A(\chi^{k+1})$",
            "Se usa χ después de la máscara",
            "Física en FP32 · FFT real",
        ],
        "physics",
        12,
    )
    box(
        ax,
        xs[2],
        bottom_y,
        width,
        height,
        "6 · Actualizar campo de fondo",
        [r"$h^{k+1}=\frac{1}{2}(y-\eta_y^k-f^{k+1}+v-\eta_v^k)$"],
        size=12.5,
    )
    box(
        ax,
        xs[1],
        bottom_y,
        width,
        height,
        "7 · Actualizar variables duales",
        [
            r"$\eta_y^{k+1}=\eta_y^k+f^{k+1}+h^{k+1}-y$",
            r"$\eta_u^{k+1}=\eta_u^k+\chi^{k+1}-u$",
            r"$\eta_v^{k+1}=\eta_v^k+h^{k+1}-v$",
        ],
        size=12.5,
    )
    box(
        ax,
        xs[0],
        bottom_y,
        width,
        height,
        "Estado para la siguiente iteración",
        [
            r"$\chi^{k+1},h^{k+1},\eta_y^{k+1},\eta_u^{k+1},\eta_v^{k+1}$",
            r"$\delta_{k+1}=\mathrm{RMS}_M(\chi^{k+1}-\chi^k)$",
            r"Inicio: $\delta_0=0$",
        ],
        "state",
        12,
    )
    for left, right in zip(xs, xs[1:]):
        arrow(ax, (left + width, top_y + height / 2), (right, top_y + height / 2))
        arrow(ax, (right, bottom_y + height / 2), (left + width, bottom_y + height / 2))
    arrow(ax, (xs[3] + width / 2, top_y), (xs[3] + width / 2, bottom_y + height))

    label(
        ax,
        0.65,
        2.35,
        r"Redes: $P_\chi$ y $P_h$ tienen parámetros independientes; cada una comparte sus pesos entre las K iteraciones.",
        12,
        INK,
        ha="left",
    )
    label(
        ax,
        0.65,
        1.93,
        r"Cada proximal aplica una corrección residual $P(x;k,a)=x+\alpha(k,a)\,U(x)$; una MLP controla la ganancia $\alpha$.",
        12,
        INK,
        ha="left",
    )
    label(
        ax,
        0.65,
        1.42,
        r"$A(x)=\mathcal{F}^{-1}[D\,\mathcal{F}(x)]$     ·     $\mathrm{RMS}_M(x)=\sqrt{\frac{\sum Mx^2}{\max(\sum M,1)}+\varepsilon^2}$     ·     $\mathcal{F}$: FFT real 3D",
        12,
        MUTED,
        ha="left",
    )

    for x, kind, title in [
        (0.65, "physics", "Física dipolar"),
        (4.45, "learned", "Redes aprendidas"),
        (8.85, "algebra", "Actualizaciones algebraicas"),
        (14.4, "state", "Datos / estado"),
    ]:
        fill, edge = COLORS[kind]
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.62),
                0.3,
                0.24,
                boxstyle="round,pad=0.02",
                facecolor=fill,
                edgecolor=edge,
            )
        )
        label(ax, x + 0.45, 0.74, title, 10.5, MUTED, ha="left")
    label(
        ax,
        21.35,
        0.22,
        "Fuente: src/whnet/model.py · configuración actual de entrenamiento · normalización activada",
        9,
        MUTED,
        ha="right",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / "model_architecture.svg"
    pdf_path = output_dir / "model_architecture.pdf"
    fig.savefig(svg_path, metadata={"Date": None})
    fig.savefig(pdf_path, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    return svg_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "figures")
    args = parser.parse_args()
    for path in draw_architecture(args.output_dir):
        print(path)


if __name__ == "__main__":
    main()

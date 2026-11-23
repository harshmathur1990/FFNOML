"""Generate a Matplotlib schematic of the FFNO3D architecture (the final model): the height operator (the vertical
branch a spectral convolution along each column's heights, HeightSpectralConv), rank-8 frequency mixing, side paths
starting at 0.5, and the level-fraction head, whose output layer turns the head's L + 1 channels per point (one per
level, one for the element's total) into populations in m^-3.

Run:
    py generate_ffno_schematic_matplotlib.py

Outputs:
    ffno_schematic_matplotlib.png
    ffno_schematic_matplotlib.pdf
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.patches import (
    FancyArrowPatch,
    FancyBboxPatch,
    Circle,
    Polygon,
    Rectangle,
)


BLUE = "#0053b5"
BLUE_EDGE = "#397dd5"
BLUE_FILL = "#eef6ff"
GREEN = "#147a23"
GREEN_EDGE = "#67aa67"
GREEN_FILL = "#f1fff0"
ORANGE = "#c17100"
ORANGE_EDGE = "#f0a51a"
ORANGE_FILL = "#fff7df"
PURPLE = "#8a6bc9"
INK = "#111111"
GRID = "#333333"


def rounded_box(
    ax,
    xy,
    wh,
    text="",
    fc="white",
    ec=INK,
    lw=0.9,
    radius=0.07,
    fontsize=7,
    color=INK,
    weight="normal",
    ls="-",
    pad=0.035,
    zorder=2,
):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad={pad},rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=ls,
        zorder=zorder,
    )
    ax.add_patch(patch)
    if text:
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color,
            weight=weight,
            linespacing=1.25,
            zorder=zorder + 1,
        )
    return patch


def text(ax, x, y, s, size=7, weight="normal", color=INK, ha="center", va="center"):
    ax.text(
        x,
        y,
        s,
        fontsize=size,
        weight=weight,
        color=color,
        ha=ha,
        va=va,
        linespacing=1.25,
    )


def arrow(ax, start, end, color=INK, lw=0.9, ms=8, rad=0.0, style="->", ls="-"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        linestyle=ls,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=1,
        shrinkB=1,
        zorder=6,
    )
    ax.add_patch(patch)
    return patch


def op_box(ax, x, y, w, h, title, body="", theme="blue", fs=5.8):
    if theme == "blue":
        fc, ec, col = "#f7fbff", BLUE_EDGE, BLUE
    elif theme == "green":
        fc, ec, col = "#f7fff6", GREEN_EDGE, GREEN
    elif theme == "orange":
        fc, ec, col = "#fffaf0", ORANGE_EDGE, ORANGE
    else:
        fc, ec, col = "white", "#888888", INK
    label = title if not body else f"{title}\n{body}"
    return rounded_box(
        ax,
        (x, y),
        (w, h),
        label,
        fc=fc,
        ec=ec,
        lw=0.75,
        radius=0.05,
        fontsize=fs,
        weight="bold" if not body else "normal",
        color=INK,
    )


def circled_symbol(ax, x, y, symbol, r=0.075, size=7):
    ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=INK, lw=0.8, zorder=8))
    text(ax, x, y - 0.002, symbol, size=size, weight="bold")


def cube(ax, x, y, s=0.55, depth=0.16, cmap="turbo", kind="heat", label=None):
    """Draw a pseudo-3D gridded cube."""
    dx, dy = depth * s, depth * s
    n = 6
    if kind == "latent":
        colors = [[0.55 for _ in range(n)] for _ in range(n)]
        cm = plt.get_cmap("Purples")
    elif kind == "green":
        colors = [[0.45 for _ in range(n)] for _ in range(n)]
        cm = plt.get_cmap("Greens")
    elif kind == "blue":
        colors = [[0.55 for _ in range(n)] for _ in range(n)]
        cm = plt.get_cmap("Blues")
    else:
        colors = [[i / (n - 1) for _ in range(n)] for i in range(n)]
        cm = plt.get_cmap(cmap)

    for i in range(n):
        for j in range(n):
            ax.add_patch(
                Rectangle(
                    (x + j * s / n, y + i * s / n),
                    s / n,
                    s / n,
                    facecolor=cm(colors[i][j]),
                    edgecolor="#222222",
                    lw=0.35,
                    zorder=4,
                )
            )

    top = Polygon(
        [(x, y + s), (x + dx, y + s + dy), (x + s + dx, y + s + dy), (x + s, y + s)],
        closed=True,
        facecolor=cm(0.78 if kind != "heat" else 0.88),
        edgecolor="#222222",
        lw=0.55,
        zorder=3,
        alpha=0.95,
    )
    side = Polygon(
        [(x + s, y), (x + s + dx, y + dy), (x + s + dx, y + s + dy), (x + s, y + s)],
        closed=True,
        facecolor=cm(0.36 if kind != "heat" else 0.68),
        edgecolor="#222222",
        lw=0.55,
        zorder=3,
        alpha=0.95,
    )
    ax.add_patch(top)
    ax.add_patch(side)

    for k in range(1, n):
        ax.plot([x + k * s / n, x + k * s / n], [y, y + s], color=GRID, lw=0.28, zorder=5)
        ax.plot([x, x + s], [y + k * s / n, y + k * s / n], color=GRID, lw=0.28, zorder=5)
    if label:
        text(ax, x + s / 2, y - 0.22, label, size=6.2)


def tiny_column(ax, x, y, h=0.85, color="#8bc37d", n=5):
    ax.plot([x, x], [y, y + h], color="#555555", lw=1.0)
    for i in range(n):
        ax.add_patch(
            Circle((x, y + i * h / (n - 1)), 0.045, facecolor=color, edgecolor="#444444", lw=0.5, zorder=7)
        )


def mini_heatmap(ax, x, y, w=0.45, h=0.55):
    z = [
        [
            npish_sin(2 * (-2 + 4 * c / 21)) * npish_cos(2 * (-2 + 4 * r / 21))
            + 0.10 * npish_sin(7 * r + 3 * c)
            for c in range(22)
        ]
        for r in range(22)
    ]
    ax.imshow(z, extent=(x, x + w, y, y + h), origin="lower", cmap="viridis", zorder=3, aspect="auto")
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="#333333", lw=0.5, zorder=4))


def npish_sin(value):
    # Short Taylor-friendly wrapper keeps the script NumPy-free.
    import math

    return math.sin(value)


def npish_cos(value):
    import math

    return math.cos(value)


def draw_axes_near_cube(ax, x, y, s=0.55):
    arrow(ax, (x - 0.05, y - 0.03), (x + s * 0.75, y - 0.18), lw=0.75, ms=6)
    arrow(ax, (x - 0.05, y - 0.03), (x - 0.23, y + 0.05), lw=0.75, ms=6)
    arrow(ax, (x - 0.05, y - 0.03), (x - 0.05, y + s * 1.05), lw=0.75, ms=6)
    text(ax, x + s * 0.7, y - 0.27, r"$x$", size=7, weight="bold")
    text(ax, x - 0.30, y - 0.02, r"$y$", size=7, weight="bold")
    text(ax, x - 0.08, y + s * 1.13, r"$z$", size=7, weight="bold")


def draw_top_architecture(ax):
    text(ax, 0.17, 9.72, "(a) Overall Architecture", size=11, weight="bold", ha="left")
    text(ax, 0.78, 9.25, "Input: 3D Atmospheric State\n(6 transformed channels)", size=7, weight="bold")
    cube(ax, 0.42, 7.62, s=0.80, depth=0.18, kind="heat")
    draw_axes_near_cube(ax, 0.42, 7.62, 0.80)
    text(ax, 0.20, 7.10, "Fields:\n$\\log_{10}T,\\,v_x,\\,v_y,\\,v_z,\\,\\log_{10}n_e,\\,\\log_{10}\\rho$", size=6.5, ha="left")
    text(ax, 0.20, 6.55, r"Shape: $(N_z,N_y,N_x,6)$", size=7, ha="left")

    rounded_box(ax, (1.86, 7.80), (0.85, 1.02), "Lift Layer\n$(P)$\n\n1x1x1\nConv3D\n+\nGN$_w$\n+\nGELU", fc="#f3f8ff", ec="#2b5c9a", fontsize=6.6, weight="bold")
    cube(ax, 3.10, 7.85, s=0.55, depth=0.16, kind="latent")
    text(ax, 3.42, 8.90, "Latent Cube\n$v_0$", size=7, weight="bold")
    text(ax, 3.42, 7.32, r"$(N_z,N_y,N_x,C)$", size=7)
    arrow(ax, (1.30, 8.27), (1.86, 8.29))
    arrow(ax, (2.71, 8.30), (3.10, 8.28))

    rounded_box(ax, (3.96, 6.22), (8.28, 3.34), "", fc="white", ec=INK, lw=1.0, radius=0.06)
    rounded_box(ax, (3.99, 6.02), (7.70, 0.20), "", fc="none", ec=INK, lw=0.8, radius=0.02)
    text(ax, 7.98, 9.77, r"FFNO Block (repeated $N=4$ times)", size=11, weight="bold")
    arrow(ax, (3.72, 8.28), (3.96, 8.28))

    rounded_box(ax, (4.12, 8.00), (5.82, 1.22), "", fc=BLUE_FILL, ec="#8bbaf0", lw=0.75, radius=0.06)
    text(ax, 7.02, 9.14, "Spectral Branch (horizontal: x-y)", size=7.0, color=BLUE, weight="bold")
    specs = [
        ("Input Gate", "Conv3D -> GELU\n-> Conv3D -> sigmoid\nthen multiply input"),
        ("FFT2", "(x, y)\nper depth\n(rfft2)"),
        ("Metric-aware\nFrequency\nEmbedding", r"$k_x,k_y,$" + "\n" + r"$\sqrt{k_x^2+k_y^2},$" + "\n" + r"$\mathrm{sign}(k_y)$"),
        ("Frequency\nMLPs", "complex\nweights\n+ rank-8"),
        ("Complex\nSpectral\nMultiplication", r"$\odot$"),
        ("iFFT2", "(x, y)"),
        ("Post Spectral\nMLP", "1x1x1 Conv3D\n-> GELU ->\n1x1x1 Conv3D"),
    ]
    x = 4.28
    widths = [0.55, 0.48, 0.70, 0.56, 0.66, 0.48, 0.76]
    for i, ((title, body), w) in enumerate(zip(specs, widths)):
        op_box(ax, x, 8.28, w, 0.68, title, body, "blue", fs=5.0)
        if i < len(specs) - 1:
            arrow(ax, (x + w + 0.03, 8.62), (x + w + 0.18, 8.62), lw=0.7, ms=6)
        x += w + 0.25
    text(ax, 9.95, 8.78, r"$v^{spec}$", size=7, color=BLUE, weight="bold", ha="left")

    rounded_box(ax, (4.12, 6.48), (5.82, 1.26), "", fc=GREEN_FILL, ec="#acd7a7", lw=0.75, radius=0.06)
    text(ax, 7.02, 7.62, "Vertical Branch (along z, per column)", size=7.0, color=GREEN, weight="bold")
    verts = [
        ("Input Gate", "as in x-y\n+ height\nfeature $z$"),
        ("Height\nTransform", "$X_m=\\int v\\,e^{-ik_m z}dz$\non the column's\nown heights"),
        ("Physical\nWavenumber", "$k_m=2\\pi m/L$\n$\\lambda\\geq 40$ km"),
        ("Frequency\nMLPs", "complex\nweights\n+ rank-8"),
        ("Column\nCutoff", "taper to 0\nat $\\pi/\\Delta z$"),
        ("Inverse\nSeries", "at $z_j$"),
        ("Post\nMLP", "1x1x1 Conv3D\n-> GELU ->\n1x1x1 Conv3D"),
    ]
    x = 4.28
    widths = [0.55, 0.78, 0.60, 0.56, 0.52, 0.42, 0.62]
    for i, ((title, body), w) in enumerate(zip(verts, widths)):
        op_box(ax, x, 6.78, w, 0.68, title, body, "green", fs=4.8)
        if i < len(verts) - 1:
            arrow(ax, (x + w + 0.03, 7.12), (x + w + 0.18, 7.12), lw=0.7, ms=6)
        x += w + 0.22
    text(ax, 7.02, 6.60, "no part counts grid points: any vertical grid and number of levels", size=4.9)
    text(ax, 9.95, 6.95, r"$v^{vert}$", size=7, color=GREEN, weight="bold", ha="left")

    rounded_box(ax, (10.38, 6.72), (1.55, 2.30), "", fc=ORANGE_FILL, ec=ORANGE_EDGE, lw=0.75, radius=0.06)
    text(ax, 11.15, 8.84, "Gated Fusion", size=7, color=ORANGE, weight="bold")
    op_box(ax, 10.51, 8.07, 1.29, 0.55, "Branch Gates", "$[v,v^{spec},v^{vert}]$\nConv3D -> GELU\n-> Conv3D -> $\\sigma$", "orange", fs=4.8)
    op_box(ax, 10.56, 7.60, 0.48, 0.25, r"$g_{spec}$", "", "orange", fs=6.3)
    op_box(ax, 11.42, 7.60, 0.45, 0.25, r"$g_{vert}$", "", "orange", fs=6.3)
    circled_symbol(ax, 10.84, 7.28, r"$\times$")
    circled_symbol(ax, 11.18, 7.02, "+")
    op_box(ax, 11.37, 6.72, 0.70, 0.62, "Fusion Network", "1x1x1 Conv3D\n-> GELU ->\n1x1x1 Conv3D\n+ GN$_w$ + GELU", "orange", fs=5.0)
    arrow(ax, (9.94, 8.67), (10.38, 8.67), lw=0.7)
    arrow(ax, (9.94, 7.10), (10.84, 7.28), lw=0.7)
    arrow(ax, (11.67, 7.60), (11.67, 7.34), lw=0.7)
    arrow(ax, (11.04, 7.60), (10.84, 7.36), lw=0.7)
    arrow(ax, (10.84, 7.20), (11.10, 7.02), lw=0.7)
    arrow(ax, (11.18, 7.02), (11.37, 7.04), lw=0.7)

    cube(ax, 12.45, 7.52, s=0.44, depth=0.13, kind="latent")
    text(ax, 12.60, 8.46, r"$v_{l+1}$", size=9, weight="bold")
    rounded_box(ax, (13.22, 7.48), (0.83, 1.06), "Projection\nHead\n\nConv3D\n$C \\to 2C$\n+ GELU\n$2C \\to L+1$:\n$(\\boldsymbol{\\delta},u)$", fc="#f3f8ff", ec="#2b5c9a", fontsize=5.8, weight="bold")
    arrow(ax, (11.93, 7.04), (12.45, 7.76), lw=0.8)
    arrow(ax, (12.91, 7.76), (13.22, 8.02), lw=0.8)

    text(ax, 14.42, 9.20, "Output: NLTE\nPopulation Cube $\\widehat{\\mathbf{n}}$", size=6.8, weight="bold")
    cube(ax, 14.12, 7.62, s=0.66, depth=0.15, kind="heat")
    draw_axes_near_cube(ax, 14.12, 7.62, 0.66)
    text(ax, 14.30, 6.62, "Output layer, no parameters:\n"
         r"$\widehat{n}_i=\widehat{n}_{\rm tot}\,f^*_i e^{\delta_i}/\sum_j f^*_j e^{\delta_j}$" "\n"
         r"$\widehat{n}_{\rm tot}=(\rho/m_{\rm a})\,10^{6\tanh(u/6)}$" "\n"
         r"$(N_z,N_y,N_x,L)$ [m$^{-3}$]", size=5.6)
    arrow(ax, (14.05, 8.03), (14.12, 8.05), lw=0.8)

    circled_symbol(ax, 11.58, 6.34, "+", r=0.085, size=9)
    arrow(ax, (11.58, 6.22), (11.58, 6.02), lw=0.7)

    ax.plot([3.98, 1.45], [6.34, 5.35], color="#777777", lw=0.7, ls=(0, (3, 3)))
    ax.plot([7.45, 6.92], [6.24, 5.35], color="#777777", lw=0.7, ls=(0, (3, 3)))
    ax.plot([7.45, 11.55], [6.24, 5.35], color="#777777", lw=0.7, ls=(0, (3, 3)))


def draw_spectral_panel(ax):
    rounded_box(ax, (0.15, 2.22), (6.05, 3.05), "", fc="#fbfdff", ec=BLUE_EDGE, lw=0.75, radius=0.05, ls="--")
    text(ax, 3.15, 5.10, "(b) Spectral Branch (horizontal: x-y)", size=7.0, color=BLUE, weight="bold")
    steps = [
        ("1. Input Gate", "Conv3D -> GELU\n-> Conv3D -> sigmoid\nthen multiply input"),
        ("2. FFT2 (x, y)", "per depth slice\n(rfft2)"),
        ("3. Metric-aware\nFrequency Embedding", r"$k_x$" + "\n" + r"$k_y$" + "\n" + r"$\sqrt{k_x^2+k_y^2}$" + "\n" + r"$\mathrm{sign}(k_y)$"),
        ("4. Frequency MLPs", r"$\to$ Complex Weights" + "\n" + "+ rank-8 term" + "\n" + r"$(a_r,a_i)$"),
        ("5. Complex\nSpectral\nMultiplication", r"$\odot$"),
        ("6. iFFT2", ""),  # noqa
        ("7. Post MLP", "1x1x1 Conv3D\n+ GELU ->\n1x1x1 Conv3D"),
    ]
    xs = [0.35, 1.18, 2.22, 3.30, 4.27, 4.98, 5.72]
    for i, (title, body) in enumerate(steps):
        title_size = 4.4 if i >= 5 else 5.0
        text(ax, xs[i], 4.76, title, size=title_size, weight="bold")
        if i in (0, 5, 6):
            cube(ax, xs[i] - 0.10, 3.44, s=0.38, depth=0.13, kind="blue")
        elif i == 1:
            cube(ax, xs[i] - 0.09, 3.62, s=0.28, depth=0.12, kind="blue")
            cube(ax, xs[i] - 0.09, 3.03, s=0.28, depth=0.12, kind="blue")
            mini_heatmap(ax, xs[i] + 0.27, 3.38, w=0.40, h=0.58)
            text(ax, xs[i] + 0.54, 3.28, r"$k_z$", size=8, weight="bold")
            ax.plot([xs[i] + 0.18, xs[i] + 0.18], [3.38, 4.02], color="#333333", lw=0.6, ls=(0, (2, 3)))
        elif i == 2:
            rounded_box(ax, (xs[i] - 0.19, 3.26), (0.52, 1.12), body, fc="#f5faff", ec=BLUE_EDGE, fontsize=6.0)
        elif i == 3:
            for r in range(3):
                for c in range(4):
                    ax.add_patch(Circle((xs[i] - 0.22 + c * 0.15, 3.47 + r * 0.20), 0.032, facecolor="#d7191c" if c < 2 else "#2c7bb6", edgecolor="none"))
        elif i == 4:
            for r in range(3):
                for c in range(4):
                    ax.add_patch(Circle((xs[i] - 0.22 + c * 0.15, 3.47 + r * 0.20), 0.032, facecolor="#54a24b", edgecolor="#2b6b29", lw=0.3))
            for r in range(3):
                ax.plot([xs[i] - 0.25, xs[i] + 0.25], [3.47 + r * 0.20, 3.47 + r * 0.20], color="#9bbd95", lw=0.4)
            for c in range(4):
                ax.plot([xs[i] - 0.22 + c * 0.15, xs[i] - 0.22 + c * 0.15], [3.42, 3.92], color="#9bbd95", lw=0.4)
        if body and i not in (2, 3, 4):
            text(ax, xs[i], 4.43, body, size=4.8)
        if i < len(xs) - 1:
            ax.plot([xs[i] + 0.46, xs[i] + 0.46], [2.92, 4.72], color="#cfd7e7", lw=0.55)
            arrow(ax, (xs[i] + 0.35, 3.67), (xs[i + 1] - 0.30, 3.67), lw=0.65, ms=6)

    text(ax, 0.38, 2.85, r"$(N_z,N_y,N_x,C)$", size=5.8)
    text(ax, 1.28, 2.85, r"$(N_z,N_y,N_x^+,C)$", size=5.8)
    text(ax, 2.25, 2.85, r"$(N_z,N_y,N_x^+,4)$", size=5.8)
    text(ax, 3.40, 2.85, r"$(N_z,N_y,N_x^+,C,2)$", size=5.8)
    text(ax, 4.36, 2.85, r"$(N_z,N_y,N_x^+,C)$", size=5.8)
    text(ax, 5.70, 2.85, r"$(N_z,N_y,N_x,C)$", size=5.8)
    rounded_box(ax, (0.43, 2.30), (3.85, 0.28), r"$N_x^+=N_x/2+1$ (rfft along x).  Full spectrum is used (no truncation)." + "\n" + r"Frequencies use physical spacings $\Delta x,\Delta y$.", fc="#ffffff", ec="#ccd6e5", fontsize=6.2, radius=0.04)


def draw_vertical_panel(ax):
    import math

    rounded_box(ax, (6.35, 2.22), (4.43, 3.05), "", fc="#fbfffb", ec=GREEN_EDGE, lw=0.75, radius=0.05, ls="--")
    text(ax, 8.56, 5.10, "(c) Vertical Branch: spectral operator along height", size=7.0, color=GREEN, weight="bold")

    # 1. a column on its own, non-uniform heights
    text(ax, 6.70, 4.80, "1. Column", size=5.0, weight="bold")
    heights = [0.0, 0.05, 0.11, 0.18, 0.27, 0.38, 0.52, 0.70, 0.92, 1.20]
    ax.plot([6.72, 6.72], [3.12, 3.12 + 1.20], color="#555555", lw=1.0)
    for h in heights:
        ax.add_patch(Circle((6.72, 3.12 + h), 0.035, facecolor="#74b95d", edgecolor="#444444", lw=0.5, zorder=7))
    for j, h in enumerate(heights[:-1]):
        ax.plot([6.80, 6.86], [3.12 + h, 3.12 + h], color="#888888", lw=0.4)
    text(ax, 6.74, 2.95, "$v(z_j)$, non-\nuniform $z_j$", size=4.6)
    text(ax, 6.74, 2.70, "gate, $+z$", size=4.6)
    arrow(ax, (6.95, 3.72), (7.17, 3.72), lw=0.65, ms=6)

    # 2. exact transform of the piecewise-linear column
    text(ax, 7.60, 4.80, "2. Height transform", size=5.0, weight="bold")
    rounded_box(ax, (7.20, 3.30), (0.80, 1.20),
                "$X_m=\\int v(z)\\,e^{-ik_m z}\\,dz$\n\nexact for the\npiecewise-linear\nprofile through $z_j$\n\n$k_m=2\\pi m/L$",
                fc="#f5fff3", ec=GREEN_EDGE, fontsize=4.4)
    arrow(ax, (8.04, 3.72), (8.22, 3.72), lw=0.65, ms=6)

    # 3. W(k): a smooth function of the physical wavenumber, cut off per column
    text(ax, 8.73, 4.80, "3. $Y_m=W(k_m)X_m$", size=5.0, weight="bold")
    x0, y0, w, h = 8.27, 3.36, 0.92, 0.95
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor="white", edgecolor="#777777", lw=0.5, zorder=3))
    ks = [i / 60 for i in range(61)]
    def taper(k):
        u = min(max((k - 0.55) / 0.27, 0.0), 1.0)
        return 0.5 * (1 + math.cos(math.pi * u))
    wk = [0.55 + 0.25 * math.cos(5 * k) * math.exp(-1.2 * k) for k in ks]
    ax.plot([x0 + k * w for k in ks], [y0 + h * v for v in wk], color="#8bc37d", lw=0.8, ls=(0, (2, 1.5)), zorder=4)
    ax.plot([x0 + k * w for k in ks], [y0 + h * v * taper(k) for v, k in zip(wk, ks)], color=GREEN, lw=1.1, zorder=5)
    for k in [i / 12 for i in range(13)]:
        ax.add_patch(Circle((x0 + k * w, y0 + h * wk[int(round(k * 60))] * taper(k)), 0.022, facecolor=GREEN,
                            edgecolor="none", zorder=6))
    ax.plot([x0 + 0.55 * w] * 2, [y0, y0 + h], color="#999999", lw=0.4, ls=(0, (2, 2)), zorder=4)
    ax.plot([x0 + 0.82 * w] * 2, [y0, y0 + h], color="#999999", lw=0.4, ls=(0, (2, 2)), zorder=4)
    text(ax, x0 + 0.52 * w, y0 - 0.09, "$\\frac{2\\pi}{3\\Delta z}$", size=4.8)
    text(ax, x0 + 0.86 * w, y0 - 0.09, "$\\frac{\\pi}{\\Delta z}$", size=4.8)
    text(ax, x0 + w + 0.03, y0 + 0.02, "$k$", size=5.5, ha="left")
    text(ax, 8.73, 2.94, "complex $C\\times C$ matrix $\\times$ MLP($k$)\n+ rank-8 $U(k)V(k)$; taper to 0 at\nthe column's Nyquist ($\\Delta z$: median spacing)",
         size=4.5)
    arrow(ax, (9.24, 3.72), (9.42, 3.72), lw=0.65, ms=6)

    # 4. back on the column's heights
    text(ax, 10.05, 4.80, "4. Inverse series", size=5.0, weight="bold")
    rounded_box(ax, (9.47, 3.30), (1.15, 1.20),
                "$v^{vert}(z_j)=\\frac{1}{L}\\,[\\,Y_0$\n$+\\,2\\,\\mathrm{Re}\\sum_{m\\geq1}Y_m e^{ik_m z_j}]$\n\nthen Post MLP\n(1x1x1 Conv3D -> GELU\n-> 1x1x1 Conv3D)",
                fc="#f5fff3", ec=GREEN_EDGE, fontsize=4.8)

    text(ax, 7.56, 2.28 + 0.18, "$(N_z,C)$ per column", size=5.0)
    text(ax, 8.73, 2.28 + 0.18, "$(M,C)$ complex", size=5.0)
    text(ax, 10.05, 2.28 + 0.18, "$(N_z,C)$ per column", size=5.0)
    text(ax, 8.56, 2.30, "$L=40$ Mm, or twice the column height if longer; shortest wavelength 40 km", size=4.9)


def draw_fusion_panel(ax):
    rounded_box(ax, (10.90, 2.22), (4.18, 3.05), "", fc="#fffdf7", ec=ORANGE_EDGE, lw=0.75, radius=0.05, ls="--")
    text(ax, 12.99, 5.10, "(d) Gated Fusion (per FFNO block)", size=7.2, color=ORANGE, weight="bold")
    text(ax, 11.10, 4.72, "Inputs $[v,v^{spec},v^{vert}]$", size=5.5, weight="bold")
    cube(ax, 11.08, 4.15, s=0.34, depth=0.13, kind="blue")
    text(ax, 11.17, 3.73, r"$(N_z,N_y,N_x,C)$", size=5.0)
    op_box(ax, 11.76, 4.17, 0.72, 0.42, "Gate Network", "Conv3D -> GELU\n-> Conv3D -> $\\sigma$", "orange", fs=4.5)
    op_box(ax, 12.72, 4.08, 0.54, 0.28, r"$g_{spec},g_{vert}$", "$\\sigma()$", "orange", fs=4.7)
    arrow(ax, (11.44, 4.40), (11.76, 4.38), lw=0.7, ms=6)
    arrow(ax, (12.48, 4.38), (12.80, 4.24), lw=0.7, ms=6)

    text(ax, 11.10, 3.34, "From Spectral Branch\n$v^{spec}$", size=5.7, weight="bold", ha="left", color=BLUE)
    cube(ax, 11.08, 2.86, s=0.34, depth=0.13, kind="blue")
    text(ax, 11.10, 2.62, "From Vertical Branch\n$v^{vert}$", size=5.7, weight="bold", ha="left", color=GREEN)
    cube(ax, 11.08, 2.35, s=0.34, depth=0.13, kind="green")
    circled_symbol(ax, 13.20, 3.32, r"$\times$", r=0.060, size=6.5)
    circled_symbol(ax, 13.47, 3.64, "+", r=0.070, size=8)
    op_box(ax, 13.82, 3.26, 0.76, 0.70, "Fusion Network", "Conv3D -> GELU\n-> Conv3D\n+ branch mean\n+ GN$_w$ + GELU", "orange", fs=4.3)
    cube(ax, 14.70, 3.25, s=0.30, depth=0.11, kind="latent")
    text(ax, 14.82, 4.23, "Fused\nOutput $v'$", size=5.4, weight="bold")
    text(ax, 14.77, 2.86, r"$(N_z,N_y,N_x,C)$", size=4.6)
    arrow(ax, (13.18, 4.08), (13.20, 3.39), lw=0.7, ms=6)
    arrow(ax, (11.42, 3.05), (13.13, 3.31), lw=0.7, ms=6)
    arrow(ax, (11.42, 2.52), (13.41, 3.58), lw=0.7, ms=6)
    arrow(ax, (13.27, 3.33), (13.43, 3.60), lw=0.7, ms=6)
    arrow(ax, (13.54, 3.64), (13.82, 3.61), lw=0.7, ms=6)
    arrow(ax, (14.58, 3.61), (14.70, 3.51), lw=0.7, ms=6)
    rounded_box(ax, (11.50, 2.30), (2.96, 0.24), r"$\otimes$ : Channel-wise multiplication      $\oplus$ : Element-wise addition", fc="#ffffff", ec="#ccd0d8", fontsize=5.6, radius=0.03)


def draw_bottom_panels(ax):
    rounded_box(ax, (0.15, 0.20), (6.05, 1.70), "", fc="#fbfaff", ec="#b9acd6", lw=0.75, radius=0.04)
    text(ax, 0.30, 1.70, "Key Points", size=7.6, color="#5a38a2", weight="bold", ha="left")
    key = (
        "- 2D Fourier operator in (x,y) at every depth slice; uses full spectrum (rfft2/irfft2).\n"
        "- Frequency weights: MLP of $(k_x,k_y,\\sqrt{k_x^2+k_y^2},\\mathrm{sign}(k_y))$, plus a rank-8 mixing term.\n"
        "- Vertical branch: the same kind of operator along height, on each column's own heights.\n"
        "- Its transform is exact for the piecewise-linear column; each column keeps only the $k$ it resolves.\n"
        "- Spectral and vertical branches are adaptively gated per block and fused.\n"
        "- Residual fused, pointwise and MLP updates; GN$_w$ weights each level by its height interval.\n"
        "- The head gives $(\\boldsymbol{\\delta},u)$; a fixed output layer turns them into populations in m$^{-3}$."
    )
    text(ax, 0.28, 1.48, key, size=5.9, ha="left", va="top")

    rounded_box(ax, (6.35, 0.20), (4.43, 1.70), "", fc="#fbfaff", ec="#b9acd6", lw=0.75, radius=0.04)
    text(ax, 6.50, 1.70, "Notation", size=7.6, color="#5a38a2", weight="bold", ha="left")
    notation = (
        "$N_x,N_y,N_z$             - grid sizes in x, y, z\n"
        "$C$                       - latent channel dimension\n"
        "$N_x^+$                    - $N_x/2+1$ (rfft along x)\n"
        "$N$                       - number of FFNO blocks (here $N=4$)\n"
        "$\\Delta x,\\Delta y$         - horizontal grid spacings (used in frequencies)\n"
        "$k_m,\\ L$                  - vertical wavenumbers, periodic length\n"
        "$\\Delta z$                    - median level spacing of a column\n"
        "$L$ levels; $f^*_i$            - LTE level fractions (Saha-Boltzmann)\n"
        "$\\sigma(x)$; GN$_w$           - sigmoid; height-weighted GroupNorm"
    )
    text(ax, 6.50, 1.48, notation, size=5.6, ha="left", va="top")

    rounded_box(ax, (10.90, 0.20), (4.18, 1.70), "", fc="#fbfaff", ec="#b9acd6", lw=0.75, radius=0.04)
    text(ax, 11.05, 1.70, "Overall Mapping", size=7.6, color="#5a38a2", weight="bold", ha="left")
    mapping = (
        r"$\mathcal{G}_{\theta}: [\log_{10}T,\ v_x,\ v_y,\ v_z,$"
        "\n"
        r"$\qquad \log_{10}n_e,\ \log_{10}\rho] \rightarrow \widehat{\mathbf{n}}$"
        "\n\nHead: $L+1$ channels, $\\delta_i$ per level and $u$\nfor the element; output layer in m$^{-3}$.\n"
        r"Loss on standardised $\log_{10}\widehat{n}_i$."
    )
    text(ax, 11.05, 1.43, mapping, size=6.8, ha="left", va="top")


def build_figure(output_prefix="ffno_schematic_matplotlib"):
    # A4 landscape in inches. The coordinate frame uses the same aspect ratio,
    # so saved PNG/PDF outputs are exactly A4 landscape, without later cropping.
    # The padded limits scale the complete schematic down uniformly and keep the
    # outer frame clear of PDF renderer/printer boundary clipping.
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    scale_padding = 0.04
    x_center, y_center = 15.2 / 2, 10.75 / 2
    x_half = (15.2 / 2) * (1 + scale_padding)
    y_half = (10.75 / 2) * (1 + scale_padding)
    ax.set_xlim(x_center - x_half, x_center + x_half)
    ax.set_ylim(y_center - y_half, y_center + y_half)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    rounded_box(ax, (0.02, 0.04), (15.10, 10.62), "", fc="white", ec="#444444", lw=0.8, radius=0.04, zorder=0)
    draw_top_architecture(ax)
    draw_spectral_panel(ax)
    draw_vertical_panel(ax)
    draw_fusion_panel(ax)
    draw_bottom_panels(ax)

    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.015, top=0.985)
    fig.savefig(f"{output_prefix}.png", dpi=300)
    fig.savefig(f"{output_prefix}.pdf")
    return fig, ax


if __name__ == "__main__":
    build_figure()
    if os.environ.get("SHOW_FFNO_SCHEMATIC"):
        plt.show()

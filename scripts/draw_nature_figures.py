from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import patches
from matplotlib.path import Path as MplPath


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "artifacts" / "figures"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


INK = "#222222"
MUTED = "#666666"
LIGHT = "#F7F7F7"
LINE = "#A8A8A8"
NEUTRAL = "#D9DEE3"
BLUE = "#0F4D92"
BLUE_LIGHT = "#E8F0F8"
GREEN = "#4D7C3F"
GREEN_LIGHT = "#EEF5EA"
GOLD = "#C9822B"
GOLD_LIGHT = "#F8EFE1"
RED = "#B64342"
BLACK_SLICE = "#111820"

NORMAL_EXAMPLE = [
    ROOT / "dataset" / "chair" / f"test_chair_0890_gate_{idx}.png" for idx in range(3)
]
IMAGE2D_EXAMPLE = [
    ROOT / "dataset" / "image2d" / f"image2d_0000_gate_{idx}.png" for idx in range(3)
]


def mm_to_in(mm: float) -> float:
    return mm / 25.4


def save_pub(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    base = FIGURE_DIR / stem
    fig.savefig(f"{base}.svg", bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    fig.savefig(f"{base}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{base}.tiff", dpi=600, bbox_inches="tight")


def setup_ax(fig: plt.Figure) -> plt.Axes:
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    return ax


def box(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    label: str,
    *,
    fc: str = "white",
    ec: str = LINE,
    lw: float = 0.7,
    text_color: str = INK,
    fontsize: float = 7,
    bold: bool = True,
    radius: float = 0.006,
) -> patches.FancyBboxPatch:
    x, y = xy
    w, h = wh
    patch = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        color=text_color,
        fontsize=fontsize,
        fontweight="bold" if bold else "normal",
    )
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = INK,
    lw: float = 0.8,
    rad: float = 0.0,
    alpha: float = 1.0,
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            linewidth=lw,
            shrinkA=0,
            shrinkB=0,
            mutation_scale=7,
            connectionstyle=f"arc3,rad={rad}",
            alpha=alpha,
        ),
    )


def panel_label(ax: plt.Axes, label: str, x: float, y: float) -> None:
    ax.text(x, y, label, fontsize=9, fontweight="bold", ha="left", va="top", color=INK)


def slice_icon(ax: plt.Axes, x: float, y: float, w: float, h: float, name: str, color: str, shape: int) -> None:
    rect = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.003,rounding_size=0.006",
        linewidth=0.65,
        edgecolor=color,
        facecolor="white",
    )
    ax.add_patch(rect)
    inner = patches.Rectangle((x + 0.012, y + 0.012), w - 0.024, h - 0.032, facecolor=color, alpha=0.13, edgecolor="none")
    ax.add_patch(inner)
    if shape == 0:
        ax.add_patch(patches.Wedge((x + w * 0.48, y + h * 0.48), w * 0.25, 12, 170, facecolor=color, edgecolor="none", alpha=0.92))
        ax.add_patch(patches.Polygon([(x + w * 0.25, y + h * 0.28), (x + w * 0.72, y + h * 0.31), (x + w * 0.86, y + h * 0.22), (x + w * 0.18, y + h * 0.19)], facecolor=color, edgecolor="none", alpha=0.92))
    elif shape == 1:
        ax.add_patch(patches.Polygon([(x + w * 0.22, y + h * 0.27), (x + w * 0.40, y + h * 0.58), (x + w * 0.66, y + h * 0.58), (x + w * 0.85, y + h * 0.25), (x + w * 0.62, y + h * 0.35), (x + w * 0.39, y + h * 0.35)], facecolor=color, edgecolor="none", alpha=0.92))
    else:
        ax.add_patch(patches.Wedge((x + w * 0.56, y + h * 0.41), w * 0.30, -25, 160, facecolor=color, edgecolor="none", alpha=0.92))
        ax.add_patch(patches.Polygon([(x + w * 0.19, y + h * 0.28), (x + w * 0.42, y + h * 0.19), (x + w * 0.78, y + h * 0.40), (x + w * 0.60, y + h * 0.50)], facecolor=color, edgecolor="none", alpha=0.92))
    ax.text(x + w / 2, y - 0.012, name, ha="center", va="top", fontsize=6.5, color=MUTED)


def slice_icon_unlabeled(ax: plt.Axes, x: float, y: float, w: float, h: float, color: str, shape: int) -> None:
    rect = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.003,rounding_size=0.006",
        linewidth=0.65,
        edgecolor=color,
        facecolor="white",
    )
    ax.add_patch(rect)
    inner = patches.Rectangle((x + 0.012, y + 0.012), w - 0.024, h - 0.024, facecolor=color, alpha=0.13, edgecolor="none")
    ax.add_patch(inner)
    if shape == 0:
        ax.add_patch(patches.Wedge((x + w * 0.48, y + h * 0.50), w * 0.25, 12, 170, facecolor=color, edgecolor="none", alpha=0.92))
        ax.add_patch(patches.Polygon([(x + w * 0.25, y + h * 0.30), (x + w * 0.72, y + h * 0.33), (x + w * 0.86, y + h * 0.22), (x + w * 0.18, y + h * 0.20)], facecolor=color, edgecolor="none", alpha=0.92))
    elif shape == 1:
        ax.add_patch(patches.Polygon([(x + w * 0.22, y + h * 0.28), (x + w * 0.40, y + h * 0.58), (x + w * 0.66, y + h * 0.58), (x + w * 0.85, y + h * 0.25), (x + w * 0.62, y + h * 0.36), (x + w * 0.39, y + h * 0.36)], facecolor=color, edgecolor="none", alpha=0.92))
    else:
        ax.add_patch(patches.Wedge((x + w * 0.56, y + h * 0.43), w * 0.30, -25, 160, facecolor=color, edgecolor="none", alpha=0.92))
        ax.add_patch(patches.Polygon([(x + w * 0.19, y + h * 0.29), (x + w * 0.42, y + h * 0.20), (x + w * 0.78, y + h * 0.42), (x + w * 0.60, y + h * 0.51)], facecolor=color, edgecolor="none", alpha=0.92))


def read_gray_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32)
    if arr.max() > arr.min():
        lo, hi = np.percentile(arr, [1, 99])
        arr = np.clip((arr - lo) / max(hi - lo, 1.0), 0, 1)
    else:
        arr = arr / 255.0
    return arr


def dataset_slice(ax: plt.Axes, path: Path, x: float, y: float, w: float, h: float, edge: str, label: str) -> None:
    arr = read_gray_image(path)
    ax.imshow(
        arr,
        cmap="gray",
        extent=(x, x + w, y, y + h),
        origin="upper",
        vmin=0,
        vmax=1,
        aspect="auto",
        zorder=2,
    )
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.002,rounding_size=0.004",
            linewidth=0.75,
            edgecolor=edge,
            facecolor="none",
            zorder=3,
        )
    )
    if label:
        ax.text(x + w / 2, y - 0.012, label, ha="center", va="top", fontsize=5.8, color=MUTED)


def draw_scaled_slice(
    ax: plt.Axes,
    path: Path,
    x: float,
    y: float,
    w: float,
    h: float,
    edge: str,
    scale: float,
    label: str,
) -> None:
    arr = np.clip(read_gray_image(path) * scale, 0, 1)
    ax.imshow(
        arr,
        cmap="gray",
        extent=(x, x + w, y, y + h),
        origin="upper",
        vmin=0,
        vmax=1,
        aspect="auto",
        zorder=2,
    )
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.002,rounding_size=0.004",
            linewidth=0.75,
            edgecolor=edge,
            facecolor="none",
            zorder=3,
        )
    )
    if label:
        ax.text(x + w / 2, y - 0.012, label, ha="center", va="top", fontsize=5.8, color=edge)


def feature_vector(ax: plt.Axes, x: float, y: float, color: str, label: str) -> None:
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            0.112,
            0.036,
            boxstyle="round,pad=0.003,rounding_size=0.004",
            linewidth=0.55,
            edgecolor=LINE,
            facecolor="white",
        )
    )
    for i, a in enumerate([0.25, 0.45, 0.75, 0.35, 0.60]):
        ax.add_patch(patches.Rectangle((x + 0.010 + i * 0.020, y + 0.010), 0.013, 0.016, facecolor=color, alpha=a, edgecolor="none"))
    ax.text(x + 0.056, y - 0.011, label, ha="center", va="top", fontsize=6.5, color=INK)


def draw_method_nature() -> None:
    fig = plt.figure(figsize=(mm_to_in(183), mm_to_in(92)), facecolor="white")
    ax = setup_ax(fig)
    panel_label(ax, "a", 0.025, 0.965)
    ax.text(0.058, 0.948, "Attention-residual fusion preserves gate-level evidence while restoring slice-specific information", ha="left", va="top", fontsize=8.2, fontweight="bold", color=INK)

    # Input slices
    ax.text(0.055, 0.845, "range-gated slices", ha="left", va="bottom", fontsize=7, color=INK)
    slice_icon(ax, 0.055, 0.725, 0.092, 0.075, "gate 0", BLUE, 0)
    slice_icon(ax, 0.055, 0.575, 0.092, 0.075, "gate 1", GOLD, 1)
    slice_icon(ax, 0.055, 0.425, 0.092, 0.075, "gate 2", GREEN, 2)

    # Encoder
    box(ax, (0.205, 0.535), (0.125, 0.160), "shared\nCNN encoder", fc=LIGHT, ec=LINE, fontsize=7.3)
    ax.text(0.267, 0.515, "one encoder, shared weights", ha="center", va="top", fontsize=6.2, color=MUTED)
    arrow(ax, (0.147, 0.762), (0.205, 0.650), color=INK, lw=0.75)
    arrow(ax, (0.147, 0.613), (0.205, 0.613), color=INK, lw=0.75)
    arrow(ax, (0.147, 0.463), (0.205, 0.575), color=INK, lw=0.75)

    # Features
    ax.text(0.395, 0.845, "slice features", ha="center", va="bottom", fontsize=7, color=INK)
    feature_vector(ax, 0.360, 0.735, BLUE, "f0")
    feature_vector(ax, 0.360, 0.585, GOLD, "f1")
    feature_vector(ax, 0.360, 0.435, GREEN, "f2")
    arrow(ax, (0.330, 0.615), (0.360, 0.753), color=INK, lw=0.65, rad=-0.12)
    arrow(ax, (0.330, 0.615), (0.360, 0.603), color=INK, lw=0.65)
    arrow(ax, (0.330, 0.615), (0.360, 0.453), color=INK, lw=0.65, rad=0.12)

    # Attention branch
    box(ax, (0.535, 0.670), (0.150, 0.110), "attention scorer", fc=BLUE_LIGHT, ec=BLUE, fontsize=7.1)
    ax.text(0.610, 0.655, "softmax over gates", ha="center", va="top", fontsize=6.2, color=MUTED)
    box(ax, (0.745, 0.680), (0.140, 0.090), "weighted sum", fc="white", ec=BLUE, fontsize=7.1)
    ax.text(0.815, 0.655, "f_att = sum alpha_i f_i", ha="center", va="top", fontsize=6.2, color=BLUE)
    for y in [0.753, 0.603, 0.453]:
        arrow(ax, (0.472, y), (0.535, 0.725), color=BLUE, lw=0.7, rad=0.08)
    arrow(ax, (0.685, 0.725), (0.745, 0.725), color=BLUE, lw=0.75)

    # Residual branch
    box(ax, (0.535, 0.340), (0.170, 0.130), "concat residual\nprojection", fc=GREEN_LIGHT, ec=GREEN, fontsize=7.1)
    ax.text(0.620, 0.318, "[f0, f1, f2] -> f_res", ha="center", va="top", fontsize=6.2, color=GREEN)
    for y in [0.753, 0.603, 0.453]:
        arrow(ax, (0.472, y), (0.535, 0.405), color=GREEN, lw=0.7, rad=-0.10)

    # Merge and output
    ax.add_patch(patches.Circle((0.810, 0.500), 0.030, facecolor="white", edgecolor=INK, linewidth=0.7))
    ax.text(0.810, 0.500, "+", ha="center", va="center", fontsize=11, fontweight="bold", color=INK)
    arrow(ax, (0.815, 0.680), (0.815, 0.530), color=BLUE, lw=0.75)
    arrow(ax, (0.705, 0.405), (0.785, 0.500), color=GREEN, lw=0.75)
    box(ax, (0.885, 0.450), (0.080, 0.100), "classifier\nK=6", fc=GOLD_LIGHT, ec=GOLD, fontsize=7.0)
    arrow(ax, (0.840, 0.500), (0.885, 0.500), color=INK, lw=0.75)
    ax.text(0.925, 0.425, "class prediction", ha="center", va="top", fontsize=6.2, color=MUTED)

    # Direct notes
    ax.plot([0.610, 0.610], [0.625, 0.570], color=BLUE, linewidth=0.6)
    ax.text(0.610, 0.555, "interpretable gate weights", ha="center", va="top", fontsize=6.2, color=BLUE)
    ax.plot([0.620, 0.620], [0.318, 0.270], color=GREEN, linewidth=0.6)
    ax.text(0.620, 0.255, "compensates feature loss from pure weighted averaging", ha="center", va="top", fontsize=6.2, color=GREEN)

    save_pub(fig, "method_architecture_attention_residual_nature")
    plt.close(fig)


def mini_model(ax: plt.Axes, x: float, y: float) -> None:
    pts1 = [(x, y), (x + 0.045, y + 0.030), (x + 0.100, y + 0.010), (x + 0.060, y - 0.035)]
    pts2 = [(x + 0.045, y + 0.030), (x + 0.075, y + 0.052), (x + 0.132, y + 0.032), (x + 0.100, y + 0.010)]
    pts3 = [(x + 0.100, y + 0.010), (x + 0.132, y + 0.032), (x + 0.090, y - 0.022), (x + 0.060, y - 0.035)]
    ax.add_patch(patches.Polygon(pts1, facecolor=NEUTRAL, edgecolor=LINE, linewidth=0.45))
    ax.add_patch(patches.Polygon(pts2, facecolor="#EEF1F4", edgecolor=LINE, linewidth=0.45))
    ax.add_patch(patches.Polygon(pts3, facecolor="#C9D0D6", edgecolor=LINE, linewidth=0.45))


def draw_gate_response(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    ax.plot([x, x], [y, y + h], color=LINE, linewidth=0.55)
    ax.plot([x, x + w], [y, y], color=LINE, linewidth=0.55)
    curves = [
        (BLUE, [0.06, 0.24, 0.38, 0.56, 0.68]),
        (GOLD, [0.20, 0.38, 0.52, 0.72, 0.86]),
        (GREEN, [0.38, 0.56, 0.70, 0.86, 0.98]),
    ]
    heights = [0.58, 0.82, 0.42]
    for (color, xs), ht in zip(curves, heights):
        px = [x + w * v for v in xs]
        py = [y, y, y + h * ht, y + h * ht, y]
        ax.plot(px, py, color=color, linewidth=1.0)
    ax.text(x - 0.012, y + h * 0.92, "Wg(R)", ha="right", va="center", fontsize=5.8, color=MUTED)
    ax.text(x + w * 0.5, y - 0.025, "range R", ha="center", va="top", fontsize=5.8, color=MUTED)


def draw_pipeline_nature() -> None:
    fig = plt.figure(figsize=(mm_to_in(183), mm_to_in(92)), facecolor="white")
    ax = setup_ax(fig)
    panel_label(ax, "b", 0.025, 0.965)
    ax.text(0.058, 0.948, "Physics-inspired range gates convert 3D geometry into multi-slice inputs and a flat-echo control", ha="left", va="top", fontsize=8.2, fontweight="bold", color=INK)

    # Main flow
    box(ax, (0.055, 0.620), (0.130, 0.145), "ModelNet10\ngeometry", fc="white", ec=LINE, fontsize=7.0)
    mini_model(ax, 0.087, 0.645)
    box(ax, (0.255, 0.585), (0.210, 0.210), "Blender scene", fc=BLUE_LIGHT, ec=BLUE, fontsize=7.2)
    ax.text(0.360, 0.655, "camera-axis depth R", ha="center", va="center", fontsize=6.2, color=MUTED)
    ax.add_patch(patches.Circle((0.310, 0.710), 0.014, facecolor=GOLD, edgecolor="none", alpha=0.9))
    ax.plot([0.300, 0.270], [0.710, 0.730], color=GOLD, linewidth=1.0)
    ax.plot([0.300, 0.270], [0.708, 0.690], color=GOLD, linewidth=1.0)
    ax.add_patch(patches.Wedge((0.375, 0.635), 0.050, 0, 180, facecolor=BLUE, edgecolor="none", alpha=0.35))
    ax.add_patch(patches.Wedge((0.410, 0.635), 0.060, 0, 180, facecolor=GREEN, edgecolor="none", alpha=0.35))
    ax.add_patch(patches.Rectangle((0.427, 0.610), 0.040, 0.060, facecolor="#6F7E8F", edgecolor="none", alpha=0.45))

    box(ax, (0.545, 0.585), (0.205, 0.210), "", fc=GOLD_LIGHT, ec=GOLD, fontsize=7.2)
    ax.text(0.647, 0.765, "range-gate response", ha="center", va="center", fontsize=7.2, fontweight="bold", color=INK)
    draw_gate_response(ax, 0.585, 0.645, 0.130, 0.085)

    box(ax, (0.825, 0.560), (0.145, 0.310), "", fc=GREEN_LIGHT, ec=GREEN, fontsize=7.2)
    ax.text(0.897, 0.835, "gated slices", ha="center", va="center", fontsize=7.2, fontweight="bold", color=INK)
    slice_rows = [
        (0.765, BLUE, NORMAL_EXAMPLE[0], "gate 0"),
        (0.675, GOLD, NORMAL_EXAMPLE[1], "gate 1"),
        (0.585, GREEN, NORMAL_EXAMPLE[2], "gate 2"),
    ]
    for yy, color, path, label in slice_rows:
        dataset_slice(ax, path, 0.875, yy, 0.078, 0.055, color, "")
        ax.text(0.862, yy + 0.027, label, ha="right", va="center", fontsize=5.8, color=MUTED)

    arrow(ax, (0.185, 0.692), (0.255, 0.692), color=INK, lw=0.75)
    arrow(ax, (0.465, 0.692), (0.545, 0.692), color=INK, lw=0.75)
    arrow(ax, (0.750, 0.692), (0.825, 0.692), color=INK, lw=0.75)

    # Model equation
    ax.text(0.360, 0.500, "simplified gated intensity", ha="center", va="center", fontsize=7.0, fontweight="bold", color=INK)
    ax.text(0.360, 0.465, "I(R) = gate response x Lambert term x range loss x atmospheric transmission", ha="center", va="center", fontsize=6.2, color=MUTED)
    ax.plot([0.170, 0.790], [0.435, 0.435], color=LINE, linewidth=0.5)
    ax.text(0.480, 0.408, "physics-inspired rendering baseline; not a full hardware-level laser imaging simulation", ha="center", va="top", fontsize=5.9, color=MUTED)

    # Dataset construction
    box(ax, (0.095, 0.205), (0.190, 0.125), "normal 3D classes", fc="white", ec=LINE, fontsize=7.0)
    ax.text(0.190, 0.230, "chair, desk, sofa, bed, toilet", ha="center", va="center", fontsize=6.1, color=MUTED)
    box(ax, (0.385, 0.205), (0.225, 0.135), "", fc="white", ec=LINE, fontsize=7.0)
    ax.text(0.497, 0.251, "flat-echo false target", ha="center", va="center", fontsize=7.0, fontweight="bold", color=INK)
    ax.text(0.497, 0.226, "same 2D reflection, gate-scaled response", ha="center", va="center", fontsize=6.1, color=MUTED)
    flat_echo_path = IMAGE2D_EXAMPLE[0]
    for x0, edge, scale, label in [
        (0.423, BLUE, 1.00, "strong"),
        (0.468, GOLD, 0.56, "mid"),
        (0.513, GREEN, 0.24, "weak"),
    ]:
        draw_scaled_slice(ax, flat_echo_path, x0, 0.294, 0.038, 0.030, edge, scale, "")
        ax.text(x0 + 0.019, 0.282, label, ha="center", va="top", fontsize=5.4, color=edge)
    box(ax, (0.720, 0.205), (0.200, 0.125), "six-class gated-slice\ndataset", fc=GOLD_LIGHT, ec=GOLD, fontsize=7.0)
    ax.text(0.820, 0.228, "[gate 0, gate 1, gate 2] per sample", ha="center", va="center", fontsize=6.1, color=MUTED)
    arrow(ax, (0.285, 0.268), (0.385, 0.268), color=INK, lw=0.75)
    arrow(ax, (0.610, 0.268), (0.720, 0.268), color=INK, lw=0.75)
    arrow(ax, (0.897, 0.560), (0.850, 0.330), color=GREEN, lw=0.75, rad=-0.25)

    save_pub(fig, "gated_slice_rendering_pipeline_nature")
    plt.close(fig)


def main() -> None:
    draw_method_nature()
    draw_pipeline_nature()
    for name in [
        "method_architecture_attention_residual_nature",
        "gated_slice_rendering_pipeline_nature",
    ]:
        print(FIGURE_DIR / f"{name}.png")
        print(FIGURE_DIR / f"{name}.svg")
        print(FIGURE_DIR / f"{name}.pdf")
        print(FIGURE_DIR / f"{name}.tiff")


if __name__ == "__main__":
    main()

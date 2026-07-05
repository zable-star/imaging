from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "artifacts" / "figures"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)


INK = "#222222"
MUTED = "#666666"
LINE = "#3B3B3B"
BLUE = "#0F4D92"
BLUE_L = "#DCE9F5"
TEAL = "#42949E"
TEAL_L = "#DDEFF1"
GREEN = "#4D7C3F"
GREEN_L = "#E7F1E2"
GOLD = "#C9822B"
GOLD_L = "#F6EAD7"
RED = "#B64342"
RED_L = "#F3DEDC"
GRAY_L = "#F3F3F3"
GRAY_M = "#D8D8D8"


def save_all(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    base = FIGURE_DIR / stem
    fig.savefig(f"{base}.svg", bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    fig.savefig(f"{base}.png", dpi=500, bbox_inches="tight")
    fig.savefig(f"{base}.tiff", dpi=500, bbox_inches="tight")


def arrow(ax, start, end, color=INK, lw=1.1, rad=0.0):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle="-|>",
            linewidth=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
            mutation_scale=9,
            connectionstyle=f"arc3,rad={rad}",
        ),
    )


def draw_stack(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    depth_display: int,
    *,
    color_a: str,
    color_b: str,
    label: str,
    op_after: str | None = None,
    dx: float = 0.010,
    dy: float = 0.010,
):
    for i in range(depth_display - 1, -1, -1):
        fc = color_a if i % 2 else color_b
        rect = patches.Rectangle(
            (x + i * dx, y + i * dy),
            w,
            h,
            facecolor=fc,
            edgecolor=LINE,
            linewidth=0.55,
            alpha=0.92,
        )
        ax.add_patch(rect)
    ax.text(x + w / 2 + (depth_display - 1) * dx / 2, y + h + depth_display * dy + 0.030, label, ha="center", va="bottom", fontsize=8, color=INK)
    if op_after:
        ax.text(x + w + depth_display * dx + 0.028, y - 0.032, op_after, ha="center", va="top", fontsize=7.2, color=MUTED)


def draw_vector(ax, x, y, h, label, color=GRAY_M, text=None):
    poly = patches.Polygon(
        [(x, y), (x + 0.016, y), (x + 0.040, y + h), (x + 0.024, y + h)],
        facecolor=color,
        edgecolor=LINE,
        linewidth=0.7,
        alpha=0.95,
    )
    ax.add_patch(poly)
    ax.text(x + 0.020, y + h + 0.035, label, ha="center", va="bottom", fontsize=8, color=INK)
    if text:
        ax.text(x + 0.020, y - 0.030, text, ha="center", va="top", fontsize=7.2, color=MUTED)


def box(ax, x, y, w, h, label, fc, ec, fontsize=8, bold=True):
    patch = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.006,rounding_size=0.008",
        facecolor=fc,
        edgecolor=ec,
        linewidth=0.9,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fontsize, color=INK, fontweight="bold" if bold else "normal")
    return patch


def draw_gate_feature(ax, x, y, label, color):
    box(ax, x, y, 0.090, 0.052, label, "white", color, fontsize=7.4)
    for i, alpha in enumerate([0.25, 0.45, 0.70, 0.35]):
        ax.add_patch(patches.Rectangle((x + 0.014 + 0.016 * i, y + 0.018), 0.010, 0.015, facecolor=color, alpha=alpha, edgecolor="none"))


def draw_network() -> None:
    fig = plt.figure(figsize=(13.5, 6.1), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    ax.text(0.030, 0.952, "NN-SVG LeNet-style visualization of the slice-attention-residual classifier", ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    ax.text(0.030, 0.914, "Sequential CNN encoder is drawn in the LeNet convention; attention and residual branches are shown as network-head modules.", ha="left", va="top", fontsize=8.2, color=MUTED)

    # Encoder panel
    ax.text(0.032, 0.835, "shared SliceEncoder for each gate", ha="left", va="center", fontsize=9.5, fontweight="bold", color=INK)
    y0 = 0.540
    draw_stack(ax, 0.045, y0, 0.070, 0.180, 1, color_a="#FFFFFF", color_b="#EDEDED", label="1@224x224", op_after="Conv3x3\nBN+ReLU")
    draw_stack(ax, 0.190, y0 + 0.030, 0.055, 0.140, 7, color_a=BLUE_L, color_b="#B7CFE7", label="32@112x112", op_after="MaxPool")
    draw_stack(ax, 0.335, y0 + 0.055, 0.045, 0.095, 9, color_a=TEAL_L, color_b="#A8D3D8", label="64@56x56", op_after="MaxPool")
    draw_stack(ax, 0.475, y0 + 0.073, 0.033, 0.060, 11, color_a=GREEN_L, color_b="#B6D3AD", label="128@1x1", op_after=None)
    ax.text(0.585, y0 - 0.020, "AdaptiveAvgPool", ha="center", va="top", fontsize=7.2, color=MUTED)
    draw_vector(ax, 0.620, y0 + 0.060, 0.090, "128-d", GRAY_M, "Flatten + Linear")

    arrow(ax, (0.125, y0 + 0.090), (0.185, y0 + 0.100))
    arrow(ax, (0.260, y0 + 0.105), (0.330, y0 + 0.108))
    arrow(ax, (0.405, y0 + 0.106), (0.470, y0 + 0.105))
    arrow(ax, (0.540, y0 + 0.105), (0.615, y0 + 0.105))
    ax.text(0.355, 0.455, "CNN blocks: Conv2d -> BatchNorm -> ReLU; first two blocks include 2x2 max pooling.", ha="center", va="top", fontsize=7.4, color=MUTED)

    # Repeated gate features
    ax.plot([0.680, 0.680], [0.440, 0.805], color=LINE, linewidth=0.7, alpha=0.35)
    ax.text(0.710, 0.835, "multi-slice fusion head", ha="left", va="center", fontsize=9.5, fontweight="bold", color=INK)
    draw_gate_feature(ax, 0.715, 0.735, "f0", BLUE)
    draw_gate_feature(ax, 0.715, 0.640, "f1", GOLD)
    draw_gate_feature(ax, 0.715, 0.545, "f2", GREEN)
    ax.text(0.760, 0.505, "same encoder for all gates", ha="center", va="top", fontsize=7.2, color=MUTED)

    box(ax, 0.835, 0.710, 0.108, 0.085, "attention\nMLP", BLUE_L, BLUE, fontsize=7.8)
    ax.text(0.889, 0.690, "128 -> 64 -> 1; softmax", ha="center", va="top", fontsize=6.8, color=BLUE)
    box(ax, 0.835, 0.525, 0.125, 0.100, "concat residual\nprojection", GREEN_L, GREEN, fontsize=7.4)
    ax.text(0.898, 0.505, "[f0,f1,f2] -> 128", ha="center", va="top", fontsize=6.8, color=GREEN)

    for y, c, rad in [(0.761, BLUE, 0.05), (0.666, BLUE, 0.00), (0.571, BLUE, -0.05)]:
        arrow(ax, (0.805, y), (0.835, 0.752), color=c, lw=0.9, rad=rad)
    for y, c, rad in [(0.761, GREEN, -0.20), (0.666, GREEN, -0.10), (0.571, GREEN, 0.00)]:
        arrow(ax, (0.805, y), (0.835, 0.575), color=c, lw=0.9, rad=rad)

    ax.add_patch(patches.Circle((0.910, 0.415), 0.025, facecolor="white", edgecolor=LINE, linewidth=0.9))
    ax.text(0.910, 0.415, "+", ha="center", va="center", fontsize=13, fontweight="bold", color=INK)
    arrow(ax, (0.890, 0.710), (0.905, 0.440), color=BLUE, lw=0.9)
    arrow(ax, (0.895, 0.525), (0.905, 0.440), color=GREEN, lw=0.9)
    box(ax, 0.855, 0.275, 0.125, 0.078, "classifier\n128 -> 64 -> 6", GOLD_L, GOLD, fontsize=7.4)
    arrow(ax, (0.910, 0.390), (0.910, 0.353), color=INK, lw=0.9)
    ax.text(0.918, 0.254, "chair, desk, sofa, bed, toilet, image2d", ha="center", va="top", fontsize=6.8, color=MUTED)

    # Small compatibility note
    note = (
        "NN-SVG LeNet page can reproduce the sequential CNN encoder directly. "
        "The residual/attention head is drawn here as a compatible schematic extension."
    )
    ax.text(0.030, 0.060, note, ha="left", va="bottom", fontsize=7.2, color=MUTED)

    save_all(fig, "slice_attention_residual_nnsvg_style")
    plt.close(fig)


def write_config() -> None:
    config = """# NN-SVG LeNet-style configuration for SliceEncoder

Source model:

```text
SliceEncoder:
  input: 1 x 224 x 224
  Conv2d(1, 32, kernel=3, padding=1) + BN + ReLU + MaxPool2d(2)
  Conv2d(32, 64, kernel=3, padding=1) + BN + ReLU + MaxPool2d(2)
  Conv2d(64, 128, kernel=3, padding=1) + BN + ReLU + AdaptiveAvgPool2d(1)
  Flatten + Linear(128, 128) + ReLU
```

Recommended NN-SVG LeNet rows:

| Depth | Height | Width | Filter height | Filter width | Op label |
|---:|---:|---:|---:|---:|---|
| 1 | 224 | 224 | 3 | 3 | Conv 3x3 + BN + ReLU |
| 32 | 112 | 112 | 2 | 2 | MaxPool 2x2 |
| 64 | 56 | 56 | 2 | 2 | MaxPool 2x2 |
| 128 | 1 | 1 | 1 | 1 | AdaptiveAvgPool |

Vector rows:

| Vector length | Meaning |
|---:|---|
| 128 | Flatten + Linear feature f_i |
| 64 | Classifier hidden layer |
| 6 | Output classes |

Practical note:

NN-SVG's LeNet page is best for the sequential CNN encoder. It does not natively
represent the three-slice attention branch or the attention-residual skip/add path,
so the generated project figure adds those parts as schematic modules around the
NN-SVG-style encoder.
"""
    (FIGURE_DIR / "slice_attention_residual_nnsvg_config.md").write_text(config, encoding="utf-8")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    draw_network()
    write_config()
    for suffix in ["svg", "pdf", "png", "tiff"]:
        print(FIGURE_DIR / f"slice_attention_residual_nnsvg_style.{suffix}")
    print(FIGURE_DIR / "slice_attention_residual_nnsvg_config.md")


if __name__ == "__main__":
    main()

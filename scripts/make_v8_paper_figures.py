"""Generate core manuscript figures for the v8 gated-imaging framework."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "writing" / "figures"
RAW_DATASET = ROOT / "dataset_new" / "Military_TrueFalse_Selected44_blender_refl_overlap_w15_m012_v8"
SAMPLE_ID = "02_Fighter_Jet__02_Fighter_Jet_02_Fighter_Jet_044_95fe1170"
RAW_SHORTCUT_CSV = (
    ROOT
    / "dataset_new"
    / "military_true_false_selected44_blender_refl_overlap_w15_m012_v8_single_gate_feature_separability.csv"
)
NORM_SHORTCUT_CSV = (
    ROOT
    / "dataset_new"
    / "military_true_false_selected44_blender_refl_overlap_w15_m012_v8_per_gate_maxnorm_single_gate_feature_separability.csv"
)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Calibri"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_box(ax: plt.Axes, xy: tuple[float, float], text: str, color: str) -> None:
    x, y = xy
    box = patches.FancyBboxPatch(
        (x, y),
        1.72,
        0.58,
        boxstyle="round,pad=0.025,rounding_size=0.03",
        linewidth=0.8,
        edgecolor="#222222",
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(x + 0.86, y + 0.29, text, ha="center", va="center", fontsize=7)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", linewidth=0.9, color="#333333", shrinkA=2, shrinkB=2),
    )


def make_pipeline_figure() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.2), constrained_layout=True)
    ax.set_xlim(0, 7.85)
    ax.set_ylim(0, 3.45)
    ax.axis("off")

    blue = "#56B4E9"
    green = "#009E73"
    orange = "#E69F00"
    purple = "#CC79A7"
    gray = "#E8E8E8"

    draw_box(ax, (0.15, 2.35), "Selected 3D\nmilitary models", blue)
    draw_box(ax, (2.1, 2.35), "Blender depth\nand reflectance", blue)
    draw_box(ax, (4.05, 2.35), "Rectangular\nlaser-gate response", green)
    draw_box(ax, (5.98, 2.35), "True 3D\ngate stack", green)

    draw_box(ax, (0.15, 1.28), "Same visible\nsilhouette", gray)
    draw_box(ax, (2.1, 1.28), "Camera-depth\nflattening", orange)
    draw_box(ax, (4.05, 1.28), "Planar false-target\nresponse", orange)
    draw_box(ax, (5.98, 1.28), "Flat false\ngate stack", orange)

    draw_box(ax, (0.15, 0.22), "Anti-shortcut\ncontrols", purple)
    draw_box(ax, (2.1, 0.22), "Single-gate\nablations", purple)
    draw_box(ax, (4.05, 0.22), "Mixed clean/noisy\nevaluation", purple)
    draw_box(ax, (5.98, 0.22), "Robustness\nclaim boundary", purple)

    for y in [2.64, 1.57, 0.51]:
        arrow(ax, (1.87, y), (2.08, y))
        arrow(ax, (3.82, y), (4.03, y))
        arrow(ax, (5.77, y), (5.96, y))

    arrow(ax, (4.91, 2.33), (4.91, 1.88))
    arrow(ax, (4.91, 1.26), (4.91, 0.81))
    arrow(ax, (6.84, 1.26), (6.84, 0.81))

    ax.text(0.15, 3.32, "A", fontsize=10, fontweight="bold", va="top")
    ax.text(0.42, 3.31, "Simulation and validation framework", fontsize=9, va="top")
    save_figure(fig, "fig1_gated_imaging_framework")


def rect_overlap(delta: np.ndarray, laser_width: float, gate_width: float) -> np.ndarray:
    left = np.maximum(delta - laser_width / 2.0, -gate_width / 2.0)
    right = np.minimum(delta + laser_width / 2.0, gate_width / 2.0)
    overlap = np.maximum(0.0, right - left)
    return overlap / min(laser_width, gate_width)


def make_overlap_figure() -> None:
    delta = np.linspace(-1.4, 1.4, 900)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), constrained_layout=True)

    configs = [
        ("Equal pulse/gate widths", 1.0, 1.0, "#0072B2", "-"),
        ("Unequal widths (v8 setting)", 0.45, 1.50, "#D55E00", "--"),
    ]
    for label, laser_width, gate_width, color, linestyle in configs:
        axes[0].plot(
            delta,
            rect_overlap(delta, laser_width, gate_width),
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=1.7,
        )
    axes[0].set_xlabel("Relative delay")
    axes[0].set_ylabel("Normalized response")
    axes[0].set_ylim(-0.03, 1.08)
    axes[0].legend(frameon=False, loc="lower right")
    axes[0].text(0.02, 0.22, "triangular", transform=axes[0].transAxes, color="#0072B2", fontsize=7)
    axes[0].text(0.38, 0.88, "trapezoidal plateau", transform=axes[0].transAxes, color="#D55E00", fontsize=7)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].grid(axis="y", alpha=0.18, linewidth=0.6)
    axes[0].text(-0.18, 1.05, "A", transform=axes[0].transAxes, fontsize=10, fontweight="bold")

    laser_width = 0.45
    gate_width = 1.50
    gate_centers = np.array([-0.90, 0.00, 0.90])
    target_depths = {
        "planar false target": np.array([0.00]),
        "3D target surfaces": np.array([-0.55, -0.10, 0.38, 0.74]),
    }
    x = np.arange(3)
    false_response = [
        float(rect_overlap(np.array([d - c]), laser_width, gate_width)[0])
        for c in gate_centers
        for d in target_depths["planar false target"]
    ]
    true_response = []
    for c in gate_centers:
        vals = rect_overlap(target_depths["3D target surfaces"] - c, laser_width, gate_width)
        true_response.append(float(vals.mean()))
    width = 0.32
    axes[1].bar(x - width / 2, true_response, width, color="#0072B2", edgecolor="black", linewidth=0.5, label="3D target")
    axes[1].bar(x + width / 2, false_response, width, color="#E69F00", edgecolor="black", linewidth=0.5, hatch="//", label="planar false")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Gate 0", "Gate 1", "Gate 2"])
    axes[1].set_ylim(0, 1.08)
    axes[1].set_ylabel("Sampled response")
    axes[1].legend(frameon=False, loc="upper right")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].grid(axis="y", alpha=0.18, linewidth=0.6)
    axes[1].text(-0.18, 1.05, "B", transform=axes[1].transAxes, fontsize=10, fontweight="bold")

    save_figure(fig, "fig2_rectangular_overlap_response")


def image_path(class_name: str, gate_idx: int) -> Path:
    return RAW_DATASET / class_name / f"{SAMPLE_ID}_gate_{gate_idx}.png"


def load_gate_images() -> dict[str, list[np.ndarray]]:
    images: dict[str, list[np.ndarray]] = {}
    for class_name in ["true3d", "flat_false"]:
        row = []
        for gate_idx in range(3):
            path = image_path(class_name, gate_idx)
            if not path.exists():
                raise FileNotFoundError(path)
            img = Image.open(path).convert("L")
            row.append(np.asarray(img, dtype=np.float32))
        images[class_name] = row
    return images


def make_gate_example_figure() -> None:
    images = load_gate_images()
    all_pixels = np.concatenate([img.ravel() for row in images.values() for img in row])
    vmax = float(np.percentile(all_pixels, 99.8))
    fig, axes = plt.subplots(2, 3, figsize=(6.2, 3.8), constrained_layout=True)
    titles = ["Gate 0", "Gate 1", "Gate 2"]
    row_labels = [("true3d", "True 3D target"), ("flat_false", "Planar false target")]

    for r, (class_name, label) in enumerate(row_labels):
        for c, title in enumerate(titles):
            ax = axes[r, c]
            ax.imshow(images[class_name][c], cmap="gray", vmin=0, vmax=vmax)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, rotation=90, labelpad=10)
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)

    axes[0, 0].text(-0.18, 1.15, "A", transform=axes[0, 0].transAxes, fontsize=10, fontweight="bold")
    axes[1, 0].text(-0.18, 1.15, "B", transform=axes[1, 0].transAxes, fontsize=10, fontweight="bold")
    save_figure(fig, "fig3_true3d_flatfalse_gate_examples")


def load_best_shortcuts(csv_path: Path) -> dict[int, dict[str, str]]:
    best: dict[int, dict[str, str]] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gate = int(row["gate"])
            current = best.get(gate)
            if current is None or float(row["best_accuracy"]) > float(current["best_accuracy"]):
                best[gate] = row
    return best


def make_scalar_shortcut_figure() -> None:
    raw = load_best_shortcuts(RAW_SHORTCUT_CSV)
    norm = load_best_shortcuts(NORM_SHORTCUT_CSV)
    gates = [0, 1, 2]
    x = np.arange(len(gates))
    width = 0.34

    fig, ax = plt.subplots(figsize=(5.4, 2.9), constrained_layout=True)
    raw_acc = [float(raw[g]["best_accuracy"]) for g in gates]
    norm_acc = [float(norm[g]["best_accuracy"]) for g in gates]
    raw_features = [raw[g]["feature"] for g in gates]
    norm_features = [norm[g]["feature"] for g in gates]

    ax.bar(
        x - width / 2,
        raw_acc,
        width,
        color="#D55E00",
        edgecolor="black",
        linewidth=0.5,
        label="raw v8",
    )
    ax.bar(
        x + width / 2,
        norm_acc,
        width,
        color="#0072B2",
        edgecolor="black",
        linewidth=0.5,
        hatch="//",
        label="per-gate maxnorm",
    )

    for i, (rv, nv) in enumerate(zip(raw_acc, norm_acc)):
        ax.text(x[i] - width / 2, rv + 0.015, f"{rv:.2f}\n{raw_features[i]}", ha="center", va="bottom", fontsize=6)
        ax.text(x[i] + width / 2, nv + 0.015, f"{nv:.2f}\n{norm_features[i]}", ha="center", va="bottom", fontsize=6)

    ax.axhline(0.5, color="#555555", linewidth=0.8, linestyle=":", label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Gate {g}" for g in gates])
    ax.set_ylim(0.45, 1.08)
    ax.set_ylabel("Best scalar threshold accuracy")
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.18, linewidth=0.6)
    save_figure(fig, "fig4_scalar_shortcut_control")


def write_manifest() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = OUT_DIR / "source_manifest_v8_paper_figures.csv"
    rows = [
        {
            "figure": "fig1_gated_imaging_framework",
            "source": "manuscript method design",
            "note": "framework schematic",
        },
        {
            "figure": "fig2_rectangular_overlap_response",
            "source": "rectangular pulse/gate overlap model",
            "note": "laser_width=0.45, receiver_gate_width=1.50 for v8 illustration",
        },
        {
            "figure": "fig3_true3d_flatfalse_gate_examples",
            "source": str(RAW_DATASET.relative_to(ROOT)),
            "note": SAMPLE_ID,
        },
        {
            "figure": "fig4_scalar_shortcut_control",
            "source": "single-gate scalar shortcut diagnostics",
            "note": "raw v8 versus per-gate max-normalized control",
        },
    ]
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["figure", "source", "note"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    configure_style()
    make_pipeline_figure()
    make_overlap_figure()
    make_gate_example_figure()
    make_scalar_shortcut_figure()
    write_manifest()
    for path in sorted(OUT_DIR.glob("fig[1234]_*.png")):
        print(path.relative_to(ROOT))
    for path in sorted(OUT_DIR.glob("fig[1234]_*.pdf")):
        print(path.relative_to(ROOT))
    print((OUT_DIR / "source_manifest_v8_paper_figures.csv").relative_to(ROOT))


if __name__ == "__main__":
    main()

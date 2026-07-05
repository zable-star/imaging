from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT / "experiments"
OUTPUT_DIR = ROOT / "artifacts" / "figures"
SPACING_ORDER = ["small", "default", "large"]
SPACING_LABELS = {"small": "Small\n0.45", "default": "Default\n0.60", "large": "Large\n0.90"}
GATE_COLORS = ["#5A7FA5", "#D49A3A", "#6FA06F"]
BAR_COLOR = "#6B7C93"
POINT_COLOR = "#2C3440"
ACCENT = "#B35C44"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def read_summary_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for summary_path in sorted(EXPERIMENT_ROOT.glob("phys_gate_spacing_*_attention_residual_seed*/summary.json")):
        run_name = summary_path.parent.name
        experiment = run_name.rsplit("_seed", 1)[0]
        spacing = experiment.replace("phys_gate_spacing_", "").replace("_attention_residual", "")
        if spacing not in SPACING_ORDER:
            continue
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        rows.append(
            {
                "spacing": spacing,
                "seed": int(summary["seed"]),
                "best_val_acc": float(summary["best_val_acc"]),
                "num_samples": int(summary["num_samples"]),
                "classes": " ".join(summary["classes"]),
            }
        )
    return rows


def read_attention_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for csv_path in sorted(EXPERIMENT_ROOT.glob("phys_gate_spacing_*_attention_residual_seed*/val_attention_weights.csv")):
        run_name = csv_path.parent.name
        experiment = run_name.rsplit("_seed", 1)[0]
        spacing = experiment.replace("phys_gate_spacing_", "").replace("_attention_residual", "")
        if spacing not in SPACING_ORDER:
            continue
        seed = int(run_name.rsplit("_seed", 1)[1])
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(
                    {
                        "spacing": spacing,
                        "seed": seed,
                        "class_name": row["class_name"],
                        "attn_gate_0": float(row["attn_gate_0"]),
                        "attn_gate_1": float(row["attn_gate_1"]),
                        "attn_gate_2": float(row["attn_gate_2"]),
                    }
                )
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def aggregate_accuracy(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {spacing: [] for spacing in SPACING_ORDER}
    for row in rows:
        grouped[str(row["spacing"])].append(row)
    aggregate = {}
    for spacing, spacing_rows in grouped.items():
        values = [float(row["best_val_acc"]) for row in spacing_rows]
        seeds = [int(row["seed"]) for row in spacing_rows]
        aggregate[spacing] = {
            "values": values,
            "seeds": seeds,
            "mean": mean(values),
            "std": sample_std(values),
        }
    return aggregate


def aggregate_attention(rows: list[dict[str, object]]) -> dict[str, list[float]]:
    aggregate = {}
    for spacing in SPACING_ORDER:
        spacing_rows = [row for row in rows if row["spacing"] == spacing]
        aggregate[spacing] = [
            mean([float(row[f"attn_gate_{gate_idx}"]) for row in spacing_rows])
            for gate_idx in range(3)
        ]
    return aggregate


def write_source_csv(acc: dict[str, dict[str, object]], attn: dict[str, list[float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "gate_spacing",
                "mean_acc",
                "std_acc",
                "seed_acc_values",
                "mean_attn_gate_0",
                "mean_attn_gate_1",
                "mean_attn_gate_2",
            ],
        )
        writer.writeheader()
        for spacing in SPACING_ORDER:
            writer.writerow(
                {
                    "gate_spacing": spacing,
                    "mean_acc": acc[spacing]["mean"],
                    "std_acc": acc[spacing]["std"],
                    "seed_acc_values": " ".join(f"{value:.4f}" for value in acc[spacing]["values"]),
                    "mean_attn_gate_0": attn[spacing][0],
                    "mean_attn_gate_1": attn[spacing][1],
                    "mean_attn_gate_2": attn[spacing][2],
                }
            )


def draw_figure(acc: dict[str, dict[str, object]], attn: dict[str, list[float]]) -> plt.Figure:
    fig = plt.figure(figsize=(7.05, 3.05))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.08, 1.0], wspace=0.34)
    ax_acc = fig.add_subplot(gs[0, 0])
    ax_attn = fig.add_subplot(gs[0, 1])

    x = np.arange(len(SPACING_ORDER))
    means = np.array([acc[spacing]["mean"] for spacing in SPACING_ORDER])
    stds = np.array([acc[spacing]["std"] for spacing in SPACING_ORDER])

    ax_acc.bar(x, means * 100, color=BAR_COLOR, width=0.56, edgecolor="#2C3440", linewidth=0.6, zorder=2)
    ax_acc.errorbar(x, means * 100, yerr=stds * 100, fmt="none", ecolor="#2C3440", elinewidth=0.8, capsize=2.5, zorder=3)
    offsets = [-0.10, 0.0, 0.10]
    for idx, spacing in enumerate(SPACING_ORDER):
        for offset, value in zip(offsets, acc[spacing]["values"]):
            ax_acc.scatter(idx + offset, value * 100, s=14, color=POINT_COLOR, linewidth=0, zorder=4)
    ax_acc.plot(x, means * 100, color=ACCENT, linewidth=1.0, marker="o", markersize=3.0, zorder=5)
    ax_acc.set_xticks(x, [SPACING_LABELS[spacing] for spacing in SPACING_ORDER])
    ax_acc.set_ylabel("Best validation accuracy (%)")
    ax_acc.set_ylim(88, 99)
    ax_acc.set_yticks([88, 90, 92, 94, 96, 98])
    ax_acc.grid(axis="y", color="#D8DEE8", linewidth=0.55, zorder=1)
    ax_acc.text(-0.18, 1.04, "a", transform=ax_acc.transAxes, fontweight="bold", fontsize=9)
    ax_acc.set_title("Gate spacing changes recognition accuracy", loc="left", fontsize=8, pad=5)

    bottom = np.zeros(len(SPACING_ORDER))
    for gate_idx, color in enumerate(GATE_COLORS):
        values = np.array([attn[spacing][gate_idx] for spacing in SPACING_ORDER]) * 100
        ax_attn.bar(
            x,
            values,
            bottom=bottom,
            color=color,
            width=0.58,
            edgecolor="white",
            linewidth=0.5,
            label=f"gate_{gate_idx}",
        )
        bottom += values
    ax_attn.set_xticks(x, [SPACING_LABELS[spacing] for spacing in SPACING_ORDER])
    ax_attn.set_ylabel("Mean attention contribution (%)")
    ax_attn.set_ylim(0, 100)
    ax_attn.set_yticks([0, 25, 50, 75, 100])
    ax_attn.text(-0.18, 1.04, "b", transform=ax_attn.transAxes, fontweight="bold", fontsize=9)
    ax_attn.set_title("Gate contribution becomes more balanced", loc="left", fontsize=8, pad=5)
    ax_attn.legend(loc="upper center", bbox_to_anchor=(0.50, -0.17), ncol=3, handlelength=1.1, columnspacing=1.0)

    return fig


def save_figure(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def main() -> int:
    configure_style()
    acc_rows = read_summary_rows()
    attn_rows = read_attention_rows()
    if not acc_rows:
        raise FileNotFoundError("No phys_gate_spacing summary.json files found.")
    if not attn_rows:
        raise FileNotFoundError("No phys_gate_spacing val_attention_weights.csv files found.")
    acc = aggregate_accuracy(acc_rows)
    attn = aggregate_attention(attn_rows)
    out_base = OUTPUT_DIR / "physical_gate_spacing_ablation"
    write_source_csv(acc, attn, OUTPUT_DIR / "physical_gate_spacing_ablation_source.csv")
    fig = draw_figure(acc, attn)
    save_figure(fig, out_base)
    plt.close(fig)
    print(f"Wrote {out_base.with_suffix('.svg')}")
    print(f"Wrote {out_base.with_suffix('.pdf')}")
    print(f"Wrote {out_base.with_suffix('.png')}")
    print(f"Wrote {out_base.with_suffix('.tiff')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

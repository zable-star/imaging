"""Generate the v8 multiview full-stack fusion robustness figure."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "experiments" / "v8_mv4_norm_mixaug_full_fusion_eval_aggregate_3seed.csv"
OUT_DIR = ROOT / "writing" / "figures"


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


def read_rows() -> list[dict[str, str]]:
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def make_figure() -> None:
    rows = read_rows()
    by_condition: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_condition[row["condition"]][row["fusion_mode"]] = row

    condition_order = ["clean", "light_noise_g002_p80_b002", "strong_noise_g005_p30_b005"]
    condition_labels = ["Clean", "Light noise", "Strong noise"]
    fusion_order = ["attention", "mean", "attention_residual"]
    fusion_labels = ["Attention", "Mean", "Attention + residual"]
    colors = ["#0072B2", "#E69F00", "#CC79A7"]
    hatches = ["", "//", "xx"]

    x = np.arange(len(condition_order))
    width = 0.24

    fig, ax = plt.subplots(figsize=(6.4, 3.1), constrained_layout=True)
    for idx, fusion_mode in enumerate(fusion_order):
        means = [float(by_condition[condition][fusion_mode]["mean_acc"]) for condition in condition_order]
        stds = [float(by_condition[condition][fusion_mode]["std_acc"]) for condition in condition_order]
        offset = (idx - 1) * width
        bars = ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=2.5,
            color=colors[idx],
            edgecolor="black",
            linewidth=0.5,
            hatch=hatches[idx],
            label=fusion_labels[idx],
        )
        for bar, value, err in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                min(value + err + 0.014, 1.075),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=6,
            )

    ax.axhline(0.5, color="#555555", linewidth=0.8, linestyle=":", label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(condition_labels)
    ax.set_ylim(0.45, 1.12)
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Four-view v8 validation: full-stack fusion comparison")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.18, linewidth=0.6)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig8_mv4_full_stack_fusion_robustness.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / "fig8_mv4_full_stack_fusion_robustness.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    configure_style()
    make_figure()
    print((OUT_DIR / "fig8_mv4_full_stack_fusion_robustness.png").relative_to(ROOT))
    print((OUT_DIR / "fig8_mv4_full_stack_fusion_robustness.pdf").relative_to(ROOT))


if __name__ == "__main__":
    main()

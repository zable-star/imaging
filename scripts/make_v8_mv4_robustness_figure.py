"""Generate the v8 multiview full-stack versus single-gate robustness figure."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "experiments" / "v8_mv4_norm_mixaug_attention_eval_aggregate_3seed.csv"
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
        by_condition[row["condition"]][row["input_variant"]] = row

    condition_order = ["clean", "light_noise_g002_p80_b002", "strong_noise_g005_p30_b005"]
    condition_labels = ["Clean", "Light noise", "Strong noise"]
    input_order = ["full", "gate0", "gate1", "gate2"]
    input_labels = ["3-gate stack", "Gate 0", "Gate 1", "Gate 2"]
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
    hatches = ["", "//", "\\\\", "xx"]

    x = np.arange(len(condition_order))
    width = 0.19

    fig, ax = plt.subplots(figsize=(6.6, 3.2), constrained_layout=True)
    for idx, input_variant in enumerate(input_order):
        means = [float(by_condition[condition][input_variant]["mean_acc"]) for condition in condition_order]
        stds = [float(by_condition[condition][input_variant]["std_acc"]) for condition in condition_order]
        offset = (idx - 1.5) * width
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
            label=input_labels[idx],
        )
        for bar, value, err in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                min(value + err + 0.014, 1.075),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=6,
                rotation=0,
            )

    ax.axhline(0.5, color="#555555", linewidth=0.8, linestyle=":", label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(condition_labels)
    ax.set_ylim(0.45, 1.12)
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Four-view v8 validation: full gate stack versus single-gate inputs")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.18, linewidth=0.6)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "pdf"]:
        fig.savefig(
            OUT_DIR / f"fig7_mv4_full_stack_vs_single_gate_robustness.{suffix}",
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    configure_style()
    make_figure()
    print((OUT_DIR / "fig7_mv4_full_stack_vs_single_gate_robustness.png").relative_to(ROOT))
    print((OUT_DIR / "fig7_mv4_full_stack_vs_single_gate_robustness.pdf").relative_to(ROOT))


if __name__ == "__main__":
    main()

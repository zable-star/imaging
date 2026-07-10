from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CONDITION_ORDER = [
    "clean",
    "light_noise_g002_p80_b002",
    "strong_noise_g005_p30_b005",
    "hard_nuisance_v3_mild",
    "hard_nuisance_v2",
]

CONDITION_LABELS = {
    "clean": "Clean",
    "light_noise_g002_p80_b002": "Light noise",
    "strong_noise_g005_p30_b005": "Strong noise",
    "hard_nuisance_v3_mild": "Hard mild",
    "hard_nuisance_v2": "Hard strong",
}

INPUT_ORDER = ["full", "gate0", "gate1", "gate2"]
INPUT_LABELS = {
    "full": "Full stack",
    "gate0": "Gate 0",
    "gate1": "Gate 1",
    "gate2": "Gate 2",
}

COLORS = {
    "full": "#1f77b4",
    "gate0": "#ff7f0e",
    "gate1": "#2ca02c",
    "gate2": "#9467bd",
}

FIELDNAMES = [
    "input_variant",
    "condition",
    "condition_label",
    "num_runs",
    "mean_acc",
    "std_acc",
    "min_acc",
    "max_acc",
    "seeds",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def normalize_row(row: dict[str, str]) -> dict[str, str]:
    condition = row["condition"]
    return {
        "input_variant": row["input_variant"],
        "condition": condition,
        "condition_label": CONDITION_LABELS.get(condition, condition),
        "num_runs": row["num_runs"],
        "mean_acc": row["mean_acc"],
        "std_acc": row["std_acc"],
        "min_acc": row["min_acc"],
        "max_acc": row["max_acc"],
        "seeds": row["seeds"],
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    by_key = {(r["input_variant"], r["condition"]): r for r in rows}
    lines = [
        "# Domain-mixed full-stack versus single-gate ablation",
        "",
        "All values are mean grouped-validation accuracies over seeds 42, 332, and 2026.",
        "",
        "| condition | full stack | gate0 | gate1 | gate2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITION_ORDER:
        values = []
        for input_variant in INPUT_ORDER:
            row = by_key[(input_variant, condition)]
            values.append(f"{float(row['mean_acc']):.4f}")
        lines.append(f"| {CONDITION_LABELS[condition]} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The full three-gate stack is the best mean performer under all five evaluation conditions.",
            "- Gate 0 is the strongest single-gate baseline under strong-noise and hard-strong evaluation.",
            "- Gate 1 is consistently weak in the domain-mixed single-gate setting.",
            "- The result supports the manuscript claim that the domain-randomized gate stack retains information beyond the best individual gate.",
            "",
            "## Files",
            "",
            "- Combined CSV: `experiments/v8_mv4_domainmix_full_vs_single_gate_eval_aggregate_3seed.csv`",
            "- Figure PNG: `writing/figures/fig11_domainmix_full_stack_vs_single_gate.png`",
            "- Figure PDF: `writing/figures/fig11_domainmix_full_stack_vs_single_gate.pdf`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_figure(path_png: Path, path_pdf: Path, rows: list[dict[str, str]]) -> None:
    by_key = {(r["input_variant"], r["condition"]): r for r in rows}
    x = np.arange(len(CONDITION_ORDER))
    width = 0.19
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(INPUT_ORDER))

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for offset, input_variant in zip(offsets, INPUT_ORDER):
        means = [float(by_key[(input_variant, c)]["mean_acc"]) for c in CONDITION_ORDER]
        stds = [float(by_key[(input_variant, c)]["std_acc"]) for c in CONDITION_ORDER]
        ax.bar(
            x + offset,
            means,
            width,
            label=INPUT_LABELS[input_variant],
            color=COLORS[input_variant],
            yerr=stds,
            capsize=3,
            linewidth=0.6,
            edgecolor="black",
        )

    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER], rotation=15, ha="right")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.11), frameon=False)
    fig.tight_layout()

    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=300)
    fig.savefig(path_pdf)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-aggregate",
        type=Path,
        default=Path("experiments/v8_mv4_domainmix_norm_hardv3_strongaug_attention_full_eval_aggregate_3seed.csv"),
    )
    parser.add_argument(
        "--single-aggregate",
        type=Path,
        default=Path("experiments/v8_mv4_domainmix_single_gate_eval_aggregate_3seed.csv"),
    )
    parser.add_argument(
        "--combined-csv",
        type=Path,
        default=Path("experiments/v8_mv4_domainmix_full_vs_single_gate_eval_aggregate_3seed.csv"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("writing/v8_mv4_domainmix_full_vs_single_gate_ablation_report_2026-07-09.md"),
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=Path("writing/figures/fig11_domainmix_full_stack_vs_single_gate.png"),
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path("writing/figures/fig11_domainmix_full_stack_vs_single_gate.pdf"),
    )
    args = parser.parse_args()

    rows = [normalize_row(r) for r in read_rows(args.full_aggregate) + read_rows(args.single_aggregate)]
    rows = [
        r
        for r in rows
        if r["input_variant"] in INPUT_ORDER and r["condition"] in CONDITION_ORDER
    ]
    rows.sort(key=lambda r: (CONDITION_ORDER.index(r["condition"]), INPUT_ORDER.index(r["input_variant"])))

    expected = len(CONDITION_ORDER) * len(INPUT_ORDER)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} rows, got {len(rows)}")

    write_csv(args.combined_csv, rows)
    write_markdown(args.markdown, rows)
    plot_figure(args.png, args.pdf, rows)
    print(f"combined_csv={args.combined_csv}")
    print(f"markdown={args.markdown}")
    print(f"png={args.png}")
    print(f"pdf={args.pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

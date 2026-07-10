"""Compare v8 multiview training strategies under noise and nuisance shifts."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "writing" / "figures"

CONDITIONS = [
    ("clean", "Clean"),
    ("light_noise_g002_p80_b002", "Light\nnoise"),
    ("strong_noise_g005_p30_b005", "Strong\nnoise"),
    ("hard_nuisance_v3_mild", "Hard\nmild"),
    ("hard_nuisance_v2", "Hard\nstrong"),
]

STRATEGIES = [
    (
        "normal_mixaug",
        "Normal mixaug",
        "#0072B2",
        [
            ROOT / "experiments" / "v8_mv4_norm_mixaug_attention_eval_aggregate_3seed.csv",
            ROOT / "experiments" / "v8_mv4_hard_nuisance_v2_eval_aggregate_3seed.csv",
            ROOT / "experiments" / "v8_mv4_hard_nuisance_v3_mild_eval_aggregate_3seed.csv",
        ],
    ),
    (
        "online_nuisance",
        "Online nuisance",
        "#E69F00",
        [
            ROOT / "experiments" / "v8_mv4_norm_nuisanceaware_attention_full_eval_aggregate_3seed.csv",
        ],
    ),
    (
        "domainmix_strongaug",
        "Domain mix + strong noise",
        "#009E73",
        [
            ROOT
            / "experiments"
            / "v8_mv4_domainmix_norm_hardv3_strongaug_attention_full_eval_aggregate_3seed.csv",
        ],
    ),
]


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


def read_strategy_rows(paths: list[Path]) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("input_variant") != "full":
                    continue
                rows[row["condition"]] = row
    return rows


def collect_comparison_rows() -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for strategy_id, strategy_label, _color, paths in STRATEGIES:
        source = read_strategy_rows(paths)
        for condition, condition_label in CONDITIONS:
            row = source.get(condition)
            if row is None:
                continue
            output.append(
                {
                    "strategy_id": strategy_id,
                    "strategy": strategy_label,
                    "condition": condition,
                    "condition_label": condition_label.replace("\n", " "),
                    "num_runs": int(row["num_runs"]),
                    "mean_acc": float(row["mean_acc"]),
                    "std_acc": float(row["std_acc"]),
                    "min_acc": float(row["min_acc"]),
                    "max_acc": float(row["max_acc"]),
                    "seeds": row["seeds"],
                }
            )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "strategy_id",
        "strategy",
        "condition",
        "condition_label",
        "num_runs",
        "mean_acc",
        "std_acc",
        "min_acc",
        "max_acc",
        "seeds",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_figure(rows: list[dict[str, object]]) -> None:
    by_strategy_condition: dict[tuple[str, str], dict[str, object]] = {
        (str(row["strategy_id"]), str(row["condition"])): row for row in rows
    }
    x = np.arange(len(CONDITIONS))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7.2, 3.35), constrained_layout=True)
    for idx, (strategy_id, strategy_label, color, _paths) in enumerate(STRATEGIES):
        means = [
            float(by_strategy_condition[(strategy_id, condition)]["mean_acc"])
            for condition, _label in CONDITIONS
        ]
        stds = [
            float(by_strategy_condition[(strategy_id, condition)]["std_acc"])
            for condition, _label in CONDITIONS
        ]
        bars = ax.bar(
            x + (idx - 1) * width,
            means,
            width,
            yerr=stds,
            capsize=2.5,
            color=color,
            edgecolor="black",
            linewidth=0.45,
            label=strategy_label,
        )
        for bar, value, err in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                min(value + err + 0.016, 1.09),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=5.7,
                rotation=90,
            )

    ax.axhline(0.5, color="#555555", linewidth=0.8, linestyle=":", label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _condition, label in CONDITIONS])
    ax.set_ylim(0.45, 1.12)
    ax.set_ylabel("Mean validation accuracy")
    ax.set_title("Training strategy comparison under noise and hard nuisance shifts")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.18, linewidth=0.6)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig10_domain_randomization_strategy_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig10_domain_randomization_strategy_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_condition: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row["condition"])].append(row)

    lines = [
        "# v8 mv4 domain-randomized training strategy report",
        "",
        "This report compares three training strategies for four-view v8 true-3D versus planar-false-target discrimination.",
        "All reported values are grouped-validation mean accuracies over seeds 42, 332, and 2026.",
        "",
        "## Aggregate accuracy",
        "",
        "| strategy | clean | light noise | strong noise | hard mild | hard strong |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    condition_order = [condition for condition, _label in CONDITIONS]
    for strategy_id, strategy_label, _color, _paths in STRATEGIES:
        strategy_rows = {str(row["condition"]): row for row in rows if row["strategy_id"] == strategy_id}
        values = [fmt(strategy_rows[condition]["mean_acc"]) for condition in condition_order]
        lines.append(f"| {strategy_label} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `Normal mixaug` is the strongest controlled-simulation baseline under clean/light/strong noise, but it collapses to chance under structured hard-nuisance shifts.",
            "- `Online nuisance` applies structured perturbations during training, but the current post-loading implementation does not recover hard-nuisance performance and substantially weakens strong-noise robustness.",
            "- `Domain mix + strong noise` explicitly trains on both normal and hard-mild domains, while retaining strong-noise augmentation. It recovers hard-nuisance performance and keeps strong-noise accuracy near the original baseline.",
            "- The current best manuscript-safe claim is therefore not that the model is generally robust, but that explicit domain-randomized simulation improves the robustness boundary compared with clean/noisy-only training.",
            "",
            "## Main numerical result",
            "",
            "Compared with the normal four-view mixed-noise baseline, `Domain mix + strong noise` changes the mean accuracies as follows:",
            "",
        ]
    )

    baseline = {str(row["condition"]): row for row in rows if row["strategy_id"] == "normal_mixaug"}
    domainmix = {str(row["condition"]): row for row in rows if row["strategy_id"] == "domainmix_strongaug"}
    for condition, label in CONDITIONS:
        delta = float(domainmix[condition]["mean_acc"]) - float(baseline[condition]["mean_acc"])
        lines.append(f"- {label.replace(chr(10), ' ')}: {fmt(delta)} absolute accuracy change.")

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- Comparison CSV: `experiments/v8_mv4_strategy_comparison_aggregate_2026-07-08.csv`",
            "- Figure PNG: `writing/figures/fig10_domain_randomization_strategy_comparison.png`",
            "- Figure PDF: `writing/figures/fig10_domain_randomization_strategy_comparison.pdf`",
            "- Domain-mix dataset: `dataset_new/Military_TF_v8_mv4_norm_plus_hardv3_mild`",
            "- Domain-mix manifest: `dataset_new/Military_TF_v8_mv4_norm_plus_hardv3_mild/variant_mixture_manifest.csv`",
            "",
            "## Caveat",
            "",
            "The hard-nuisance v2 result remains lower than clean/noise validation, and the dataset still contains only 44 selected source military models. This result should be framed as a robustness-boundary improvement in simulation, not real-world deployment robustness.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    configure_style()
    rows = collect_comparison_rows()
    write_csv(ROOT / "experiments" / "v8_mv4_strategy_comparison_aggregate_2026-07-08.csv", rows)
    make_figure(rows)
    write_markdown(ROOT / "writing" / "v8_mv4_domain_randomization_strategy_report_2026-07-08.md", rows)
    print((ROOT / "experiments" / "v8_mv4_strategy_comparison_aggregate_2026-07-08.csv").relative_to(ROOT))
    print((OUT_DIR / "fig10_domain_randomization_strategy_comparison.png").relative_to(ROOT))
    print((OUT_DIR / "fig10_domain_randomization_strategy_comparison.pdf").relative_to(ROOT))
    print((ROOT / "writing" / "v8_mv4_domain_randomization_strategy_report_2026-07-08.md").relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

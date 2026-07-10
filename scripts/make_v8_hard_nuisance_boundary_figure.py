"""Plot the four-view robustness boundary under hard nuisance shifts."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "writing" / "figures"


def read_aggregate(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_acc(rows: list[dict[str, str]], input_variant: str, condition: str) -> float:
    for row in rows:
        if row["input_variant"] == input_variant and row["condition"] == condition:
            return float(row["mean_acc"])
    raise KeyError((input_variant, condition))


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


def main() -> None:
    configure_style()
    normal = read_aggregate(ROOT / "experiments" / "v8_mv4_norm_mixaug_attention_eval_aggregate_3seed.csv")
    hard_v2 = read_aggregate(ROOT / "experiments" / "v8_mv4_hard_nuisance_v2_eval_aggregate_3seed.csv")
    hard_v3 = read_aggregate(ROOT / "experiments" / "v8_mv4_hard_nuisance_v3_mild_eval_aggregate_3seed.csv")

    labels = ["Clean", "Light\nnoise", "Strong\nnoise", "Hard\nmild", "Hard\nstrong"]
    full = [
        get_acc(normal, "full", "clean"),
        get_acc(normal, "full", "light_noise_g002_p80_b002"),
        get_acc(normal, "full", "strong_noise_g005_p30_b005"),
        get_acc(hard_v3, "full", "hard_nuisance_v3_mild"),
        get_acc(hard_v2, "full", "hard_nuisance_v2"),
    ]
    best_single = [
        max(get_acc(normal, gate, "clean") for gate in ["gate0", "gate1", "gate2"]),
        max(get_acc(normal, gate, "light_noise_g002_p80_b002") for gate in ["gate0", "gate1", "gate2"]),
        max(get_acc(normal, gate, "strong_noise_g005_p30_b005") for gate in ["gate0", "gate1", "gate2"]),
        max(get_acc(hard_v3, gate, "hard_nuisance_v3_mild") for gate in ["gate0", "gate1", "gate2"]),
        max(get_acc(hard_v2, gate, "hard_nuisance_v2") for gate in ["gate0", "gate1", "gate2"]),
    ]

    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(6.4, 3.0), constrained_layout=True)
    ax.plot(x, full, marker="o", linewidth=1.8, color="#0072B2", label="3-gate stack")
    ax.plot(x, best_single, marker="s", linewidth=1.5, color="#E69F00", linestyle="--", label="best single gate")
    ax.axhline(0.5, color="#555555", linewidth=0.8, linestyle=":", label="chance")
    for idx, value in enumerate(full):
        ax.text(idx, value + 0.025, f"{value:.2f}", ha="center", va="bottom", fontsize=6, color="#0072B2")
    for idx, value in enumerate(best_single):
        ax.text(idx, value - 0.045, f"{value:.2f}", ha="center", va="top", fontsize=6, color="#8A5A00")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0.35, 1.05)
    ax.set_ylabel("Mean validation accuracy")
    ax.set_title("Four-view v8 robustness boundary under hard nuisance shifts")
    ax.legend(frameon=False, loc="lower left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.18, linewidth=0.6)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig9_hard_nuisance_failure_boundary.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / "fig9_hard_nuisance_failure_boundary.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print((OUT_DIR / "fig9_hard_nuisance_failure_boundary.png").relative_to(ROOT))
    print((OUT_DIR / "fig9_hard_nuisance_failure_boundary.pdf").relative_to(ROOT))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_OVERVIEW = Path("experiments/military_selected44_results_overview_2026-07-06.csv")
DEFAULT_HARD_DIAGNOSTICS = Path("dataset_new/military_true_false_selected44_hard_projection_gate_diagnostics_by_class_2026-07-06.csv")
DEFAULT_RECT_DIAGNOSTICS = Path(
    "dataset_new/military_true_false_selected44_hard_rect_overlap_gate_diagnostics_by_class_2026-07-06.csv"
)
DEFAULT_NORM_DIAGNOSTICS = Path(
    "dataset_new/military_true_false_selected44_gain10_per_gate_norm_gate_diagnostics_by_class_2026-07-06.csv"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/figures/military_selected44_2026-07-06")


@dataclass(frozen=True)
class AggregateRow:
    experiment: str
    mean: float
    std: float
    minimum: float
    maximum: float
    seeds: str
    note: str


@dataclass(frozen=True)
class DiagnosticRow:
    class_name: str
    corr: float
    mask_iou: float
    absdiff: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PPT-ready figures for selected-44 military gated experiments.")
    parser.add_argument("--overview-csv", type=Path, default=DEFAULT_OVERVIEW)
    parser.add_argument("--hard-diagnostics-csv", type=Path, default=DEFAULT_HARD_DIAGNOSTICS)
    parser.add_argument("--rect-diagnostics-csv", type=Path, default=DEFAULT_RECT_DIAGNOSTICS)
    parser.add_argument("--norm-diagnostics-csv", type=Path, default=DEFAULT_NORM_DIAGNOSTICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_overview(path: Path) -> dict[str, AggregateRow]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = {}
        for row in csv.DictReader(f):
            rows[row["experiment"]] = AggregateRow(
                experiment=row["experiment"],
                mean=float(row["mean_best_val_acc"]),
                std=float(row["std_best_val_acc"]),
                minimum=float(row["min_best_val_acc"]),
                maximum=float(row["max_best_val_acc"]),
                seeds=row["seeds"],
                note=row.get("note", ""),
            )
    return rows


def read_diagnostics(path: Path) -> dict[str, DiagnosticRow]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return {
            row["class_name"]: DiagnosticRow(
                class_name=row["class_name"],
                corr=float(row["mean_pair_corr_maxnorm"]),
                mask_iou=float(row["mean_pair_mask_iou"]),
                absdiff=float(row["mean_pair_absdiff_maxnorm"]),
            )
            for row in csv.DictReader(f)
        }


def percent(value: float) -> float:
    return value * 100.0


def require_rows(rows: dict[str, AggregateRow], keys: list[str]) -> list[AggregateRow]:
    missing = [key for key in keys if key not in rows]
    if missing:
        raise KeyError(f"Missing experiments in overview CSV: {missing}")
    return [rows[key] for key in keys]


def style_axes(ax) -> None:
    ax.grid(axis="y", color="#D8DDE3", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#A8B0B8")
    ax.spines["bottom"].set_color("#A8B0B8")
    ax.tick_params(colors="#27313A")


def save_accuracy_bars(
    labels: list[str],
    rows: list[AggregateRow],
    colors: list[str],
    title: str,
    out_path: Path,
    ylim: tuple[float, float],
) -> None:
    means = [percent(row.mean) for row in rows]
    stds = [percent(row.std) for row in rows]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = list(range(len(labels)))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="#263238", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Best validation accuracy (%)")
    ax.set_title(title)
    ax.set_ylim(*ylim)
    style_axes(ax)
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(mean + 1.0, ylim[1] - 0.8),
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#263238",
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_military_transfer(rows: dict[str, AggregateRow], output_dir: Path) -> Path:
    keys = [
        "military_selected44_transfer_frozen_20ep",
        "military_selected44_transfer_finetune_20ep",
        "military_selected44_scratch_20ep",
    ]
    out_path = output_dir / "military_3class_transfer_vs_scratch.png"
    save_accuracy_bars(
        labels=["Transfer\nfrozen", "Transfer\nfinetune", "Scratch"],
        rows=require_rows(rows, keys),
        colors=["#4E79A7", "#59A14F", "#E15759"],
        title="Selected-44 Military 3-Class Recognition",
        out_path=out_path,
        ylim=(45.0, 95.0),
    )
    return out_path


def plot_hard_projection_ablation(rows: dict[str, AggregateRow], output_dir: Path) -> Path:
    keys = [
        "military_truefalse_hard_projection_scratch_20ep",
        "military_truefalse_hard_projection_single_gate0_scratch_10ep",
        "military_truefalse_hard_projection_single_gate1_scratch_10ep",
        "military_truefalse_hard_projection_single_gate2_scratch_10ep",
    ]
    out_path = output_dir / "hard_projection_full_stack_vs_single_gate.png"
    save_accuracy_bars(
        labels=["Full\n3-gate", "Gate 0", "Gate 1", "Gate 2"],
        rows=require_rows(rows, keys),
        colors=["#4E79A7", "#F28E2B", "#F28E2B", "#F28E2B"],
        title="Hard Projection False Target: Full Stack vs Single Gate",
        out_path=out_path,
        ylim=(40.0, 105.0),
    )
    return out_path


def plot_rect_overlap_ablation(rows: dict[str, AggregateRow], output_dir: Path) -> Path:
    keys = [
        "military_truefalse_hard_rect_overlap_scratch_20ep",
        "military_truefalse_hard_rect_overlap_single_gate0_scratch_10ep",
        "military_truefalse_hard_rect_overlap_single_gate1_scratch_10ep",
        "military_truefalse_hard_rect_overlap_single_gate2_scratch_10ep",
    ]
    out_path = output_dir / "hard_rect_overlap_full_stack_vs_single_gate.png"
    save_accuracy_bars(
        labels=["Full\n3-gate", "Gate 0", "Gate 1", "Gate 2"],
        rows=require_rows(rows, keys),
        colors=["#4E79A7", "#F28E2B", "#F28E2B", "#F28E2B"],
        title="Rectangular-Overlap False Target: Full Stack vs Single Gate",
        out_path=out_path,
        ylim=(40.0, 105.0),
    )
    return out_path


def plot_rect_overlap_exposure_matched_ablation(rows: dict[str, AggregateRow], output_dir: Path) -> Path:
    keys = [
        "military_truefalse_hard_rect_overlap_mean_classgate_matched_scratch_20ep",
        "military_truefalse_hard_rect_overlap_mean_classgate_matched_single_gate0_scratch_10ep",
        "military_truefalse_hard_rect_overlap_mean_classgate_matched_single_gate1_scratch_10ep",
        "military_truefalse_hard_rect_overlap_mean_classgate_matched_single_gate2_scratch_10ep",
    ]
    out_path = output_dir / "hard_rect_overlap_exposure_matched_full_stack_vs_single_gate.png"
    save_accuracy_bars(
        labels=["Full\n3-gate", "Gate 0", "Gate 1", "Gate 2"],
        rows=require_rows(rows, keys),
        colors=["#4E79A7", "#F28E2B", "#F28E2B", "#F28E2B"],
        title="Exposure-Matched Rectangular-Overlap False Target",
        out_path=out_path,
        ylim=(40.0, 105.0),
    )
    return out_path


def plot_gate1_residual_controls(rows: dict[str, AggregateRow], output_dir: Path) -> Path:
    keys = [
        "military_truefalse_hard_rect_overlap_single_gate1_scratch_10ep",
        "military_truefalse_hard_rect_overlap_mean_classgate_matched_single_gate1_scratch_10ep",
        "military_truefalse_hard_rect_overlap_foreground_classgate_matched_single_gate1_scratch_10ep",
        "military_truefalse_hard_rect_overlap_p99_classgate_matched_single_gate1_scratch_10ep",
        "military_truefalse_hard_rect_overlap_mean_classgate_matched_scratch_20ep",
    ]
    out_path = output_dir / "hard_rect_overlap_gate1_residual_controls.png"
    save_accuracy_bars(
        labels=["Gate 1\nraw", "Gate 1\nmean matched", "Gate 1\nfg matched", "Gate 1\np99 matched", "Full stack\nmean matched"],
        rows=require_rows(rows, keys),
        colors=["#F28E2B", "#B07AA1", "#B07AA1", "#B07AA1", "#4E79A7"],
        title="Gate-1 Residual Cue Controls",
        out_path=out_path,
        ylim=(40.0, 105.0),
    )
    return out_path


def plot_robustness(rows: dict[str, AggregateRow], output_dir: Path) -> Path:
    keys = [
        "military_truefalse_gain10_per_gate_norm_scratch_20ep",
        "military_truefalse_gain10_per_gate_norm_gate_dropout_random_scratch_10ep",
        "military_truefalse_gain10_per_gate_norm_noise_bg_poisson_scratch_10ep",
    ]
    out_path = output_dir / "per_gate_norm_robustness.png"
    save_accuracy_bars(
        labels=["Clean\nfull stack", "Random\ngate dropout", "Noise + bg\n+ Poisson"],
        rows=require_rows(rows, keys),
        colors=["#4E79A7", "#B07AA1", "#59A14F"],
        title="Per-Gate Normalized True/False Robustness",
        out_path=out_path,
        ylim=(75.0, 105.0),
    )
    return out_path


def plot_diagnostics(
    norm: dict[str, DiagnosticRow],
    hard: dict[str, DiagnosticRow],
    rect: dict[str, DiagnosticRow],
    output_dir: Path,
) -> Path:
    out_path = output_dir / "gate_stack_physical_diagnostics.png"
    labels = [
        "Per-gate norm\nflat false",
        "Per-gate norm\ntrue 3D",
        "Hard projection\nflat false",
        "Hard projection\ntrue 3D",
        "Rect overlap\nflat false",
        "Rect overlap\ntrue 3D",
    ]
    rows = [norm["flat_false"], norm["true3d"], hard["flat_false"], hard["true3d"], rect["flat_false"], rect["true3d"]]
    corr = [row.corr for row in rows]
    iou = [row.mask_iou for row in rows]

    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    width = 0.36
    x = list(range(len(labels)))
    corr_bars = ax.bar(
        [idx - width / 2 for idx in x],
        corr,
        width=width,
        color="#4E79A7",
        edgecolor="#263238",
        linewidth=0.8,
        label="Inter-gate correlation",
    )
    iou_bars = ax.bar(
        [idx + width / 2 for idx in x],
        iou,
        width=width,
        color="#F28E2B",
        edgecolor="#263238",
        linewidth=0.8,
        label="Foreground mask IoU",
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean pairwise score")
    ax.set_title("Gate Stack Physical Diagnostics")
    ax.set_ylim(0.0, 1.08)
    style_axes(ax)
    ax.legend(frameon=False, loc="upper right")
    for bars in [corr_bars, iou_bars]:
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.025,
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#263238",
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    return out_path


def write_figure_summary(output_dir: Path, figure_paths: list[Path], rows: dict[str, AggregateRow]) -> Path:
    out_path = output_dir / "military_selected44_figure_summary.md"
    hard_full = rows["military_truefalse_hard_projection_scratch_20ep"]
    hard_g0 = rows["military_truefalse_hard_projection_single_gate0_scratch_10ep"]
    hard_g1 = rows["military_truefalse_hard_projection_single_gate1_scratch_10ep"]
    hard_g2 = rows["military_truefalse_hard_projection_single_gate2_scratch_10ep"]
    rect_full = rows["military_truefalse_hard_rect_overlap_scratch_20ep"]
    rect_g0 = rows["military_truefalse_hard_rect_overlap_single_gate0_scratch_10ep"]
    rect_g1 = rows["military_truefalse_hard_rect_overlap_single_gate1_scratch_10ep"]
    rect_g2 = rows["military_truefalse_hard_rect_overlap_single_gate2_scratch_10ep"]
    matched_full = rows["military_truefalse_hard_rect_overlap_mean_classgate_matched_scratch_20ep"]
    matched_g0 = rows["military_truefalse_hard_rect_overlap_mean_classgate_matched_single_gate0_scratch_10ep"]
    matched_g1 = rows["military_truefalse_hard_rect_overlap_mean_classgate_matched_single_gate1_scratch_10ep"]
    matched_g2 = rows["military_truefalse_hard_rect_overlap_mean_classgate_matched_single_gate2_scratch_10ep"]
    fg_g1 = rows["military_truefalse_hard_rect_overlap_foreground_classgate_matched_single_gate1_scratch_10ep"]
    p99_g1 = rows["military_truefalse_hard_rect_overlap_p99_classgate_matched_single_gate1_scratch_10ep"]
    out_path.write_text(
        "\n".join(
            [
                "# Military Selected44 Figure Summary",
                "",
                "Generated figures:",
                *[f"- `{path.name}`" for path in figure_paths],
                "",
                "Key PPT statement:",
                "",
                (
                    f"- Hard projection full stack reaches {percent(hard_full.mean):.1f}% mean best validation accuracy, "
                    f"while single-gate inputs are {percent(hard_g0.mean):.1f}%, {percent(hard_g1.mean):.1f}%, "
                    f"and {percent(hard_g2.mean):.1f}%."
                ),
                (
                    f"- With the rectangular pulse-gate overlap response, full stack reaches {percent(rect_full.mean):.1f}%, "
                    f"while single gates are {percent(rect_g0.mean):.1f}%, {percent(rect_g1.mean):.1f}%, "
                    f"and {percent(rect_g2.mean):.1f}%."
                ),
                (
                    f"- After per-gate class-mean exposure matching, full stack remains {percent(matched_full.mean):.1f}%, "
                    f"while single gates are {percent(matched_g0.mean):.1f}%, {percent(matched_g1.mean):.1f}%, "
                    f"and {percent(matched_g2.mean):.1f}%."
                ),
                (
                    f"- Gate 1 remains {percent(fg_g1.mean):.1f}% after foreground-mean matching and "
                    f"{percent(p99_g1.mean):.1f}% after p99 matching, suggesting a residual structural cue."
                ),
                "- This supports the claim that the discriminative cue is the gated sequence response, not ordinary single-frame appearance.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return out_path


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_overview(args.overview_csv)
    hard_diag = read_diagnostics(args.hard_diagnostics_csv)
    rect_diag = read_diagnostics(args.rect_diagnostics_csv)
    norm_diag = read_diagnostics(args.norm_diagnostics_csv)

    figures = [
        plot_military_transfer(rows, args.output_dir),
        plot_hard_projection_ablation(rows, args.output_dir),
        plot_rect_overlap_ablation(rows, args.output_dir),
        plot_rect_overlap_exposure_matched_ablation(rows, args.output_dir),
        plot_gate1_residual_controls(rows, args.output_dir),
        plot_robustness(rows, args.output_dir),
        plot_diagnostics(norm_diag, hard_diag, rect_diag, args.output_dir),
    ]
    summary = write_figure_summary(args.output_dir, figures, rows)

    print(f"output_dir={args.output_dir}")
    for path in figures:
        print(path)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

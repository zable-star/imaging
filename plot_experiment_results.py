from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_EXPERIMENT_ROOT = Path(__file__).resolve().parent / "experiments"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts" / "figures"


@dataclass(frozen=True)
class AggregateResult:
    experiment: str
    mean: float
    std: float
    minimum: float
    maximum: float
    seeds: str


@dataclass(frozen=True)
class RunResult:
    experiment: str
    seed: str
    accuracy: float
    input_mode: str
    gate_index: str
    fusion_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot summary figures from experiment CSV files.")
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_aggregate(path: Path) -> AggregateResult:
    with path.open("r", newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    experiment = row["experiment"]
    if path.parent.name == "six_class_attention_residual_seedmatched":
        experiment = "six_class_attention_residual_seedmatched"
    return AggregateResult(
        experiment=experiment,
        mean=float(row["mean_best_val_acc"]),
        std=float(row["std_best_val_acc"]),
        minimum=float(row["min_best_val_acc"]),
        maximum=float(row["max_best_val_acc"]),
        seeds=row["seeds"],
    )


def read_runs(path: Path) -> list[RunResult]:
    experiment_override = "six_class_attention_residual_seedmatched" if path.parent.name == "six_class_attention_residual_seedmatched" else None
    with path.open("r", newline="", encoding="utf-8") as f:
        return [
            RunResult(
                experiment=experiment_override or row["experiment"],
                seed=row["seed"],
                accuracy=float(row["best_val_acc"]),
                input_mode=row.get("input_mode", ""),
                gate_index=row.get("single_gate_index", ""),
                fusion_mode=row.get("fusion_mode", ""),
            )
            for row in csv.DictReader(f)
        ]


def load_aggregates(experiment_root: Path) -> dict[str, AggregateResult]:
    aggregates = {}
    for path in sorted(experiment_root.rglob("aggregate_results.csv")):
        result = read_aggregate(path)
        aggregates[result.experiment] = result
    return aggregates


def load_runs(experiment_root: Path) -> list[RunResult]:
    runs = []
    for path in sorted(experiment_root.rglob("results.csv")):
        runs.extend(read_runs(path))
    return runs


def as_percent(value: float) -> float:
    return value * 100.0


def save_bar_chart(
    labels: list[str],
    means: list[float],
    stds: list[float],
    colors: list[str],
    title: str,
    ylabel: str,
    out_path: Path,
    ylim: tuple[float, float] = (70.0, 100.0),
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = range(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="#222222", linewidth=0.8)
    ax.set_xticks(list(x), labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.6,
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_fusion_ablation(aggregates: dict[str, AggregateResult], output_dir: Path) -> None:
    items = [
        ("Mean", "six_class_mean", "#8FB9A8"),
        ("Attention", "six_class_attention", "#4F7CAC"),
        ("Concat", "six_class_concat", "#D99058"),
    ]
    residual_key = (
        "six_class_attention_residual_seedmatched"
        if "six_class_attention_residual_seedmatched" in aggregates
        else "six_class_attention_residual"
    )
    if residual_key in aggregates:
        items.insert(2, ("Attention\nResidual", residual_key, "#9E768F"))
    save_bar_chart(
        labels=[label for label, _, _ in items],
        means=[as_percent(aggregates[name].mean) for _, name, _ in items],
        stds=[as_percent(aggregates[name].std) for _, name, _ in items],
        colors=[color for _, _, color in items],
        title="Fusion Ablation on Six-Class Dataset",
        ylabel="Best validation accuracy (%)",
        out_path=output_dir / "fusion_ablation_accuracy.png",
        ylim=(85.0, 100.0),
    )


def plot_single_gate_controls(aggregates: dict[str, AggregateResult], output_dir: Path) -> None:
    items = [
        ("Multi\nattention", "six_class_attention", "#4F7CAC"),
        ("Gate 0", "single_gate_g0", "#8FB9A8"),
        ("Gate 1", "single_gate_g1", "#8FB9A8"),
        ("Gate 2", "single_gate_g2", "#8FB9A8"),
    ]
    save_bar_chart(
        labels=[label for label, _, _ in items],
        means=[as_percent(aggregates[name].mean) for _, name, _ in items],
        stds=[as_percent(aggregates[name].std) for _, name, _ in items],
        colors=[color for _, _, color in items],
        title="Multi-Slice Input vs Single 2D Gate",
        ylabel="Best validation accuracy (%)",
        out_path=output_dir / "single_gate_controls_accuracy.png",
        ylim=(70.0, 100.0),
    )


def plot_black_slice_controls(aggregates: dict[str, AggregateResult], output_dir: Path) -> None:
    labels = ["Gate 0", "Gate 1", "Gate 2"]
    single = [aggregates[f"single_gate_g{i}"] for i in range(3)]
    black = [aggregates[f"single_gate_black_g{i}"] for i in range(3)]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    width = 0.36
    x = list(range(len(labels)))
    single_bars = ax.bar(
        [idx - width / 2 for idx in x],
        [as_percent(item.mean) for item in single],
        yerr=[as_percent(item.std) for item in single],
        width=width,
        capsize=4,
        label="single-gate",
        color="#8FB9A8",
        edgecolor="#222222",
        linewidth=0.8,
    )
    black_bars = ax.bar(
        [idx + width / 2 for idx in x],
        [as_percent(item.mean) for item in black],
        yerr=[as_percent(item.std) for item in black],
        width=width,
        capsize=4,
        label="single-gate-black",
        color="#C1666B",
        edgecolor="#222222",
        linewidth=0.8,
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Best validation accuracy (%)")
    ax.set_title("Single 2D Gate vs Black-Slice Controls")
    ax.set_ylim(70.0, 90.0)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False)
    for bars in [single_bars, black_bars]:
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{bar.get_height():.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(output_dir / "black_slice_controls_accuracy.png", dpi=220)
    plt.close(fig)


def plot_seed_stability(runs: list[RunResult], output_dir: Path) -> None:
    selected = {
        "six_class_attention": "Attention",
        "six_class_mean": "Mean",
        "six_class_concat": "Concat",
        "single_gate_g2": "Single gate 2",
        "single_gate_black_g2": "Black gate 2",
    }
    grouped: dict[str, list[RunResult]] = {name: [] for name in selected}
    for run in runs:
        if run.experiment in grouped:
            grouped[run.experiment].append(run)

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    colors = ["#4F7CAC", "#8FB9A8", "#D99058", "#9E768F", "#C1666B"]
    for color, (experiment, label) in zip(colors, selected.items()):
        sorted_runs = sorted(grouped[experiment], key=lambda item: int(item.seed))
        ax.plot(
            [run.seed for run in sorted_runs],
            [as_percent(run.accuracy) for run in sorted_runs],
            marker="o",
            linewidth=2,
            label=label,
            color=color,
        )
    ax.set_ylabel("Best validation accuracy (%)")
    ax.set_xlabel("Seed")
    ax.set_title("Seed-Level Accuracy Stability")
    ax.set_ylim(70.0, 100.0)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "seed_stability_accuracy.png", dpi=220)
    plt.close(fig)


def write_summary_csv(aggregates: dict[str, AggregateResult], output_dir: Path) -> None:
    order = [
        "six_class_attention",
        "six_class_attention_residual_seedmatched",
        "six_class_attention_residual",
        "six_class_mean",
        "six_class_concat",
        "single_gate_g0",
        "single_gate_g1",
        "single_gate_g2",
        "single_gate_black_g0",
        "single_gate_black_g1",
        "single_gate_black_g2",
    ]
    with (output_dir / "experiment_plot_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment",
                "mean_accuracy_percent",
                "std_accuracy_percent",
                "min_accuracy_percent",
                "max_accuracy_percent",
                "seeds",
            ],
        )
        writer.writeheader()
        for experiment in order:
            if experiment not in aggregates:
                continue
            result = aggregates[experiment]
            writer.writerow(
                {
                    "experiment": experiment,
                    "mean_accuracy_percent": as_percent(result.mean),
                    "std_accuracy_percent": as_percent(result.std),
                    "min_accuracy_percent": as_percent(result.minimum),
                    "max_accuracy_percent": as_percent(result.maximum),
                    "seeds": result.seeds,
                }
            )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    aggregates = load_aggregates(args.experiment_root)
    runs = load_runs(args.experiment_root)

    plot_fusion_ablation(aggregates, args.output_dir)
    plot_single_gate_controls(aggregates, args.output_dir)
    plot_black_slice_controls(aggregates, args.output_dir)
    plot_seed_stability(runs, args.output_dir)
    write_summary_csv(aggregates, args.output_dir)

    print(f"Wrote figures to {args.output_dir}")
    for path in sorted(args.output_dir.glob("*.png")):
        print(path)
    print(args.output_dir / "experiment_plot_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
EVALUATE_SCRIPT = ROOT / "scripts" / "evaluate_gate_model.py"

FUSION_EXPERIMENTS = {
    "attention": "v8_mv4_norm_mixaug_attention_full_20ep",
    "mean": "v8_mv4_norm_mixaug_mean_full_20ep",
    "attention_residual": "v8_mv4_norm_mixaug_attention_residual_full_20ep",
}

CONDITIONS = {
    "clean": {"gaussian": 0.0, "poisson": 0.0, "background": 0.0},
    "light_noise_g002_p80_b002": {"gaussian": 0.02, "poisson": 80.0, "background": 0.02},
    "strong_noise_g005_p30_b005": {"gaussian": 0.05, "poisson": 30.0, "background": 0.05},
}

SUMMARY_FIELDS = [
    "eval_dir",
    "fusion_mode",
    "seed",
    "condition",
    "acc",
    "loss",
    "num_val_samples",
    "summary_path",
]

AGGREGATE_FIELDS = [
    "fusion_mode",
    "condition",
    "num_runs",
    "mean_acc",
    "std_acc",
    "min_acc",
    "max_acc",
    "seeds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate v8 mv4 full-stack fusion modes.")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset_new" / "Military_TF_v8_mv4_norm")
    parser.add_argument("--experiment-root", type=Path, default=ROOT / "experiments")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 332, 2026])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--aggregate-csv", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, default=None)
    parser.add_argument("--use-amp", action="store_true", default=True)
    return parser.parse_args()


def run_eval(args: argparse.Namespace, fusion_mode: str, seed: int, condition: str, params: dict[str, float]) -> Path:
    train_root = FUSION_EXPERIMENTS[fusion_mode]
    run_name = f"{train_root}_seed{seed}"
    model_path = args.experiment_root / train_root / run_name / "slice_attention_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {model_path}")

    artifact_dir = args.experiment_root / f"eval_v8_mv4_fusion_{fusion_mode}_seed{seed}_{condition}"
    command = [
        sys.executable,
        str(EVALUATE_SCRIPT),
        "--dataset-root",
        str(args.dataset_root),
        "--model-path",
        str(model_path),
        "--artifact-dir",
        str(artifact_dir),
        "--input-mode",
        "multi",
        "--single-gate-index",
        "0",
        "--fusion-mode",
        fusion_mode,
        "--batch-size",
        str(args.batch_size),
        "--val-ratio",
        "0.25",
        "--split-group-by-sample-id",
        "--seed",
        str(seed),
        "--gaussian-noise-std",
        str(params["gaussian"]),
        "--poisson-peak",
        str(params["poisson"]),
        "--background-scatter",
        str(params["background"]),
        "--degradation-probability",
        "1.0",
    ]
    if args.use_amp:
        command.append("--use-amp")

    print(f"Evaluating {artifact_dir.name}")
    subprocess.run(command, check=True)
    return artifact_dir / "eval_summary.json"


def read_summary(path: Path, fusion_mode: str, seed: int, condition: str) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    return {
        "eval_dir": path.parent.name,
        "fusion_mode": fusion_mode,
        "seed": seed,
        "condition": condition,
        "acc": float(summary["acc"]),
        "loss": float(summary["loss"]),
        "num_val_samples": int(summary["num_val_samples"]),
        "summary_path": str(path),
    }


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["fusion_mode"]), str(row["condition"]))].append(row)

    output: list[dict[str, object]] = []
    for (fusion_mode, condition), group_rows in sorted(grouped.items()):
        accs = [float(row["acc"]) for row in group_rows]
        seeds = sorted(int(row["seed"]) for row in group_rows)
        output.append(
            {
                "fusion_mode": fusion_mode,
                "condition": condition,
                "num_runs": len(accs),
                "mean_acc": mean(accs),
                "std_acc": stdev(accs) if len(accs) > 1 else 0.0,
                "min_acc": min(accs),
                "max_acc": max(accs),
                "seeds": " ".join(str(seed) for seed in seeds),
            }
        )
    return output


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(path: Path, aggregate: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_condition: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in aggregate:
        by_condition[str(row["condition"])].append(row)

    condition_order = ["clean", "light_noise_g002_p80_b002", "strong_noise_g005_p30_b005"]
    fusion_order = ["attention", "mean", "attention_residual"]
    lines = [
        "# v8 mv4 full-stack fusion robustness report",
        "",
        "This report compares full-stack fusion modes on the four-view v8 normalized dataset.",
        "",
        "| fusion | condition | runs | mean acc | std | min | max | seeds |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for condition in condition_order:
        rows = {str(row["fusion_mode"]): row for row in by_condition.get(condition, [])}
        for fusion in fusion_order:
            row = rows.get(fusion)
            if row is None:
                continue
            lines.append(
                "| {fusion} | {condition} | {num_runs} | {mean_acc} | {std_acc} | {min_acc} | {max_acc} | {seeds} |".format(
                    fusion=fusion,
                    condition=condition,
                    num_runs=row["num_runs"],
                    mean_acc=fmt(row["mean_acc"]),
                    std_acc=fmt(row["std_acc"]),
                    min_acc=fmt(row["min_acc"]),
                    max_acc=fmt(row["max_acc"]),
                    seeds=row["seeds"],
                )
            )

    lines.extend(["", "## Interpretation", ""])
    for condition in condition_order:
        rows = by_condition.get(condition, [])
        if rows:
            best = max(rows, key=lambda row: float(row["mean_acc"]))
            lines.append(f"- `{condition}`: best mean accuracy is `{fmt(best['mean_acc'])}` with `{best['fusion_mode']}`.")
    lines.extend(
        [
            "",
            "The fusion result should be used as an engineering ablation. The paper's main claim remains the gate-stack advantage and anti-shortcut validation protocol, not a universal claim about one fusion head.",
            "",
            "Files:",
            "",
            "- `experiments/v8_mv4_norm_mixaug_full_fusion_eval_summary_3seed.csv`",
            "- `experiments/v8_mv4_norm_mixaug_full_fusion_eval_aggregate_3seed.csv`",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for fusion_mode in FUSION_EXPERIMENTS:
        for seed in args.seeds:
            for condition, params in CONDITIONS.items():
                summary_path = run_eval(args, fusion_mode, seed, condition, params)
                rows.append(read_summary(summary_path, fusion_mode, seed, condition))

    aggregate = aggregate_rows(rows)
    rows = sorted(rows, key=lambda row: (str(row["fusion_mode"]), int(row["seed"]), str(row["condition"])))
    write_csv(args.summary_csv, rows, SUMMARY_FIELDS)
    write_csv(args.aggregate_csv, aggregate, AGGREGATE_FIELDS)
    if args.markdown_out is not None:
        write_markdown(args.markdown_out, aggregate)

    print(f"summary_rows={len(rows)}")
    print(f"aggregate_rows={len(aggregate)}")
    print(f"summary_csv={args.summary_csv}")
    print(f"aggregate_csv={args.aggregate_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

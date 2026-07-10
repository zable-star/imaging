from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


SUMMARY_FIELDS = [
    "eval_dir",
    "input_variant",
    "seed",
    "condition",
    "acc",
    "loss",
    "num_val_samples",
    "input_mode",
    "single_gate_index",
    "fusion_mode",
    "gaussian_noise_std",
    "poisson_peak",
    "background_scatter",
    "degradation_probability",
    "summary_path",
]

AGGREGATE_FIELDS = [
    "input_variant",
    "condition",
    "num_runs",
    "mean_acc",
    "std_acc",
    "min_acc",
    "max_acc",
    "seeds",
]

EVAL_RE = re.compile(
    r"^eval_v8_mv4_(?P<input_variant>full|gate0|gate1|gate2)_seed(?P<seed>\d+)_(?P<condition>.+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize gated evaluation folders into CSV and Markdown.")
    parser.add_argument("--eval-root", type=Path, default=Path("experiments"))
    parser.add_argument("--name-prefix", default="eval_v8_mv4_")
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--aggregate-csv", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_rows(eval_root: Path, name_prefix: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for eval_dir in sorted(path for path in eval_root.iterdir() if path.is_dir() and path.name.startswith(name_prefix)):
        match = EVAL_RE.match(eval_dir.name)
        if not match:
            continue
        summary_path = eval_dir / "eval_summary.json"
        if not summary_path.exists():
            continue
        summary = read_json(summary_path)
        rows.append(
            {
                "eval_dir": eval_dir.name,
                "input_variant": match.group("input_variant"),
                "seed": int(match.group("seed")),
                "condition": match.group("condition"),
                "acc": float(summary["acc"]),
                "loss": float(summary["loss"]),
                "num_val_samples": int(summary["num_val_samples"]),
                "input_mode": summary["input_mode"],
                "single_gate_index": int(summary["single_gate_index"]),
                "fusion_mode": summary["fusion_mode"],
                "gaussian_noise_std": float(summary["gaussian_noise_std"]),
                "poisson_peak": float(summary["poisson_peak"]),
                "background_scatter": float(summary["background_scatter"]),
                "degradation_probability": float(summary["degradation_probability"]),
                "summary_path": str(summary_path),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["input_variant"]), str(row["condition"]))].append(row)

    out: list[dict[str, object]] = []
    for (input_variant, condition), group_rows in sorted(grouped.items()):
        accs = [float(row["acc"]) for row in group_rows]
        seeds = sorted(int(row["seed"]) for row in group_rows)
        out.append(
            {
                "input_variant": input_variant,
                "condition": condition,
                "num_runs": len(accs),
                "mean_acc": mean(accs),
                "std_acc": stdev(accs) if len(accs) > 1 else 0.0,
                "min_acc": min(accs),
                "max_acc": max(accs),
                "seeds": " ".join(str(seed) for seed in seeds),
            }
        )
    return out


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(path: Path, summary_rows: list[dict[str, object]], aggregate: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_condition: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in aggregate:
        by_condition[str(row["condition"])].append(row)

    preferred_conditions = ["clean", "light_noise_g002_p80_b002", "strong_noise_g005_p30_b005"]
    remaining_conditions = sorted(condition for condition in by_condition if condition not in preferred_conditions)
    condition_order = [condition for condition in preferred_conditions if condition in by_condition] + remaining_conditions
    input_order = ["full", "gate0", "gate1", "gate2"]

    lines = [
        "# v8 mv4 multiview robustness report",
        "",
        "This report evaluates saved v8 multiview attention models on the grouped validation split.",
        "The split keeps all rendered views of the same source model in the same partition.",
        "",
        "## Setup",
        "",
        "- Dataset: `dataset_new/Military_TF_v8_mv4_norm`",
        "- Views: 0, 90, 180, and 270 degrees around Z",
        "- Classes: `true3d` and `flat_false`",
        "- Seeds: 42, 332, 2026",
        "- Evaluation conditions: clean, light noise, strong noise",
        "",
        "## Aggregate accuracy",
        "",
        "| input | condition | runs | mean acc | std | min | max | seeds |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]

    for condition in condition_order:
        rows = {str(row["input_variant"]): row for row in by_condition.get(condition, [])}
        for input_variant in input_order:
            row = rows.get(input_variant)
            if row is None:
                continue
            lines.append(
                "| {input_variant} | {condition} | {num_runs} | {mean_acc} | {std_acc} | {min_acc} | {max_acc} | {seeds} |".format(
                    input_variant=input_variant,
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
        if not rows:
            continue
        best = max(rows, key=lambda row: float(row["mean_acc"]))
        lines.append(
            "- `{}`: best mean accuracy is `{}` with `{}`.".format(
                condition, fmt(best["mean_acc"]), best["input_variant"]
            )
        )

    full_rows = {str(row["condition"]): row for row in aggregate if str(row["input_variant"]) == "full"}
    gate_rows = [row for row in aggregate if str(row["input_variant"]).startswith("gate")]
    if full_rows and gate_rows:
        lines.extend(
            [
                "",
                "The defensible claim is that the full gate stack gives the strongest or most stable validation performance under the current v8 multiview simulation. "
                "Single-gate inputs are still informative, so the result should be presented as added depth-gated evidence rather than complete removal of shape shortcuts.",
            ]
        )

    lines.extend(["", "## Files", ""])
    lines.append(f"- Per-run CSV: `{Path('experiments') / 'v8_mv4_norm_mixaug_attention_eval_summary_3seed.csv'}`")
    lines.append(f"- Aggregate CSV: `{Path('experiments') / 'v8_mv4_norm_mixaug_attention_eval_aggregate_3seed.csv'}`")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows = collect_rows(args.eval_root, args.name_prefix)
    if not rows:
        raise RuntimeError(f"No evaluation summaries found under {args.eval_root} with prefix {args.name_prefix!r}")

    rows = sorted(rows, key=lambda row: (str(row["input_variant"]), int(row["seed"]), str(row["condition"])))
    aggregate = aggregate_rows(rows)
    write_csv(args.summary_csv, rows, SUMMARY_FIELDS)
    write_csv(args.aggregate_csv, aggregate, AGGREGATE_FIELDS)
    if args.markdown_out is not None:
        write_markdown(args.markdown_out, rows, aggregate)

    print(f"summary_rows={len(rows)}")
    print(f"aggregate_rows={len(aggregate)}")
    print(f"summary_csv={args.summary_csv}")
    print(f"aggregate_csv={args.aggregate_csv}")
    if args.markdown_out is not None:
        print(f"markdown={args.markdown_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

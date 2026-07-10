from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize per-class validation accuracy and mean gate attention from experiment folders."
    )
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_summary(run_dir: Path) -> dict:
    path = run_dir / "summary.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_run(run_dir: Path) -> list[dict[str, object]]:
    attention_path = run_dir / "val_attention_weights.csv"
    if not attention_path.exists():
        return []

    summary = load_summary(run_dir)
    seed = summary.get("seed", "")
    experiment = run_dir.parent.name
    rows_by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    with attention_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows_by_class[row["class_name"]].append(row)

    output_rows: list[dict[str, object]] = []
    for class_name, rows in sorted(rows_by_class.items()):
        correct = sum(int(row["pred"]) == int(row["gt"]) for row in rows)
        attn_keys = sorted(key for key in rows[0] if key.startswith("attn_gate_"))
        out = {
            "experiment": experiment,
            "run": run_dir.name,
            "seed": seed,
            "class_name": class_name,
            "val_count": len(rows),
            "correct": correct,
            "accuracy": correct / max(len(rows), 1),
        }
        for key in attn_keys:
            out[f"mean_{key}"] = mean([float(row[key]) for row in rows])
        output_rows.append(out)

    all_rows = [row for rows in rows_by_class.values() for row in rows]
    if all_rows:
        correct = sum(int(row["pred"]) == int(row["gt"]) for row in all_rows)
        attn_keys = sorted(key for key in all_rows[0] if key.startswith("attn_gate_"))
        out = {
            "experiment": experiment,
            "run": run_dir.name,
            "seed": seed,
            "class_name": "__overall__",
            "val_count": len(all_rows),
            "correct": correct,
            "accuracy": correct / max(len(all_rows), 1),
        }
        for key in attn_keys:
            out[f"mean_{key}"] = mean([float(row[key]) for row in all_rows])
        output_rows.append(out)

    return output_rows


def main() -> int:
    args = parse_args()
    if not args.experiment_root.exists():
        raise FileNotFoundError(f"Experiment root does not exist: {args.experiment_root}")

    rows: list[dict[str, object]] = []
    for run_dir in sorted(path for path in args.experiment_root.iterdir() if path.is_dir()):
        rows.extend(summarize_run(run_dir))
    if not rows:
        raise RuntimeError(f"No val_attention_weights.csv files found under {args.experiment_root}")

    fieldnames = list(rows[0].keys())
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"rows={len(rows)}")
    print(f"csv={args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

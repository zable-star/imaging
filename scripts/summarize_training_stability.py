from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path


FIELDNAMES = [
    "experiment",
    "run",
    "seed",
    "epochs",
    "best_val_acc",
    "best_epoch",
    "final_val_acc",
    "last_k",
    "last_k_mean_val_acc",
    "last_k_std_val_acc",
    "first_epoch_ge_0.9",
    "first_epoch_ge_1.0",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize best/final/late-epoch validation stability.")
    parser.add_argument("experiment_dirs", nargs="+", type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--last-k", type=int, default=5)
    return parser.parse_args()


def read_history(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_seed(run_name: str) -> str:
    match = re.search(r"_seed(\d+)$", run_name)
    return match.group(1) if match else ""


def first_epoch(rows: list[dict[str, str]], threshold: float) -> str:
    for row in rows:
        if float(row["val_acc"]) >= threshold:
            return row["epoch"]
    return ""


def summarize_run(experiment: str, run_dir: Path, last_k: int) -> dict[str, object]:
    rows = read_history(run_dir / "training_history.csv")
    if not rows:
        raise RuntimeError(f"Empty training history: {run_dir}")

    best_row = max(rows, key=lambda row: float(row["val_acc"]))
    final_row = rows[-1]
    late_rows = rows[-last_k:]
    late_accs = [float(row["val_acc"]) for row in late_rows]

    return {
        "experiment": experiment,
        "run": run_dir.name,
        "seed": parse_seed(run_dir.name),
        "epochs": len(rows),
        "best_val_acc": float(best_row["val_acc"]),
        "best_epoch": best_row["epoch"],
        "final_val_acc": float(final_row["val_acc"]),
        "last_k": min(last_k, len(rows)),
        "last_k_mean_val_acc": statistics.mean(late_accs),
        "last_k_std_val_acc": statistics.stdev(late_accs) if len(late_accs) > 1 else 0.0,
        "first_epoch_ge_0.9": first_epoch(rows, 0.9),
        "first_epoch_ge_1.0": first_epoch(rows, 1.0),
    }


def collect_rows(experiment_dirs: list[Path], last_k: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for experiment_dir in experiment_dirs:
        experiment = experiment_dir.name
        for run_dir in sorted(path for path in experiment_dir.iterdir() if path.is_dir()):
            history_path = run_dir / "training_history.csv"
            if history_path.exists():
                rows.append(summarize_run(experiment, run_dir, last_k))
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.last_k <= 0:
        raise ValueError("last-k must be positive")
    rows = collect_rows(args.experiment_dirs, args.last_k)
    write_csv(args.output_csv, rows)
    print(f"Wrote stability summary: {args.output_csv}")
    print(f"Runs summarized: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

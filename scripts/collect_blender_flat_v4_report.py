from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


RUN_RE = re.compile(r"localgpu_blender_flat_rr_gain2_min035_v4_(full|gate[0-2])_(\d+)ep_seed(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect the Blender flat-target v4 evidence report.")
    parser.add_argument(
        "--gate-stack-classes",
        type=Path,
        default=Path("dataset_new/military_true_false_selected44_blender_flat_rr_gain2_min035_v4_gate_stack_classes.csv"),
    )
    parser.add_argument(
        "--single-gate-separability",
        type=Path,
        default=Path(
            "dataset_new/military_true_false_selected44_blender_flat_rr_gain2_min035_v4_single_gate_feature_separability.csv"
        ),
    )
    parser.add_argument(
        "--false-metadata-root",
        type=Path,
        default=Path("dataset_new/Military_FlatFalse_Selected44_blender_flat_rr_gain2_min035_v4"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("experiments/localgpu_blender_flat_rr_gain2_min035_v4_ablation_summary.csv"),
    )
    parser.add_argument(
        "--output-aggregate-csv",
        type=Path,
        default=Path("experiments/localgpu_blender_flat_rr_gain2_min035_v4_ablation_aggregate.csv"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("writing/blender_flat_rr_gain2_min035_v4_report_2026-07-07.md"),
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def best_epoch(history_path: Path) -> str:
    rows = read_csv(history_path)
    if not rows:
        return ""
    return max(rows, key=lambda row: float(row["val_acc"]))["epoch"]


def label_for_kind(kind: str) -> str:
    if kind == "full":
        return "Full 3-gate stack"
    return f"Gate {kind[-1]} only"


def collect_training_rows(experiment_root: Path = Path("experiments")) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for run_dir in sorted(experiment_root.glob("localgpu_blender_flat_rr_gain2_min035_v4_*_20ep_seed*")):
        match = RUN_RE.match(run_dir.name)
        if not match:
            continue
        kind, epochs, seed = match.groups()
        summary_path = run_dir / "summary.json"
        history_path = run_dir / "training_history.csv"
        if not summary_path.exists():
            continue
        summary = read_summary(summary_path)
        rows.append(
            {
                "input": label_for_kind(kind),
                "best_val_acc": f"{float(summary['best_val_acc']):.4f}",
                "best_epoch": best_epoch(history_path) if history_path.exists() else "",
                "seed": seed,
                "epochs": epochs,
                "split_group_by_sample_id": str(summary.get("split_group_by_sample_id", "")),
                "artifact_dir": str(run_dir),
            }
        )
    return rows


def aggregate_training_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    order = ["Full 3-gate stack", "Gate 0 only", "Gate 1 only", "Gate 2 only"]
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["input"], []).append(row)

    aggregate_rows: list[dict[str, str]] = []
    for label in order:
        group = grouped.get(label, [])
        if not group:
            continue
        accs = [float(row["best_val_acc"]) for row in group]
        aggregate_rows.append(
            {
                "input": label,
                "num_runs": str(len(accs)),
                "mean_best_val_acc": f"{statistics.mean(accs):.4f}",
                "std_best_val_acc": f"{statistics.stdev(accs):.4f}" if len(accs) > 1 else "0.0000",
                "min_best_val_acc": f"{min(accs):.4f}",
                "max_best_val_acc": f"{max(accs):.4f}",
                "seeds": "/".join(row["seed"] for row in sorted(group, key=lambda item: int(item["seed"]))),
            }
        )
    return aggregate_rows


def collect_gate_stack_rows(path: Path) -> list[dict[str, str]]:
    rows = []
    for row in read_csv(path):
        rows.append(
            {
                "class_name": row["class_name"],
                "num_samples": row["num_samples"],
                "corr_maxnorm": f"{float(row['mean_pair_corr_maxnorm']):.4f}",
                "mask_iou": f"{float(row['mean_pair_mask_iou']):.4f}",
                "absdiff_maxnorm": f"{float(row['mean_pair_absdiff_maxnorm']):.4f}",
            }
        )
    return rows


def collect_top_single_gate_features(path: Path) -> list[dict[str, str]]:
    best_by_gate: dict[str, dict[str, str]] = {}
    for row in read_csv(path):
        gate = row["gate"]
        if gate not in best_by_gate or float(row["best_accuracy"]) > float(best_by_gate[gate]["best_accuracy"]):
            best_by_gate[gate] = row
    return [
        {
            "gate": gate,
            "feature": row["feature"],
            "accuracy": f"{float(row['best_accuracy']):.4f}",
            "true3d_mean": f"{float(row['class0_mean']):.4f}",
            "flat_false_mean": f"{float(row['class1_mean']):.4f}",
            "cohen_d": f"{float(row['cohen_d']):.3f}",
        }
        for gate, row in sorted(best_by_gate.items(), key=lambda item: int(item[0]))
    ]


def collect_false_metadata(root: Path) -> dict[str, object]:
    gate_counts: dict[int, int] = {}
    max_flat_span = 0.0
    bad = 0
    for path in root.rglob("*_metadata.json"):
        metadata = read_summary(path)
        gate = int(metadata["flat_target_gate_index"])
        gate_counts[gate] = gate_counts.get(gate, 0) + 1
        span = abs(float(metadata["flat_geometry_depth_max"]) - float(metadata["flat_geometry_depth_min"]))
        max_flat_span = max(max_flat_span, span)
        if metadata.get("flat_geometry_mode") != "flatten-camera-depth":
            bad += 1
        if metadata.get("flat_target_gate_index_mode") != "round-robin":
            bad += 1
    return {
        "num_metadata": sum(gate_counts.values()),
        "gate_counts": gate_counts,
        "max_flat_span": max_flat_span,
        "bad_metadata": bad,
    }


def write_training_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "input",
        "best_val_acc",
        "best_epoch",
        "seed",
        "epochs",
        "split_group_by_sample_id",
        "artifact_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_aggregate_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "input",
        "num_runs",
        "mean_best_val_acc",
        "std_best_val_acc",
        "min_best_val_acc",
        "max_best_val_acc",
        "seeds",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def table(lines: list[str], headers: list[str], rows: list[list[str]]) -> None:
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")


def write_markdown(
    path: Path,
    training_rows: list[dict[str, str]],
    aggregate_rows: list[dict[str, str]],
    gate_stack_rows: list[dict[str, str]],
    top_feature_rows: list[dict[str, str]],
    metadata_summary: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gate_counts = metadata_summary["gate_counts"]
    lines = [
        "# Blender flat false-target v4 evidence report",
        "",
        "This report records the current physics-side dataset revision and the first local network ablation.",
        "It should be treated as evidence plus limitation, not as a final paper result.",
        "",
        "## Dataset construction",
        "",
        f"- True target root: `dataset_new/Military_3D_Gated_Selected44_blender_norm_v2`.",
        f"- Flat false root: `dataset_new/Military_FlatFalse_Selected44_blender_flat_rr_gain2_min035_v4`.",
        f"- Binary dataset: `dataset_new/Military_TrueFalse_Selected44_blender_flat_rr_gain2_min035_v4`.",
        f"- Flat false metadata files: {metadata_summary['num_metadata']}.",
        f"- Flat target gate distribution: {dict(sorted(gate_counts.items()))}.",
        f"- Maximum flattened camera-depth span: {float(metadata_summary['max_flat_span']):.6g}.",
        f"- Metadata mode errors: {metadata_summary['bad_metadata']}.",
        "",
        "Interpretation: the false target is now generated in Blender as a camera-depth-flattened target,",
        "with round-robin gate placement and stronger in-render brightness, instead of PNG-level post-processing.",
        "",
        "## Gate-stack diagnostics",
        "",
    ]
    table(
        lines,
        ["class", "samples", "corr", "mask IoU", "absdiff"],
        [
            [
                row["class_name"],
                row["num_samples"],
                row["corr_maxnorm"],
                row["mask_iou"],
                row["absdiff_maxnorm"],
            ]
            for row in gate_stack_rows
        ],
    )
    lines.extend(
        [
            "Interpretation: flat false samples have nearly identical normalized gate stacks, while true 3D samples",
            "show much lower cross-gate correlation and larger normalized differences. This supports the physical",
            "story of planar echo consistency versus 3D depth slicing.",
            "",
            "## Network ablation, three seeds",
            "",
        ]
    )
    table(
        lines,
        ["input", "runs", "mean", "std", "min", "max", "seeds"],
        [
            [
                row["input"],
                row["num_runs"],
                row["mean_best_val_acc"],
                row["std_best_val_acc"],
                row["min_best_val_acc"],
                row["max_best_val_acc"],
                row["seeds"],
            ]
            for row in aggregate_rows
        ],
    )
    lines.extend(["Per-run details:", ""])
    table(
        lines,
        ["input", "best val acc", "best epoch", "seed"],
        [
            [
                row["input"],
                row["best_val_acc"],
                row["best_epoch"],
                row["seed"],
            ]
            for row in training_rows
        ],
    )
    lines.extend(
        [
            "Interpretation: the full gate stack reaches 1.0000 across all three seeds, but single-gate inputs remain high.",
            "Gate 1 also reaches 1.0000 across all three seeds, so the result cannot be claimed as pure gate-stack superiority.",
            "",
            "## Single-gate shortcut diagnostics",
            "",
        ]
    )
    table(
        lines,
        ["gate", "strongest scalar feature", "threshold acc", "true3d mean", "flat false mean", "Cohen d"],
        [
            [
                row["gate"],
                row["feature"],
                row["accuracy"],
                row["true3d_mean"],
                row["flat_false_mean"],
                row["cohen_d"],
            ]
            for row in top_feature_rows
        ],
    )
    lines.extend(
        [
            "Interpretation: max-value and foreground statistics still separate the two classes in single gates.",
            "This is a limitation of the current simulation and should drive the next revision: add background, detector noise,",
            "reflectance variation, distance/exposure balancing, and multi-view rendering before claiming strong generalization.",
            "",
            "## Paper-use wording",
            "",
            "- Safe claim: Blender-side flattened false targets produce a physically interpretable control with high gate-stack consistency.",
            "- Safe claim: full gate stacks are stable, but current single-gate shortcuts remain and must be treated as limitations.",
            "- Do not claim: a single gate is useless, or the method is already validated on real battlefield data.",
            "- Next required evidence: multi-seed training on v4/v5, background/noise robustness, and a 3090 run with higher epochs.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    training_rows = collect_training_rows()
    aggregate_rows = aggregate_training_rows(training_rows)
    gate_stack_rows = collect_gate_stack_rows(args.gate_stack_classes)
    top_feature_rows = collect_top_single_gate_features(args.single_gate_separability)
    metadata_summary = collect_false_metadata(args.false_metadata_root)
    write_training_csv(args.output_csv, training_rows)
    write_aggregate_csv(args.output_aggregate_csv, aggregate_rows)
    write_markdown(args.output_md, training_rows, aggregate_rows, gate_stack_rows, top_feature_rows, metadata_summary)
    print(f"Wrote CSV: {args.output_csv}")
    print(f"Wrote aggregate CSV: {args.output_aggregate_csv}")
    print(f"Wrote Markdown: {args.output_md}")
    print(f"Training runs: {len(training_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

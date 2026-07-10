from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect the Blender reflectance-randomized v5 evidence report.")
    parser.add_argument("--experiment-tag", default="blender_refl_rr_gain2_min035_v5")
    parser.add_argument("--report-label", default="current")
    parser.add_argument("--true-root-label", default="dataset_new/Military_3D_Gated_Selected44_blender_refl_v5")
    parser.add_argument("--false-root-label", default="dataset_new/Military_FlatFalse_Selected44_blender_refl_rr_gain2_min035_v5")
    parser.add_argument("--binary-root-label", default="dataset_new/Military_TrueFalse_Selected44_blender_refl_rr_gain2_min035_v5")
    parser.add_argument(
        "--gate-stack-classes",
        type=Path,
        default=Path("dataset_new/military_true_false_selected44_blender_refl_rr_gain2_min035_v5_gate_stack_classes.csv"),
    )
    parser.add_argument(
        "--single-gate-separability",
        type=Path,
        default=Path(
            "dataset_new/military_true_false_selected44_blender_refl_rr_gain2_min035_v5_single_gate_feature_separability.csv"
        ),
    )
    parser.add_argument(
        "--false-metadata-root",
        type=Path,
        default=Path("dataset_new/Military_FlatFalse_Selected44_blender_refl_rr_gain2_min035_v5"),
    )
    parser.add_argument("--experiment-root", type=Path, default=Path("experiments"))
    parser.add_argument(
        "--reference-aggregate",
        type=Path,
        default=Path("experiments/localgpu_blender_flat_rr_gain2_min035_v4_ablation_aggregate.csv"),
    )
    parser.add_argument("--reference-label", default="reference")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("experiments/localgpu_blender_refl_rr_gain2_min035_v5_ablation_summary.csv"),
    )
    parser.add_argument(
        "--output-aggregate-csv",
        type=Path,
        default=Path("experiments/localgpu_blender_refl_rr_gain2_min035_v5_ablation_aggregate.csv"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("writing/blender_refl_rr_gain2_min035_v5_report_2026-07-07.md"),
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, object]:
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


def collect_training_rows(experiment_root: Path, experiment_tag: str) -> list[dict[str, str]]:
    run_re = re.compile(rf"localgpu_{re.escape(experiment_tag)}_(full|gate[0-2])_(\d+)ep_seed(\d+)$")
    rows: list[dict[str, str]] = []
    for run_dir in sorted(experiment_root.glob(f"localgpu_{experiment_tag}_*_20ep_seed*")):
        match = run_re.match(run_dir.name)
        if not match:
            continue
        kind, epochs, seed = match.groups()
        summary_path = run_dir / "summary.json"
        history_path = run_dir / "training_history.csv"
        if not summary_path.exists():
            continue
        summary = read_json(summary_path)
        rows.append(
            {
                "input": label_for_kind(kind),
                "best_val_acc": f"{float(summary['best_val_acc']):.4f}",
                "best_epoch": best_epoch(history_path) if history_path.exists() else "",
                "seed": seed,
                "epochs": epochs,
                "split_group_by_sample_id": str(summary.get("split_group_by_sample_id", "")),
                "use_amp": str(summary.get("use_amp", "")),
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
        sorted_group = sorted(group, key=lambda item: int(item["seed"]))
        aggregate_rows.append(
            {
                "input": label,
                "num_runs": str(len(accs)),
                "mean_best_val_acc": f"{statistics.mean(accs):.4f}",
                "std_best_val_acc": f"{statistics.stdev(accs):.4f}" if len(accs) > 1 else "0.0000",
                "min_best_val_acc": f"{min(accs):.4f}",
                "max_best_val_acc": f"{max(accs):.4f}",
                "seeds": "/".join(row["seed"] for row in sorted_group),
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
                "max_mean_ratio": f"{float(row['mean_max_mean_ratio']):.4f}",
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
    spans: list[float] = []
    reflectances: list[float] = []
    bad = 0
    for path in root.rglob("*_metadata.json"):
        metadata = read_json(path)
        gate = int(metadata["flat_target_gate_index"])
        gate_counts[gate] = gate_counts.get(gate, 0) + 1
        spans.append(abs(float(metadata["flat_geometry_depth_max"]) - float(metadata["flat_geometry_depth_min"])))
        reflectances.append(float(metadata["target_reflectance"]))
        if metadata.get("flat_geometry_mode") != "flatten-camera-depth":
            bad += 1
        if metadata.get("flat_target_gate_index_mode") != "round-robin":
            bad += 1
        if metadata.get("reflectance_mode") != "hash-log-uniform":
            bad += 1
    return {
        "num_metadata": sum(gate_counts.values()),
        "gate_counts": gate_counts,
        "max_flat_span": max(spans) if spans else 0.0,
        "reflectance_min": min(reflectances) if reflectances else 0.0,
        "reflectance_max": max(reflectances) if reflectances else 0.0,
        "reflectance_mean": statistics.mean(reflectances) if reflectances else 0.0,
        "bad_metadata": bad,
    }


def write_dict_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_table(lines: list[str], headers: list[str], rows: list[list[str]]) -> None:
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")


def aggregate_lookup(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["input"]: row for row in rows}


def read_v4_comparison(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return read_csv(path)


def write_markdown(
    path: Path,
    training_rows: list[dict[str, str]],
    aggregate_rows: list[dict[str, str]],
    gate_stack_rows: list[dict[str, str]],
    top_feature_rows: list[dict[str, str]],
    metadata_summary: dict[str, object],
    reference_rows: list[dict[str, str]],
    reference_label: str,
    report_label: str,
    true_root_label: str,
    false_root_label: str,
    binary_root_label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gate_counts = dict(sorted(metadata_summary["gate_counts"].items()))
    lines = [
        "# Blender false-target evidence report",
        "",
        f"This report records the {report_label} revision of the laser-gated military true/false-target dataset.",
        "The purpose is to move the work from a visually plausible demo toward a paper-grade validation chain.",
        "",
        "## Dataset construction",
        "",
        f"- True target root: `{true_root_label}`.",
        f"- Flat false root: `{false_root_label}`.",
        f"- Binary dataset: `{binary_root_label}`.",
        f"- Flat false metadata files: {metadata_summary['num_metadata']}.",
        f"- Flat target gate distribution: {gate_counts}.",
        f"- Maximum flattened camera-depth span: {float(metadata_summary['max_flat_span']):.6g}.",
        (
            f"- Reflectance range: {float(metadata_summary['reflectance_min']):.4f} to "
            f"{float(metadata_summary['reflectance_max']):.4f}; mean "
            f"{float(metadata_summary['reflectance_mean']):.4f}."
        ),
        f"- Metadata mode errors: {metadata_summary['bad_metadata']}.",
        "",
        "Interpretation: the planar false target is generated in Blender with metadata-recorded geometry, gate response,",
        "and reflectance settings. This is preferable to PNG-level histogram or clipping post-processing because the",
        "sample metadata records the physical nuisance factors.",
        "",
        "## Gate-stack diagnostics",
        "",
    ]
    add_table(
        lines,
        ["class", "samples", "corr", "mask IoU", "absdiff", "max/mean ratio"],
        [
            [
                row["class_name"],
                row["num_samples"],
                row["corr_maxnorm"],
                row["mask_iou"],
                row["absdiff_maxnorm"],
                row["max_mean_ratio"],
            ]
            for row in gate_stack_rows
        ],
    )
    lines.extend(
        [
            "Interpretation: this table should be read as a physics-control diagnostic, not only as an accuracy metric.",
            "A useful setting should keep planar false-target responses interpretable while preserving enough true-3D",
            "cross-gate diversity. If the true3d correlation becomes too high, the gates may be over-smoothed; if",
            "flat_false correlation becomes too low, the false target may have black-frame shortcuts.",
            "",
            "## Network ablation, three seeds",
            "",
        ]
    )
    add_table(
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
    add_table(
        lines,
        ["input", "best val acc", "best epoch", "seed"],
        [[row["input"], row["best_val_acc"], row["best_epoch"], row["seed"]] for row in training_rows],
    )
    if reference_rows:
        v5_by_input = aggregate_lookup(aggregate_rows)
        lines.extend([f"## {reference_label} comparison", ""])
        comparison_rows = []
        for row in reference_rows:
            label = row["input"]
            if label not in v5_by_input:
                continue
            v5 = v5_by_input[label]
            delta = float(v5["mean_best_val_acc"]) - float(row["mean_best_val_acc"])
            comparison_rows.append(
                [
                    label,
                    row["mean_best_val_acc"],
                    v5["mean_best_val_acc"],
                    f"{delta:+.4f}",
                ]
            )
        add_table(lines, ["input", f"{reference_label} mean", "current mean", "delta"], comparison_rows)
    lines.extend(
        [
            "## Single-gate shortcut diagnostics",
            "",
        ]
    )
    add_table(
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
            "## Paper-use interpretation",
            "",
            "- Stronger claim now supported: the simulator can generate metadata-proven planar false targets and 3D targets whose gate-stack statistics differ in the expected direction.",
            "- More cautious claim: single-gate classifiers and scalar shortcuts must remain explicit controls before claiming robust gate-stack discrimination.",
            "- Current limitation: the dataset has only 44 selected military models per class condition, so the result is a controlled simulation validation rather than deployment evidence.",
            "- Next validation step: add background/clutter, range/exposure balancing, multi-view renders, and train a stricter model with single-gate and gate-dropout controls on the 3090.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    training_rows = collect_training_rows(args.experiment_root, args.experiment_tag)
    aggregate_rows = aggregate_training_rows(training_rows)
    gate_stack_rows = collect_gate_stack_rows(args.gate_stack_classes)
    top_feature_rows = collect_top_single_gate_features(args.single_gate_separability)
    metadata_summary = collect_false_metadata(args.false_metadata_root)
    reference_rows = read_v4_comparison(args.reference_aggregate)

    write_dict_csv(
        args.output_csv,
        training_rows,
        ["input", "best_val_acc", "best_epoch", "seed", "epochs", "split_group_by_sample_id", "use_amp", "artifact_dir"],
    )
    write_dict_csv(
        args.output_aggregate_csv,
        aggregate_rows,
        ["input", "num_runs", "mean_best_val_acc", "std_best_val_acc", "min_best_val_acc", "max_best_val_acc", "seeds"],
    )
    write_markdown(
        args.output_md,
        training_rows,
        aggregate_rows,
        gate_stack_rows,
        top_feature_rows,
        metadata_summary,
        reference_rows,
        args.reference_label,
        args.report_label,
        args.true_root_label,
        args.false_root_label,
        args.binary_root_label,
    )
    print(f"training_rows={len(training_rows)}")
    print(f"aggregate_rows={len(aggregate_rows)}")
    print(f"output_csv={args.output_csv}")
    print(f"output_aggregate_csv={args.output_aggregate_csv}")
    print(f"output_md={args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

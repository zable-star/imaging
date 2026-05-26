from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
DEFAULT_DATASET_ROOT = Path(__file__).resolve().parent / "dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize gate attention scores from val_attention_weights.csv.")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--attention-csv", type=Path, default=None)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def load_classes(artifact_dir: Path, rows: list[dict[str, str]]) -> list[str]:
    summary_path = artifact_dir / "summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            return list(json.load(f)["classes"])
    return sorted({row["class_name"] for row in rows})


def infer_gate_columns(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return []
    return sorted(key for key in rows[0] if key.startswith("attn_gate_"))


def class_attention_summary(rows: list[dict[str, str]], classes: list[str], gate_columns: list[str]) -> list[dict[str, object]]:
    rows_by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_class[row["class_name"]].append(row)

    out_rows: list[dict[str, object]] = []
    for class_name in classes:
        class_rows = rows_by_class.get(class_name, [])
        summary = {"class_name": class_name, "num_samples": len(class_rows)}
        values = [mean([float(row[gate]) for row in class_rows]) for gate in gate_columns]
        for gate, value in zip(gate_columns, values):
            summary[gate] = value
        if values:
            best_gate_idx = max(range(len(values)), key=values.__getitem__)
            summary["max_attention_gate"] = gate_columns[best_gate_idx].replace("attn_gate_", "gate_")
            summary["max_attention"] = values[best_gate_idx]
        out_rows.append(summary)
    return out_rows


def load_image2d_manifest(dataset_root: Path) -> dict[str, int]:
    manifest_path = dataset_root / "image2d" / "manifest.csv"
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        return {row["target_sample_id"]: int(row["informative_gate"]) for row in csv.DictReader(f)}


def image2d_active_attention_summary(
    rows: list[dict[str, str]],
    manifest: dict[str, int],
    gate_columns: list[str],
) -> list[dict[str, object]]:
    image2d_rows = [row for row in rows if row["class_name"] == "image2d" and row["sample_id"] in manifest]
    by_active_gate: dict[int, list[dict[str, str]]] = defaultdict(list)
    active_scores = []
    black_scores = []

    for row in image2d_rows:
        active_gate = manifest[row["sample_id"]]
        by_active_gate[active_gate].append(row)
        for gate_idx, gate_column in enumerate(gate_columns):
            value = float(row[gate_column])
            if gate_idx == active_gate:
                active_scores.append(value)
            else:
                black_scores.append(value)

    out_rows: list[dict[str, object]] = [
        {
            "group": "all_image2d_active_gate",
            "num_samples": len(image2d_rows),
            "mean_attention": mean(active_scores),
        },
        {
            "group": "all_image2d_black_gates",
            "num_samples": len(image2d_rows) * max(len(gate_columns) - 1, 0),
            "mean_attention": mean(black_scores),
        },
    ]

    for gate_idx in sorted(by_active_gate):
        gate_rows = by_active_gate[gate_idx]
        active_values = [float(row[f"attn_gate_{gate_idx}"]) for row in gate_rows]
        other_values = []
        for row in gate_rows:
            for other_idx in range(len(gate_columns)):
                if other_idx != gate_idx:
                    other_values.append(float(row[f"attn_gate_{other_idx}"]))
        out_rows.append(
            {
                "group": f"image2d_active_gate_{gate_idx}",
                "num_samples": len(gate_rows),
                "mean_attention": mean(active_values),
                "mean_black_gate_attention": mean(other_values),
            }
        )
    return out_rows


def plot_class_attention(summary_rows: list[dict[str, object]], gate_columns: list[str], out_path: Path) -> None:
    class_names = [str(row["class_name"]) for row in summary_rows]
    values = [[float(row[gate]) for gate in gate_columns] for row in summary_rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    im = ax.imshow(values, cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(gate_columns)), [gate.replace("attn_", "") for gate in gate_columns])
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_title("Mean Attention Score by Class")

    for row_idx, row_values in enumerate(values):
        for col_idx, value in enumerate(row_values):
            ax.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", color="black", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    attention_csv = args.attention_csv or args.artifact_dir / "val_attention_weights.csv"
    rows = read_rows(attention_csv)
    gate_columns = infer_gate_columns(rows)
    classes = load_classes(args.artifact_dir, rows)

    class_rows = class_attention_summary(rows, classes, gate_columns)
    class_csv = args.artifact_dir / "attention_mean_by_class.csv"
    write_csv(
        class_csv,
        ["class_name", "num_samples", *gate_columns, "max_attention_gate", "max_attention"],
        class_rows,
    )
    plot_class_attention(class_rows, gate_columns, args.artifact_dir / "attention_mean_by_class.png")

    manifest = load_image2d_manifest(args.dataset_root)
    image2d_rows = image2d_active_attention_summary(rows, manifest, gate_columns)
    image2d_csv = args.artifact_dir / "attention_image2d_active_vs_black.csv"
    write_csv(
        image2d_csv,
        ["group", "num_samples", "mean_attention", "mean_black_gate_attention"],
        image2d_rows,
    )

    print(f"Wrote {class_csv}")
    print(f"Wrote {args.artifact_dir / 'attention_mean_by_class.png'}")
    print(f"Wrote {image2d_csv}")
    print("\nClass attention mean:")
    for row in class_rows:
        scores = ", ".join(f"{gate}={float(row[gate]):.4f}" for gate in gate_columns)
        print(f"{row['class_name']}: {scores}")

    print("\nimage2d active-vs-black attention:")
    for row in image2d_rows:
        black = row.get("mean_black_gate_attention", "")
        if black == "":
            print(f"{row['group']}: mean_attention={float(row['mean_attention']):.4f}")
        else:
            print(
                f"{row['group']}: mean_attention={float(row['mean_attention']):.4f}, "
                f"mean_black_gate_attention={float(black):.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

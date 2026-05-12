from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASELINE_ROOT = Path(r"E:\wjz\test1\dataset\dataset_obj\slice_attention_baseline")
ARTIFACT_DIR = BASELINE_ROOT / "artifacts"
CSV_PATH = ARTIFACT_DIR / "val_attention_weights.csv"


def load_rows(csv_path: Path) -> list[dict]:
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {
                "sample_id": row["sample_id"],
                "class_name": row["class_name"],
                "pred": int(row["pred"]),
                "gt": int(row["gt"]),
            }
            gate_keys = sorted([k for k in row.keys() if k.startswith("attn_gate_")], key=lambda x: int(x.split("_")[-1]))
            parsed["gate_keys"] = gate_keys
            parsed["attn"] = [float(row[k]) for k in gate_keys]
            rows.append(parsed)
    return rows


def plot_sample_attention(rows: list[dict], out_path: Path) -> None:
    rows = sorted(rows, key=lambda r: (r["class_name"], r["sample_id"]))
    gate_keys = rows[0]["gate_keys"]
    gate_labels = [key.replace("attn_", "") for key in gate_keys]
    x = np.arange(len(rows))

    plt.figure(figsize=(14, 6))
    bottom = np.zeros(len(rows))
    colors = ["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#06B6D4"]

    for idx, gate_label in enumerate(gate_labels):
        values = np.array([row["attn"][idx] for row in rows])
        plt.bar(
            x,
            values,
            bottom=bottom,
            width=0.82,
            label=gate_label,
            color=colors[idx % len(colors)],
        )
        bottom += values

    tick_labels = [f"{row['class_name']}\n{row['sample_id']}" for row in rows]
    plt.xticks(x, tick_labels, rotation=90, fontsize=8)
    plt.ylabel("Attention Weight")
    plt.xlabel("Validation Samples")
    plt.title("Attention Distribution of Each Validation Sample")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_gate_means(rows: list[dict], out_path: Path) -> None:
    gate_keys = rows[0]["gate_keys"]
    gate_labels = [key.replace("attn_", "") for key in gate_keys]

    overall = np.mean(np.array([row["attn"] for row in rows]), axis=0)
    class_groups: dict[str, list[list[float]]] = defaultdict(list)
    for row in rows:
        class_groups[row["class_name"]].append(row["attn"])

    class_names = sorted(class_groups.keys())
    class_means = {name: np.mean(np.array(vals), axis=0) for name, vals in class_groups.items()}

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.bar(gate_labels, overall, color="#2563EB")
    plt.ylim(0, 1.0)
    plt.ylabel("Mean Attention Weight")
    plt.title("Overall Mean Attention by Gate")

    plt.subplot(1, 2, 2)
    x = np.arange(len(gate_labels))
    width = 0.35 if len(class_names) <= 2 else 0.8 / len(class_names)
    for idx, class_name in enumerate(class_names):
        offset = (idx - (len(class_names) - 1) / 2) * width
        plt.bar(x + offset, class_means[class_name], width=width, label=class_name)
    plt.xticks(x, gate_labels)
    plt.ylim(0, 1.0)
    plt.ylabel("Mean Attention Weight")
    plt.title("Mean Attention by Gate and Class")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def print_summary(rows: list[dict]) -> None:
    gate_keys = rows[0]["gate_keys"]
    gate_labels = [key.replace("attn_", "") for key in gate_keys]
    attn = np.array([row["attn"] for row in rows])
    overall = attn.mean(axis=0)

    print("Overall mean attention:")
    for label, value in zip(gate_labels, overall):
        print(f"  {label}: {value:.4f}")

    class_groups: dict[str, list[list[float]]] = defaultdict(list)
    for row in rows:
        class_groups[row["class_name"]].append(row["attn"])

    print("\nMean attention by class:")
    for class_name in sorted(class_groups.keys()):
        means = np.mean(np.array(class_groups[class_name]), axis=0)
        joined = ", ".join(f"{label}={value:.4f}" for label, value in zip(gate_labels, means))
        print(f"  {class_name}: {joined}")


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Attention CSV not found: {CSV_PATH}")

    rows = load_rows(CSV_PATH)
    if not rows:
        raise RuntimeError("No rows found in attention CSV.")

    plot_sample_attention(rows, ARTIFACT_DIR / "attention_per_sample.png")
    plot_gate_means(rows, ARTIFACT_DIR / "attention_mean_by_gate.png")
    print_summary(rows)
    print(f"\nSaved figures to: {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


GATE_PATTERN = re.compile(r"^(?P<sample_id>.+?)_gate_(?P<gate>\d+)\.png$")
FEATURES = [
    "mean_all",
    "foreground_ratio",
    "foreground_mean",
    "p95",
    "p99",
    "max_value",
    "bbox_area_ratio",
    "bbox_aspect",
    "edge_density",
]


@dataclass(frozen=True)
class SingleGateFeatureRow:
    class_name: str
    sample_id: str
    gate: int
    path: str
    mean_all: float
    foreground_ratio: float
    foreground_mean: float
    p95: float
    p99: float
    max_value: float
    bbox_area_ratio: float
    bbox_aspect: float
    edge_density: float


@dataclass(frozen=True)
class ClassFeatureSummaryRow:
    class_name: str
    gate: int
    num_images: int
    feature: str
    mean: float
    std: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class FeatureSeparabilityRow:
    gate: int
    feature: str
    best_threshold: float
    direction: str
    best_accuracy: float
    class0: str
    class1: str
    class0_mean: float
    class1_mean: float
    cohen_d: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether simple single-gate scalar features can separate true3d and flat_false samples."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sample-csv-out", type=Path, required=True)
    parser.add_argument("--class-csv-out", type=Path, required=True)
    parser.add_argument("--separability-csv-out", type=Path, required=True)
    parser.add_argument("--classes", nargs=2, default=["true3d", "flat_false"])
    parser.add_argument("--foreground-threshold", type=float, default=8.0 / 255.0)
    parser.add_argument("--edge-threshold", type=float, default=0.04)
    return parser.parse_args()


def read_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def parse_gate_path(path: Path) -> tuple[str, int] | None:
    match = GATE_PATTERN.match(path.name)
    if not match:
        return None
    return match.group("sample_id"), int(match.group("gate"))


def bbox_features(mask: np.ndarray) -> tuple[float, float]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return 0.0, 0.0
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    height = max(1, int(y1 - y0))
    width = max(1, int(x1 - x0))
    area_ratio = float(width * height) / float(mask.shape[0] * mask.shape[1])
    aspect = float(width) / float(height)
    return area_ratio, aspect


def edge_density(arr: np.ndarray, active_mask: np.ndarray, edge_threshold: float) -> float:
    if not active_mask.any():
        return 0.0
    grad_y = np.abs(np.diff(arr, axis=0, prepend=arr[:1, :]))
    grad_x = np.abs(np.diff(arr, axis=1, prepend=arr[:, :1]))
    edges = (grad_x + grad_y) > edge_threshold
    return float((edges & active_mask).sum()) / float(active_mask.sum())


def analyze_image(class_name: str, path: Path, foreground_threshold: float, edge_threshold: float) -> SingleGateFeatureRow:
    parsed = parse_gate_path(path)
    if parsed is None:
        raise ValueError(f"Not a gate image: {path}")
    sample_id, gate = parsed
    arr = read_gray(path)
    foreground = arr > foreground_threshold
    bbox_area_ratio, bbox_aspect = bbox_features(foreground)
    return SingleGateFeatureRow(
        class_name=class_name,
        sample_id=sample_id,
        gate=gate,
        path=str(path),
        mean_all=float(arr.mean()),
        foreground_ratio=float(foreground.mean()),
        foreground_mean=float(arr[foreground].mean()) if foreground.any() else 0.0,
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
        max_value=float(arr.max()),
        bbox_area_ratio=bbox_area_ratio,
        bbox_aspect=bbox_aspect,
        edge_density=edge_density(arr, foreground, edge_threshold),
    )


def collect_rows(root: Path, classes: list[str], foreground_threshold: float, edge_threshold: float) -> list[SingleGateFeatureRow]:
    rows: list[SingleGateFeatureRow] = []
    for class_name in classes:
        class_dir = root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for path in sorted(class_dir.glob("*.png")):
            if parse_gate_path(path) is None:
                continue
            rows.append(analyze_image(class_name, path, foreground_threshold, edge_threshold))
    if not rows:
        raise RuntimeError(f"No gate images found under {root}")
    return rows


def summarize_by_class(rows: list[SingleGateFeatureRow]) -> list[ClassFeatureSummaryRow]:
    summaries: list[ClassFeatureSummaryRow] = []
    keys = sorted({(row.class_name, row.gate) for row in rows})
    for class_name, gate in keys:
        subset = [row for row in rows if row.class_name == class_name and row.gate == gate]
        for feature in FEATURES:
            values = np.asarray([float(getattr(row, feature)) for row in subset], dtype=np.float64)
            summaries.append(
                ClassFeatureSummaryRow(
                    class_name=class_name,
                    gate=gate,
                    num_images=len(subset),
                    feature=feature,
                    mean=float(values.mean()),
                    std=float(values.std(ddof=0)),
                    minimum=float(values.min()),
                    maximum=float(values.max()),
                )
            )
    return summaries


def threshold_accuracy(values0: np.ndarray, values1: np.ndarray) -> tuple[float, float, str]:
    values = np.concatenate([values0, values1])
    thresholds = np.unique(values)
    if thresholds.size > 1:
        thresholds = (thresholds[:-1] + thresholds[1:]) / 2.0
    candidates = np.concatenate([thresholds, [values.min() - 1e-9, values.max() + 1e-9]])

    best_accuracy = -1.0
    best_threshold = float(candidates[0])
    best_direction = "class1_if_ge"
    for threshold in candidates:
        acc_ge = (np.count_nonzero(values0 < threshold) + np.count_nonzero(values1 >= threshold)) / float(values.size)
        if acc_ge > best_accuracy:
            best_accuracy = float(acc_ge)
            best_threshold = float(threshold)
            best_direction = "class1_if_ge"
        acc_lt = (np.count_nonzero(values0 >= threshold) + np.count_nonzero(values1 < threshold)) / float(values.size)
        if acc_lt > best_accuracy:
            best_accuracy = float(acc_lt)
            best_threshold = float(threshold)
            best_direction = "class1_if_lt"
    return best_threshold, best_accuracy, best_direction


def cohen_d(values0: np.ndarray, values1: np.ndarray) -> float:
    pooled = np.sqrt((values0.var(ddof=0) + values1.var(ddof=0)) / 2.0)
    if pooled <= 1e-12:
        return 0.0
    return float((values1.mean() - values0.mean()) / pooled)


def separability(rows: list[SingleGateFeatureRow], classes: list[str]) -> list[FeatureSeparabilityRow]:
    output: list[FeatureSeparabilityRow] = []
    gates = sorted({row.gate for row in rows})
    class0, class1 = classes
    for gate in gates:
        rows0 = [row for row in rows if row.gate == gate and row.class_name == class0]
        rows1 = [row for row in rows if row.gate == gate and row.class_name == class1]
        if not rows0 or not rows1:
            continue
        for feature in FEATURES:
            values0 = np.asarray([float(getattr(row, feature)) for row in rows0], dtype=np.float64)
            values1 = np.asarray([float(getattr(row, feature)) for row in rows1], dtype=np.float64)
            threshold, accuracy, direction = threshold_accuracy(values0, values1)
            output.append(
                FeatureSeparabilityRow(
                    gate=gate,
                    feature=feature,
                    best_threshold=threshold,
                    direction=direction,
                    best_accuracy=accuracy,
                    class0=class0,
                    class1=class1,
                    class0_mean=float(values0.mean()),
                    class1_mean=float(values1.mean()),
                    cohen_d=cohen_d(values0, values1),
                )
            )
    return output


def write_csv(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    fields = list(rows[0].__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def main() -> int:
    args = parse_args()
    if not args.root.exists():
        raise FileNotFoundError(f"Root does not exist: {args.root}")
    rows = collect_rows(args.root, args.classes, args.foreground_threshold, args.edge_threshold)
    summaries = summarize_by_class(rows)
    separability_rows = separability(rows, args.classes)
    write_csv(args.sample_csv_out, rows)
    write_csv(args.class_csv_out, summaries)
    write_csv(args.separability_csv_out, separability_rows)

    print(f"images={len(rows)}")
    print(f"sample_csv={args.sample_csv_out}")
    print(f"class_csv={args.class_csv_out}")
    print(f"separability_csv={args.separability_csv_out}")
    for row in sorted(separability_rows, key=lambda item: (-item.best_accuracy, item.gate, item.feature))[:8]:
        print(
            f"gate={row.gate} feature={row.feature} acc={row.best_accuracy:.4f} "
            f"means=({row.class0_mean:.4f},{row.class1_mean:.4f}) d={row.cohen_d:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

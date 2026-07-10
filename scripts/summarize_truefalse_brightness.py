from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


GATE_PATTERN = re.compile(r"_gate_(?P<gate>\d+)\.png$")


@dataclass(frozen=True)
class BrightnessRow:
    dataset: str
    class_name: str
    gate: int
    num_images: int
    mean_all: float
    foreground_ratio: float
    foreground_mean: float
    p95: float
    p99: float
    mean_max: float
    saturated_ratio_098: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize brightness statistics for true3d/flat_false gate datasets.")
    parser.add_argument("--roots", type=Path, nargs="+", required=True, help="Dataset roots containing true3d/flat_false dirs.")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--foreground-threshold", type=float, default=8.0 / 255.0)
    parser.add_argument("--saturation-threshold", type=float, default=0.98)
    return parser.parse_args()


def read_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def gate_from_path(path: Path) -> int | None:
    match = GATE_PATTERN.search(path.name)
    if not match:
        return None
    return int(match.group("gate"))


def summarize_gate(
    dataset_name: str,
    class_name: str,
    gate: int,
    paths: list[Path],
    foreground_threshold: float,
    saturation_threshold: float,
) -> BrightnessRow:
    means: list[float] = []
    foreground_ratios: list[float] = []
    foreground_means: list[float] = []
    p95s: list[float] = []
    p99s: list[float] = []
    maxes: list[float] = []
    saturated: list[float] = []

    for path in paths:
        arr = read_gray(path)
        foreground = arr > foreground_threshold
        means.append(float(arr.mean()))
        foreground_ratios.append(float(foreground.mean()))
        foreground_means.append(float(arr[foreground].mean()) if foreground.any() else 0.0)
        p95s.append(float(np.percentile(arr, 95)))
        p99s.append(float(np.percentile(arr, 99)))
        maxes.append(float(arr.max()))
        saturated.append(float((arr > saturation_threshold).mean()))

    return BrightnessRow(
        dataset=dataset_name,
        class_name=class_name,
        gate=gate,
        num_images=len(paths),
        mean_all=float(np.mean(means)),
        foreground_ratio=float(np.mean(foreground_ratios)),
        foreground_mean=float(np.mean(foreground_means)),
        p95=float(np.mean(p95s)),
        p99=float(np.mean(p99s)),
        mean_max=float(np.mean(maxes)),
        saturated_ratio_098=float(np.mean(saturated)),
    )


def summarize_root(root: Path, foreground_threshold: float, saturation_threshold: float) -> list[BrightnessRow]:
    rows: list[BrightnessRow] = []
    for class_name in ["true3d", "flat_false"]:
        class_dir = root / class_name
        if not class_dir.exists():
            continue
        by_gate: dict[int, list[Path]] = {}
        for path in sorted(class_dir.glob("*.png")):
            gate = gate_from_path(path)
            if gate is None:
                continue
            by_gate.setdefault(gate, []).append(path)
        for gate, paths in sorted(by_gate.items()):
            rows.append(
                summarize_gate(
                    dataset_name=root.name,
                    class_name=class_name,
                    gate=gate,
                    paths=paths,
                    foreground_threshold=foreground_threshold,
                    saturation_threshold=saturation_threshold,
                )
            )
    return rows


def write_rows(path: Path, rows: list[BrightnessRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(BrightnessRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def main() -> int:
    args = parse_args()
    rows: list[BrightnessRow] = []
    for root in args.roots:
        if not root.exists():
            raise FileNotFoundError(f"Missing dataset root: {root}")
        rows.extend(summarize_root(root, args.foreground_threshold, args.saturation_threshold))
    if not rows:
        raise RuntimeError("No gate images found under the requested roots.")
    write_rows(args.output_csv, rows)
    print(f"rows={len(rows)}")
    print(f"output_csv={args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


GATE_PATTERN = re.compile(r"^(?P<base>.+?)_gate_(?P<gate>\d+)\.png$")


@dataclass(frozen=True)
class GateQualityRow:
    class_name: str
    sample_id: str
    gate: int
    image_path: str
    max_gray: int
    mean_gray: float
    nonzero_pixels: int
    foreground_pixels: int
    low_contrast: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit brightness/foreground quality of gated PNG datasets.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    parser.add_argument("--foreground-threshold", type=int, default=5)
    parser.add_argument("--low-max-threshold", type=int, default=10)
    return parser.parse_args()


def iter_gate_images(root: Path):
    for class_dir in sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith("_")):
        for path in sorted(class_dir.rglob("*.png")):
            match = GATE_PATTERN.match(path.name)
            if not match:
                continue
            yield class_dir.name, match.group("base"), int(match.group("gate")), path


def audit_image(
    class_name: str,
    sample_id: str,
    gate: int,
    path: Path,
    foreground_threshold: int,
    low_max_threshold: int,
) -> GateQualityRow:
    arr = np.array(Image.open(path).convert("L"))
    max_gray = int(arr.max())
    return GateQualityRow(
        class_name=class_name,
        sample_id=sample_id,
        gate=gate,
        image_path=str(path),
        max_gray=max_gray,
        mean_gray=float(arr.mean()),
        nonzero_pixels=int((arr > 0).sum()),
        foreground_pixels=int((arr > foreground_threshold).sum()),
        low_contrast=max_gray < low_max_threshold,
    )


def write_csv(path: Path, rows: list[GateQualityRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(GateQualityRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def main() -> int:
    args = parse_args()
    if not args.root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {args.root}")

    rows = [
        audit_image(class_name, sample_id, gate, path, args.foreground_threshold, args.low_max_threshold)
        for class_name, sample_id, gate, path in iter_gate_images(args.root)
    ]
    if not rows:
        raise RuntimeError(f"No *_gate_*.png images found under {args.root}")

    write_csv(args.csv_out, rows)
    low_count = sum(row.low_contrast for row in rows)
    sample_count = len({(row.class_name, row.sample_id) for row in rows})
    print(f"images={len(rows)}")
    print(f"samples={sample_count}")
    print(f"low_contrast_images={low_count}")
    print(f"csv={args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

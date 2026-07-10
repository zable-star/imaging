from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


GATE_PATTERN = re.compile(r"^(?P<base>.+?)_gate_(?P<gate>\d+)\.png$")
MODEL_EXTENSIONS = {".glb", ".gltf", ".obj", ".fbx", ".stl", ".off"}


@dataclass(frozen=True)
class ClassReadiness:
    class_name: str
    raw_models: int
    gate_pngs: int
    grouped_samples: int
    valid_samples: int
    incomplete_samples: int
    train_ready: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether a class-folder dataset is ready for gated-slice training. "
            "A valid sample must have the expected number of *_gate_*.png images."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument("--expected-num-slices", type=int, default=3)
    parser.add_argument("--csv-out", type=Path, default=None)
    return parser.parse_args()


def discover_classes(root: Path, class_names: list[str] | None) -> list[str]:
    if class_names is not None:
        return class_names
    return sorted(path.name for path in root.iterdir() if path.is_dir() and not path.name.startswith("_"))


def count_raw_models(class_dir: Path) -> int:
    return sum(1 for path in class_dir.rglob("*") if path.is_file() and path.suffix.lower() in MODEL_EXTENSIONS)


def group_gate_images(class_dir: Path) -> dict[str, set[int]]:
    grouped: dict[str, set[int]] = {}
    for path in class_dir.rglob("*.png"):
        match = GATE_PATTERN.match(path.name)
        if not match:
            continue
        grouped.setdefault(match.group("base"), set()).add(int(match.group("gate")))
    return grouped


def inspect_class(root: Path, class_name: str, expected_num_slices: int) -> ClassReadiness:
    class_dir = root / class_name
    if not class_dir.exists():
        return ClassReadiness(class_name, 0, 0, 0, 0, 0, False)

    grouped = group_gate_images(class_dir)
    valid_samples = sum(1 for gates in grouped.values() if len(gates) == expected_num_slices)
    incomplete_samples = sum(1 for gates in grouped.values() if len(gates) != expected_num_slices)
    gate_pngs = sum(len(gates) for gates in grouped.values())
    return ClassReadiness(
        class_name=class_name,
        raw_models=count_raw_models(class_dir),
        gate_pngs=gate_pngs,
        grouped_samples=len(grouped),
        valid_samples=valid_samples,
        incomplete_samples=incomplete_samples,
        train_ready=valid_samples > 0,
    )


def write_csv(path: Path, rows: list[ClassReadiness]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ClassReadiness.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def print_table(rows: list[ClassReadiness]) -> None:
    headers = [
        "class",
        "raw_models",
        "gate_pngs",
        "grouped_samples",
        "valid_samples",
        "incomplete",
        "ready",
    ]
    print(",".join(headers))
    for row in rows:
        print(
            ",".join(
                [
                    row.class_name,
                    str(row.raw_models),
                    str(row.gate_pngs),
                    str(row.grouped_samples),
                    str(row.valid_samples),
                    str(row.incomplete_samples),
                    str(row.train_ready),
                ]
            )
        )


def main() -> int:
    args = parse_args()
    if args.expected_num_slices <= 0:
        raise ValueError("--expected-num-slices must be positive")
    if not args.root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {args.root}")

    rows = [
        inspect_class(args.root, class_name, args.expected_num_slices)
        for class_name in discover_classes(args.root, args.classes)
    ]
    print_table(rows)
    if args.csv_out is not None:
        write_csv(args.csv_out, rows)
        print(f"Wrote readiness CSV: {args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

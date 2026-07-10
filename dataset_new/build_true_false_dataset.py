from __future__ import annotations

import argparse
import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


GATE_PATTERN = re.compile(r"^(?P<base>.+?)_gate_(?P<gate>\d+)\.png$")


@dataclass(frozen=True)
class GateSample:
    source_kind: str
    source_class: str
    sample_id: str
    paths: dict[int, Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a binary gated-image dataset from selected military true-3D "
            "renders and selected flat-echo false-target renders."
        )
    )
    parser.add_argument("--true-root", type=Path, required=True)
    parser.add_argument("--false-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--true-class-name", default="true3d")
    parser.add_argument("--false-class-name", default="flat_false")
    parser.add_argument("--expected-num-slices", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("._-")
    return value or "sample"


def grouped_samples(root: Path, source_kind: str, expected_num_slices: int) -> list[GateSample]:
    samples: list[GateSample] = []
    for class_dir in sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith("_")):
        grouped: dict[str, dict[int, Path]] = {}
        for path in sorted(class_dir.rglob("*.png")):
            match = GATE_PATTERN.match(path.name)
            if not match:
                continue
            grouped.setdefault(match.group("base"), {})[int(match.group("gate"))] = path
        for sample_id, paths in sorted(grouped.items()):
            if len(paths) != expected_num_slices:
                continue
            samples.append(GateSample(source_kind, class_dir.name, sample_id, paths))
    return samples


def prepare_output_root(output_root: Path, overwrite: bool) -> None:
    resolved = output_root.resolve()
    if overwrite and resolved.exists():
        if len(resolved.parts) < 4:
            raise ValueError(f"Refusing to remove a shallow output path: {resolved}")
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def copy_samples(samples: list[GateSample], class_name: str, output_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    class_dir = output_root / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        output_base = f"{slug(sample.source_class)}__{slug(sample.sample_id)}"
        for gate in sorted(sample.paths):
            source = sample.paths[gate]
            target = class_dir / f"{output_base}_gate_{gate}.png"
            shutil.copy2(source, target)
        rows.append(
            {
                "binary_class": class_name,
                "source_kind": sample.source_kind,
                "source_class": sample.source_class,
                "source_sample_id": sample.sample_id,
                "output_sample_id": output_base,
                "num_slices": len(sample.paths),
            }
        )
    return rows


def write_manifest(output_root: Path, rows: list[dict[str, object]]) -> None:
    manifest_path = output_root / "binary_manifest.csv"
    fieldnames = [
        "binary_class",
        "source_kind",
        "source_class",
        "source_sample_id",
        "output_sample_id",
        "num_slices",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.expected_num_slices <= 0:
        raise ValueError("--expected-num-slices must be positive")
    if not args.true_root.exists():
        raise FileNotFoundError(f"True render root does not exist: {args.true_root}")
    if not args.false_root.exists():
        raise FileNotFoundError(f"False render root does not exist: {args.false_root}")

    true_samples = grouped_samples(args.true_root, "true3d", args.expected_num_slices)
    false_samples = grouped_samples(args.false_root, "flat_false", args.expected_num_slices)
    if not true_samples:
        raise RuntimeError(f"No valid true-3D samples found under {args.true_root}")
    if not false_samples:
        raise RuntimeError(f"No valid flat-false samples found under {args.false_root}")

    prepare_output_root(args.output_root, args.overwrite)
    rows = []
    rows.extend(copy_samples(true_samples, args.true_class_name, args.output_root))
    rows.extend(copy_samples(false_samples, args.false_class_name, args.output_root))
    write_manifest(args.output_root, rows)

    print(f"true_samples={len(true_samples)}")
    print(f"false_samples={len(false_samples)}")
    print(f"output_root={args.output_root}")
    print(f"manifest={args.output_root / 'binary_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

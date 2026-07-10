from __future__ import annotations

import argparse
import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


GATE_PATTERN = re.compile(r"^(?P<base>.+?)_gate_(?P<gate>\d+)\.png$")


@dataclass(frozen=True)
class NormalizedImageRow:
    class_name: str
    sample_id: str
    gate: int
    source_path: str
    output_path: str
    source_max: int
    output_max: int
    mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a normalized copy of a gated PNG dataset. "
            "Use per-gate-max to reduce absolute brightness shortcuts in true/false target experiments."
        )
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=["per-gate-max", "stack-max"], default="per-gate-max")
    parser.add_argument("--target-max", type=int, default=180)
    parser.add_argument("--min-source-max", type=int, default=2)
    parser.add_argument("--manifest-out", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def group_gate_images(class_dir: Path) -> dict[str, dict[int, Path]]:
    grouped: dict[str, dict[int, Path]] = {}
    for path in sorted(class_dir.rglob("*.png")):
        match = GATE_PATTERN.match(path.name)
        if not match:
            continue
        grouped.setdefault(match.group("base"), {})[int(match.group("gate"))] = path
    return grouped


def prepare_output_root(output_root: Path, overwrite: bool) -> None:
    resolved = output_root.resolve()
    if overwrite and resolved.exists():
        if len(resolved.parts) < 4:
            raise ValueError(f"Refusing to remove a shallow output path: {resolved}")
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def scale_image(arr: np.ndarray, scale_max: int, target_max: int, min_source_max: int) -> np.ndarray:
    if scale_max < min_source_max:
        return arr.copy()
    scale = float(target_max) / float(scale_max)
    return np.clip(np.rint(arr.astype(np.float32) * scale), 0, 255).astype(np.uint8)


def normalize_sample(
    class_name: str,
    sample_id: str,
    paths: dict[int, Path],
    output_dir: Path,
    mode: str,
    target_max: int,
    min_source_max: int,
) -> list[NormalizedImageRow]:
    arrays: dict[int, np.ndarray] = {}
    source_maxes: dict[int, int] = {}
    for gate, path in paths.items():
        arr = np.array(Image.open(path).convert("L"))
        arrays[gate] = arr
        source_maxes[gate] = int(arr.max())

    stack_max = max(source_maxes.values()) if source_maxes else 0
    rows: list[NormalizedImageRow] = []
    for gate, path in sorted(paths.items()):
        scale_max = source_maxes[gate] if mode == "per-gate-max" else stack_max
        out_arr = scale_image(arrays[gate], scale_max, target_max, min_source_max)
        out_path = output_dir / path.name
        Image.fromarray(out_arr, mode="L").save(out_path)
        rows.append(
            NormalizedImageRow(
                class_name=class_name,
                sample_id=sample_id,
                gate=gate,
                source_path=str(path),
                output_path=str(out_path),
                source_max=source_maxes[gate],
                output_max=int(out_arr.max()),
                mode=mode,
            )
        )
    return rows


def write_manifest(path: Path, rows: list[NormalizedImageRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(NormalizedImageRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def main() -> int:
    args = parse_args()
    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {args.input_root}")
    if args.target_max <= 0 or args.target_max > 255:
        raise ValueError("--target-max must be in [1, 255]")
    if args.min_source_max < 0:
        raise ValueError("--min-source-max must be non-negative")

    prepare_output_root(args.output_root, args.overwrite)
    rows: list[NormalizedImageRow] = []
    for class_dir in sorted(path for path in args.input_root.iterdir() if path.is_dir() and not path.name.startswith("_")):
        out_class_dir = args.output_root / class_dir.name
        out_class_dir.mkdir(parents=True, exist_ok=True)
        for sample_id, paths in sorted(group_gate_images(class_dir).items()):
            rows.extend(
                normalize_sample(
                    class_dir.name,
                    sample_id,
                    paths,
                    out_class_dir,
                    args.mode,
                    args.target_max,
                    args.min_source_max,
                )
            )

    if not rows:
        raise RuntimeError(f"No gated PNG images found under {args.input_root}")

    manifest_out = args.manifest_out or args.output_root / "normalization_manifest.csv"
    write_manifest(manifest_out, rows)
    print(f"images={len(rows)}")
    print(f"samples={len({(row.class_name, row.sample_id) for row in rows})}")
    print(f"mode={args.mode}")
    print(f"output_root={args.output_root}")
    print(f"manifest={manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

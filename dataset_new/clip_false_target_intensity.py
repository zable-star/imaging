from __future__ import annotations

import argparse
import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


GATE_PATTERN = re.compile(r"^(?P<sample_id>.+?)_gate_(?P<gate>\d+)\.png$")
STATS = ["max_value", "p99"]


@dataclass(frozen=True)
class ClipIntensityRow:
    class_name: str
    sample_id: str
    gate: int
    source_path: str
    output_path: str
    clip_stat: str
    clip_scope: str
    source_stat: float
    clip_value: float
    output_stat: float
    clipped_pixel_fraction: float
    copied: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy true3d unchanged and clip flat_false high-intensity pixels to true3d gate-level statistics. "
            "Unlike exposure scaling, clipping avoids globally dimming the whole false target."
        )
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--true-class", default="true3d")
    parser.add_argument("--false-class", default="flat_false")
    parser.add_argument("--clip-stat", choices=STATS, default="max_value")
    parser.add_argument("--clip-scope", choices=["class-gate-mean"], default="class-gate-mean")
    parser.add_argument("--foreground-threshold", type=float, default=8.0 / 255.0)
    parser.add_argument("--manifest-out", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_output_root(output_root: Path, overwrite: bool) -> None:
    resolved = output_root.resolve()
    if overwrite and resolved.exists():
        if len(resolved.parts) < 4:
            raise ValueError(f"Refusing to remove a shallow output path: {resolved}")
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def parse_gate_path(path: Path) -> tuple[str, int] | None:
    match = GATE_PATTERN.match(path.name)
    if not match:
        return None
    return match.group("sample_id"), int(match.group("gate"))


def collect_gate_images(class_dir: Path) -> dict[int, list[Path]]:
    by_gate: dict[int, list[Path]] = {}
    for path in sorted(class_dir.glob("*.png")):
        parsed = parse_gate_path(path)
        if parsed is None:
            continue
        _, gate = parsed
        by_gate.setdefault(gate, []).append(path)
    return by_gate


def read_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def image_stat(arr: np.ndarray, stat: str) -> float:
    if stat == "max_value":
        return float(arr.max())
    if stat == "p99":
        return float(np.percentile(arr, 99))
    raise ValueError(f"Unsupported stat: {stat}")


def target_clip_values(true_dir: Path, stat: str) -> dict[int, float]:
    targets: dict[int, float] = {}
    for gate, paths in sorted(collect_gate_images(true_dir).items()):
        values = [image_stat(read_gray(path), stat) for path in paths]
        targets[gate] = float(np.mean(values))
    if not targets:
        raise RuntimeError(f"No true gate images found under {true_dir}")
    return targets


def copy_true_image(source_path: Path, output_path: Path, clip_stat: str) -> ClipIntensityRow:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, output_path)
    arr = read_gray(source_path)
    parsed = parse_gate_path(source_path)
    if parsed is None:
        raise ValueError(f"Not a gate image: {source_path}")
    sample_id, gate = parsed
    stat = image_stat(arr, clip_stat)
    return ClipIntensityRow(
        class_name=source_path.parent.name,
        sample_id=sample_id,
        gate=gate,
        source_path=str(source_path),
        output_path=str(output_path),
        clip_stat=clip_stat,
        clip_scope="copied",
        source_stat=stat,
        clip_value=stat,
        output_stat=stat,
        clipped_pixel_fraction=0.0,
        copied=True,
    )


def clip_false_image(
    source_path: Path,
    output_path: Path,
    clip_value: float,
    clip_stat: str,
    clip_scope: str,
) -> ClipIntensityRow:
    arr = read_gray(source_path)
    clipped = np.minimum(arr, clip_value)
    output_u8 = np.clip(np.rint(clipped * 255.0), 0, 255).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output_u8, mode="L").save(output_path)

    parsed = parse_gate_path(source_path)
    if parsed is None:
        raise ValueError(f"Not a gate image: {source_path}")
    sample_id, gate = parsed
    output_arr = np.asarray(output_u8, dtype=np.float32) / 255.0
    return ClipIntensityRow(
        class_name=source_path.parent.name,
        sample_id=sample_id,
        gate=gate,
        source_path=str(source_path),
        output_path=str(output_path),
        clip_stat=clip_stat,
        clip_scope=clip_scope,
        source_stat=image_stat(arr, clip_stat),
        clip_value=float(clip_value),
        output_stat=image_stat(output_arr, clip_stat),
        clipped_pixel_fraction=float((arr > clip_value).mean()),
        copied=False,
    )


def write_manifest(path: Path, rows: list[ClipIntensityRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ClipIntensityRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def main() -> int:
    args = parse_args()
    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {args.input_root}")
    true_dir = args.input_root / args.true_class
    false_dir = args.input_root / args.false_class
    if not true_dir.exists() or not false_dir.exists():
        raise FileNotFoundError("Input root must contain true and false class directories.")

    prepare_output_root(args.output_root, args.overwrite)
    targets = target_clip_values(true_dir, args.clip_stat)
    rows: list[ClipIntensityRow] = []

    for source_path in sorted(true_dir.glob("*.png")):
        if parse_gate_path(source_path) is None:
            continue
        rows.append(copy_true_image(source_path, args.output_root / args.true_class / source_path.name, args.clip_stat))

    for source_path in sorted(false_dir.glob("*.png")):
        parsed = parse_gate_path(source_path)
        if parsed is None:
            continue
        _, gate = parsed
        rows.append(
            clip_false_image(
                source_path,
                args.output_root / args.false_class / source_path.name,
                targets[gate],
                args.clip_stat,
                args.clip_scope,
            )
        )

    if not rows:
        raise RuntimeError(f"No gate images found under {args.input_root}")
    manifest_out = args.manifest_out or args.output_root / "intensity_clip_manifest.csv"
    write_manifest(manifest_out, rows)
    print(f"images={len(rows)}")
    print(f"output_root={args.output_root}")
    print(f"clip_stat={args.clip_stat}")
    print(f"clip_scope={args.clip_scope}")
    print(f"manifest={manifest_out}")
    for gate, value in sorted(targets.items()):
        print(f"gate_{gate}_{args.clip_stat}_clip={value:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

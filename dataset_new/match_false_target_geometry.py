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


@dataclass(frozen=True)
class GeometryMatchRow:
    class_name: str
    sample_id: str
    gate: int
    source_path: str
    target_path: str
    output_path: str
    match_scope: str
    foreground_threshold: float
    source_foreground_pixels: int
    target_foreground_pixels: int
    output_foreground_pixels: int
    source_foreground_ratio: float
    target_foreground_ratio: float
    output_foreground_ratio: float
    copied: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a stricter diagnostic control dataset by reducing flat_false foreground area "
            "toward true3d same-gate foreground area while copying true3d unchanged."
        )
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--true-class", default="true3d")
    parser.add_argument("--false-class", default="flat_false")
    parser.add_argument(
        "--match-scope",
        choices=["class-gate-mean", "paired-image"],
        default="class-gate-mean",
        help=(
            "class-gate-mean uses the mean true foreground area for each gate; "
            "paired-image uses a deterministic same-gate true image."
        ),
    )
    parser.add_argument("--foreground-threshold", type=float, default=8.0 / 255.0)
    parser.add_argument(
        "--allow-grow",
        action="store_true",
        help="If set, keep the source mask when the false foreground is smaller than the target. No dilation is performed.",
    )
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


def foreground_mask(arr: np.ndarray, foreground_threshold: float) -> np.ndarray:
    return arr > foreground_threshold


def erode_once(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    neighbors = [
        padded[0:-2, 0:-2],
        padded[0:-2, 1:-1],
        padded[0:-2, 2:],
        padded[1:-1, 0:-2],
        padded[1:-1, 1:-1],
        padded[1:-1, 2:],
        padded[2:, 0:-2],
        padded[2:, 1:-1],
        padded[2:, 2:],
    ]
    out = neighbors[0].copy()
    for neighbor in neighbors[1:]:
        out &= neighbor
    return out


def chessboard_distance(mask: np.ndarray) -> np.ndarray:
    distance = np.zeros(mask.shape, dtype=np.int32)
    active = mask.copy()
    depth = 1
    while active.any():
        distance[active] = depth
        active = erode_once(active)
        depth += 1
    return distance


def area_matched_mask(arr: np.ndarray, foreground_threshold: float, target_pixels: int) -> np.ndarray:
    mask = foreground_mask(arr, foreground_threshold)
    source_pixels = int(mask.sum())
    if target_pixels <= 0 or source_pixels == 0:
        return np.zeros_like(mask, dtype=bool)
    if target_pixels >= source_pixels:
        return mask

    distance = chessboard_distance(mask)
    yy, xx = np.indices(mask.shape)
    # Keep central, brighter pixels first; x/y tie-breakers make the selection deterministic.
    order = np.lexsort(
        (
            xx[mask].ravel(),
            yy[mask].ravel(),
            -arr[mask].ravel(),
            -distance[mask].ravel(),
        )
    )
    coords = np.argwhere(mask)
    selected = coords[order[:target_pixels]]
    out = np.zeros_like(mask, dtype=bool)
    out[selected[:, 0], selected[:, 1]] = True
    return out


def true_target_pixel_counts(true_dir: Path, foreground_threshold: float) -> dict[int, int]:
    targets: dict[int, int] = {}
    for gate, paths in sorted(collect_gate_images(true_dir).items()):
        counts = [int(foreground_mask(read_gray(path), foreground_threshold).sum()) for path in paths]
        targets[gate] = int(round(float(np.mean(counts))))
    if not targets:
        raise RuntimeError(f"No true gate images found under {true_dir}")
    return targets


def build_paired_targets(true_dir: Path, false_dir: Path) -> dict[Path, Path]:
    true_by_gate = collect_gate_images(true_dir)
    false_by_gate = collect_gate_images(false_dir)
    pairs: dict[Path, Path] = {}
    for gate, false_paths in false_by_gate.items():
        true_paths = true_by_gate.get(gate, [])
        if not true_paths:
            raise ValueError(f"No true image available for gate {gate}")
        for index, false_path in enumerate(false_paths):
            pairs[false_path] = true_paths[index % len(true_paths)]
    return pairs


def copy_true_image(source_path: Path, output_path: Path, foreground_threshold: float) -> GeometryMatchRow:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, output_path)
    arr = read_gray(source_path)
    mask = foreground_mask(arr, foreground_threshold)
    parsed = parse_gate_path(source_path)
    if parsed is None:
        raise ValueError(f"Not a gate image: {source_path}")
    sample_id, gate = parsed
    pixels = int(mask.sum())
    ratio = float(mask.mean())
    return GeometryMatchRow(
        class_name=source_path.parent.name,
        sample_id=sample_id,
        gate=gate,
        source_path=str(source_path),
        target_path=str(source_path),
        output_path=str(output_path),
        match_scope="copied",
        foreground_threshold=foreground_threshold,
        source_foreground_pixels=pixels,
        target_foreground_pixels=pixels,
        output_foreground_pixels=pixels,
        source_foreground_ratio=ratio,
        target_foreground_ratio=ratio,
        output_foreground_ratio=ratio,
        copied=True,
    )


def match_false_image(
    source_path: Path,
    output_path: Path,
    target_pixels: int,
    target_path: Path | None,
    match_scope: str,
    foreground_threshold: float,
    allow_grow: bool,
) -> GeometryMatchRow:
    arr = read_gray(source_path)
    source_mask = foreground_mask(arr, foreground_threshold)
    source_pixels = int(source_mask.sum())
    effective_target = target_pixels if allow_grow else min(target_pixels, source_pixels)
    output_mask = area_matched_mask(arr, foreground_threshold, effective_target)
    out = np.where(output_mask, arr, 0.0)
    output_u8 = np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output_u8, mode="L").save(output_path)

    parsed = parse_gate_path(source_path)
    if parsed is None:
        raise ValueError(f"Not a gate image: {source_path}")
    sample_id, gate = parsed
    total_pixels = float(arr.size)
    return GeometryMatchRow(
        class_name=source_path.parent.name,
        sample_id=sample_id,
        gate=gate,
        source_path=str(source_path),
        target_path=str(target_path) if target_path is not None else f"true_gate_{gate}_mean",
        output_path=str(output_path),
        match_scope=match_scope,
        foreground_threshold=foreground_threshold,
        source_foreground_pixels=source_pixels,
        target_foreground_pixels=int(target_pixels),
        output_foreground_pixels=int(output_mask.sum()),
        source_foreground_ratio=float(source_pixels / total_pixels),
        target_foreground_ratio=float(target_pixels / total_pixels),
        output_foreground_ratio=float(output_mask.mean()),
        copied=False,
    )


def write_manifest(path: Path, rows: list[GeometryMatchRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(GeometryMatchRow.__dataclass_fields__.keys()))
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
    target_counts = true_target_pixel_counts(true_dir, args.foreground_threshold)
    paired_targets = build_paired_targets(true_dir, false_dir) if args.match_scope == "paired-image" else {}
    rows: list[GeometryMatchRow] = []

    for source_path in sorted(true_dir.glob("*.png")):
        if parse_gate_path(source_path) is None:
            continue
        rows.append(copy_true_image(source_path, args.output_root / args.true_class / source_path.name, args.foreground_threshold))

    for source_path in sorted(false_dir.glob("*.png")):
        parsed = parse_gate_path(source_path)
        if parsed is None:
            continue
        _, gate = parsed
        if args.match_scope == "paired-image":
            target_path = paired_targets[source_path]
            target_pixels = int(foreground_mask(read_gray(target_path), args.foreground_threshold).sum())
        else:
            target_path = None
            target_pixels = target_counts[gate]
        rows.append(
            match_false_image(
                source_path,
                args.output_root / args.false_class / source_path.name,
                target_pixels,
                target_path,
                args.match_scope,
                args.foreground_threshold,
                args.allow_grow,
            )
        )

    if not rows:
        raise RuntimeError(f"No gate images found under {args.input_root}")
    manifest_out = args.manifest_out or args.output_root / "geometry_match_manifest.csv"
    write_manifest(manifest_out, rows)
    print(f"images={len(rows)}")
    print(f"output_root={args.output_root}")
    print(f"match_scope={args.match_scope}")
    print(f"foreground_threshold={args.foreground_threshold:.6f}")
    print(f"manifest={manifest_out}")
    for gate, count in sorted(target_counts.items()):
        print(f"gate_{gate}_target_foreground_pixels={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

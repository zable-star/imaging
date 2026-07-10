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
class HistogramMatchRow:
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
    source_foreground_mean: float
    target_foreground_mean: float
    output_foreground_mean: float
    copied: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a stronger single-frame control dataset. true3d images are copied unchanged; "
            "flat_false foreground intensity histograms are matched to true3d foreground histograms gate-wise."
        )
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--true-class", default="true3d")
    parser.add_argument("--false-class", default="flat_false")
    parser.add_argument(
        "--match-scope",
        choices=["class-gate", "paired-image"],
        default="class-gate",
        help=(
            "class-gate matches each false gate image to the pooled true foreground distribution for that gate; "
            "paired-image matches each false image to a deterministic same-gate true image."
        ),
    )
    parser.add_argument("--foreground-threshold", type=float, default=8.0 / 255.0)
    parser.add_argument("--num-quantiles", type=int, default=256)
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


def foreground_values(arr: np.ndarray, foreground_threshold: float) -> np.ndarray:
    return arr[arr > foreground_threshold].astype(np.float32, copy=False)


def quantile_match_values(source_values: np.ndarray, target_values: np.ndarray, num_quantiles: int) -> np.ndarray:
    if source_values.size == 0 or target_values.size == 0:
        return source_values.copy()
    if num_quantiles < 2:
        raise ValueError("--num-quantiles must be at least 2")

    quantile_count = min(num_quantiles, max(2, source_values.size), max(2, target_values.size))
    quantiles = np.linspace(0.0, 1.0, quantile_count)
    source_quantiles = np.quantile(source_values, quantiles)
    target_quantiles = np.quantile(target_values, quantiles)

    unique_source, unique_indices = np.unique(source_quantiles, return_index=True)
    unique_target = target_quantiles[unique_indices]
    if unique_source.size == 1:
        return np.full_like(source_values, float(np.median(target_values)))
    return np.interp(source_values, unique_source, unique_target).astype(np.float32)


def pooled_true_foregrounds(
    true_dir: Path,
    foreground_threshold: float,
) -> dict[int, np.ndarray]:
    pooled: dict[int, list[np.ndarray]] = {}
    for gate, paths in collect_gate_images(true_dir).items():
        pooled[gate] = [foreground_values(read_gray(path), foreground_threshold) for path in paths]
    if not pooled:
        raise RuntimeError(f"No true gate images found under {true_dir}")
    return {
        gate: np.concatenate([values for values in groups if values.size > 0])
        if any(values.size > 0 for values in groups)
        else np.asarray([], dtype=np.float32)
        for gate, groups in pooled.items()
    }


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


def copy_true_image(source_path: Path, output_path: Path, foreground_threshold: float) -> HistogramMatchRow:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, output_path)
    arr = read_gray(source_path)
    values = foreground_values(arr, foreground_threshold)
    parsed = parse_gate_path(source_path)
    if parsed is None:
        raise ValueError(f"Not a gate image: {source_path}")
    sample_id, gate = parsed
    mean_value = float(values.mean()) if values.size else 0.0
    return HistogramMatchRow(
        class_name=source_path.parent.name,
        sample_id=sample_id,
        gate=gate,
        source_path=str(source_path),
        target_path=str(source_path),
        output_path=str(output_path),
        match_scope="copied",
        foreground_threshold=foreground_threshold,
        source_foreground_pixels=int(values.size),
        target_foreground_pixels=int(values.size),
        source_foreground_mean=mean_value,
        target_foreground_mean=mean_value,
        output_foreground_mean=mean_value,
        copied=True,
    )


def match_false_image(
    source_path: Path,
    output_path: Path,
    target_values: np.ndarray,
    target_path: Path | None,
    match_scope: str,
    foreground_threshold: float,
    num_quantiles: int,
) -> HistogramMatchRow:
    arr = read_gray(source_path)
    mask = arr > foreground_threshold
    source_values = arr[mask].astype(np.float32, copy=False)
    matched_values = quantile_match_values(source_values, target_values, num_quantiles)
    out = arr.copy()
    out[mask] = matched_values
    output_u8 = np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output_u8, mode="L").save(output_path)

    parsed = parse_gate_path(source_path)
    if parsed is None:
        raise ValueError(f"Not a gate image: {source_path}")
    sample_id, gate = parsed
    output_arr = np.asarray(output_u8, dtype=np.float32) / 255.0
    output_values = output_arr[mask]
    return HistogramMatchRow(
        class_name=source_path.parent.name,
        sample_id=sample_id,
        gate=gate,
        source_path=str(source_path),
        target_path=str(target_path) if target_path is not None else f"pooled_true_gate_{gate}",
        output_path=str(output_path),
        match_scope=match_scope,
        foreground_threshold=foreground_threshold,
        source_foreground_pixels=int(source_values.size),
        target_foreground_pixels=int(target_values.size),
        source_foreground_mean=float(source_values.mean()) if source_values.size else 0.0,
        target_foreground_mean=float(target_values.mean()) if target_values.size else 0.0,
        output_foreground_mean=float(output_values.mean()) if output_values.size else 0.0,
        copied=False,
    )


def write_manifest(path: Path, rows: list[HistogramMatchRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(HistogramMatchRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def main() -> int:
    args = parse_args()
    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {args.input_root}")
    if args.num_quantiles < 2:
        raise ValueError("--num-quantiles must be at least 2")
    true_dir = args.input_root / args.true_class
    false_dir = args.input_root / args.false_class
    if not true_dir.exists() or not false_dir.exists():
        raise FileNotFoundError("Input root must contain true and false class directories.")

    prepare_output_root(args.output_root, args.overwrite)
    pooled_targets = pooled_true_foregrounds(true_dir, args.foreground_threshold)
    paired_targets = build_paired_targets(true_dir, false_dir) if args.match_scope == "paired-image" else {}
    rows: list[HistogramMatchRow] = []

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
            target_values = foreground_values(read_gray(target_path), args.foreground_threshold)
        else:
            target_path = None
            target_values = pooled_targets.get(gate, np.asarray([], dtype=np.float32))
        rows.append(
            match_false_image(
                source_path,
                args.output_root / args.false_class / source_path.name,
                target_values,
                target_path,
                args.match_scope,
                args.foreground_threshold,
                args.num_quantiles,
            )
        )

    if not rows:
        raise RuntimeError(f"No gate images found under {args.input_root}")
    manifest_out = args.manifest_out or args.output_root / "histogram_match_manifest.csv"
    write_manifest(manifest_out, rows)
    print(f"images={len(rows)}")
    print(f"output_root={args.output_root}")
    print(f"match_scope={args.match_scope}")
    print(f"foreground_threshold={args.foreground_threshold:.6f}")
    print(f"manifest={manifest_out}")
    for gate, values in sorted(pooled_targets.items()):
        print(f"gate_{gate}_true_foreground_pixels={values.size} mean={values.mean() if values.size else 0.0:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

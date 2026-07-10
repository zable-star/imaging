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
STATS = ["mean_all", "foreground_mean", "p99", "max_value"]


@dataclass(frozen=True)
class ExposureMatchRow:
    class_name: str
    sample_id: str
    gate: int
    source_path: str
    output_path: str
    match_stat: str
    match_scope: str
    source_stat: float
    target_stat: float
    scale: float
    source_mean_all: float
    output_mean_all: float
    copied: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an exposure-matched true3d/flat_false dataset. true3d images are copied unchanged; "
            "flat_false images are scaled gate-wise to match the true3d gate-level brightness statistic."
        )
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--true-class", default="true3d")
    parser.add_argument("--false-class", default="flat_false")
    parser.add_argument("--match-stat", choices=STATS, default="mean_all")
    parser.add_argument(
        "--match-scope",
        choices=["class-gate", "per-image"],
        default="class-gate",
        help=(
            "class-gate uses one scale per false-target gate to match the true-class gate statistic; "
            "per-image scales each false image toward the true-class gate statistic."
        ),
    )
    parser.add_argument("--foreground-threshold", type=float, default=8.0 / 255.0)
    parser.add_argument("--max-scale", type=float, default=8.0)
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


def gate_from_path(path: Path) -> tuple[str, int] | None:
    match = GATE_PATTERN.match(path.name)
    if not match:
        return None
    return match.group("base"), int(match.group("gate"))


def collect_gate_images(class_dir: Path) -> dict[int, list[Path]]:
    by_gate: dict[int, list[Path]] = {}
    for path in sorted(class_dir.glob("*.png")):
        parsed = gate_from_path(path)
        if parsed is None:
            continue
        _, gate = parsed
        by_gate.setdefault(gate, []).append(path)
    return by_gate


def read_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def image_stat(arr: np.ndarray, stat: str, foreground_threshold: float) -> float:
    if stat == "mean_all":
        return float(arr.mean())
    if stat == "p99":
        return float(np.percentile(arr, 99))
    if stat == "max_value":
        return float(arr.max())
    foreground = arr > foreground_threshold
    return float(arr[foreground].mean()) if foreground.any() else 0.0


def target_stats(true_dir: Path, stat: str, foreground_threshold: float) -> dict[int, float]:
    targets: dict[int, float] = {}
    for gate, paths in sorted(collect_gate_images(true_dir).items()):
        values = [image_stat(read_gray(path), stat, foreground_threshold) for path in paths]
        targets[gate] = float(np.mean(values))
    if not targets:
        raise RuntimeError(f"No true-class gate images found under {true_dir}")
    return targets


def source_stats_by_gate(class_dir: Path, stat: str, foreground_threshold: float) -> dict[int, float]:
    sources: dict[int, float] = {}
    for gate, paths in sorted(collect_gate_images(class_dir).items()):
        values = [image_stat(read_gray(path), stat, foreground_threshold) for path in paths]
        sources[gate] = float(np.mean(values))
    if not sources:
        raise RuntimeError(f"No gate images found under {class_dir}")
    return sources


def scale_false_image(
    source_path: Path,
    output_path: Path,
    target_stat: float,
    match_stat: str,
    match_scope: str,
    foreground_threshold: float,
    max_scale: float,
    fixed_scale: float | None = None,
) -> ExposureMatchRow:
    arr = read_gray(source_path)
    source_stat = image_stat(arr, match_stat, foreground_threshold)
    if fixed_scale is not None:
        scale = fixed_scale
    elif source_stat <= 1e-8 or target_stat <= 0.0:
        scale = 1.0
    else:
        scale = min(max_scale, target_stat / source_stat)
    out = np.clip(np.rint(arr * scale * 255.0), 0, 255).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out, mode="L").save(output_path)

    parsed = gate_from_path(source_path)
    if parsed is None:
        raise ValueError(f"Not a gate image: {source_path}")
    sample_id, gate = parsed
    return ExposureMatchRow(
        class_name=source_path.parent.name,
        sample_id=sample_id,
        gate=gate,
        source_path=str(source_path),
        output_path=str(output_path),
        match_stat=match_stat,
        match_scope=match_scope,
        source_stat=source_stat,
        target_stat=target_stat,
        scale=float(scale),
        source_mean_all=float(arr.mean()),
        output_mean_all=float(np.asarray(out, dtype=np.float32).mean() / 255.0),
        copied=False,
    )


def copy_true_image(source_path: Path, output_path: Path, match_stat: str, foreground_threshold: float) -> ExposureMatchRow:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, output_path)
    arr = read_gray(source_path)
    parsed = gate_from_path(source_path)
    if parsed is None:
        raise ValueError(f"Not a gate image: {source_path}")
    sample_id, gate = parsed
    stat = image_stat(arr, match_stat, foreground_threshold)
    return ExposureMatchRow(
        class_name=source_path.parent.name,
        sample_id=sample_id,
        gate=gate,
        source_path=str(source_path),
        output_path=str(output_path),
        match_stat=match_stat,
        match_scope="copied",
        source_stat=stat,
        target_stat=stat,
        scale=1.0,
        source_mean_all=float(arr.mean()),
        output_mean_all=float(arr.mean()),
        copied=True,
    )


def write_manifest(path: Path, rows: list[ExposureMatchRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ExposureMatchRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def main() -> int:
    args = parse_args()
    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {args.input_root}")
    if args.max_scale <= 0.0:
        raise ValueError("--max-scale must be positive")
    true_dir = args.input_root / args.true_class
    false_dir = args.input_root / args.false_class
    if not true_dir.exists() or not false_dir.exists():
        raise FileNotFoundError("Input root must contain true and false class directories.")

    prepare_output_root(args.output_root, args.overwrite)
    targets = target_stats(true_dir, args.match_stat, args.foreground_threshold)
    false_sources = source_stats_by_gate(false_dir, args.match_stat, args.foreground_threshold)
    class_gate_scales = {
        gate: (
            1.0
            if false_sources.get(gate, 0.0) <= 1e-8 or target <= 0.0
            else min(args.max_scale, target / false_sources[gate])
        )
        for gate, target in targets.items()
    }
    rows: list[ExposureMatchRow] = []

    for source_path in sorted(true_dir.glob("*.png")):
        parsed = gate_from_path(source_path)
        if parsed is None:
            continue
        rows.append(
            copy_true_image(
                source_path,
                args.output_root / args.true_class / source_path.name,
                args.match_stat,
                args.foreground_threshold,
            )
        )

    for source_path in sorted(false_dir.glob("*.png")):
        parsed = gate_from_path(source_path)
        if parsed is None:
            continue
        _, gate = parsed
        if gate not in targets:
            raise ValueError(f"No target statistic for gate {gate}")
        rows.append(
            scale_false_image(
                source_path,
                args.output_root / args.false_class / source_path.name,
                targets[gate],
                args.match_stat,
                args.match_scope,
                args.foreground_threshold,
                args.max_scale,
                fixed_scale=class_gate_scales[gate] if args.match_scope == "class-gate" else None,
            )
        )

    if not rows:
        raise RuntimeError(f"No gate images found under {args.input_root}")
    manifest_out = args.manifest_out or args.output_root / "exposure_match_manifest.csv"
    write_manifest(manifest_out, rows)
    print(f"images={len(rows)}")
    print(f"output_root={args.output_root}")
    print(f"match_stat={args.match_stat}")
    print(f"match_scope={args.match_scope}")
    print(f"manifest={manifest_out}")
    for gate, value in sorted(targets.items()):
        print(
            f"gate_{gate}_{args.match_stat}: "
            f"true_target={value:.6f} false_source={false_sources.get(gate, 0.0):.6f} "
            f"scale={class_gate_scales.get(gate, 1.0):.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

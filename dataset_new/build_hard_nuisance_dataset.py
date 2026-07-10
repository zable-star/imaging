from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


GATE_PATTERN = re.compile(r"^(?P<base>.+?)_gate_(?P<gate>\d+)\.png$")


@dataclass(frozen=True)
class NuisanceSampleRow:
    class_name: str
    sample_id: str
    nuisance_key: str
    images_written: int
    reflectance_strength: float
    background_strength: float
    occlusion_alpha: float
    occlusion_fraction: float
    foreground_fraction: float
    source_paths: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a hard-nuisance gated dataset by applying deterministic reflectance texture, "
            "weak background scatter, and partial occlusion to both true and false classes."
        )
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--classes", nargs="+", default=["true3d", "flat_false"])
    parser.add_argument("--expected-num-slices", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--foreground-threshold", type=int, default=4)
    parser.add_argument("--reflectance-strength", type=float, default=0.18)
    parser.add_argument("--background-strength", type=float, default=9.0)
    parser.add_argument("--background-gate-jitter", type=float, default=0.18)
    parser.add_argument("--occlusion-probability", type=float, default=0.75)
    parser.add_argument("--occlusion-min-fraction", type=float, default=0.08)
    parser.add_argument("--occlusion-max-fraction", type=float, default=0.18)
    parser.add_argument("--occlusion-alpha", type=float, default=0.35)
    parser.add_argument("--preserve-input-max", action="store_true")
    parser.add_argument("--manifest-out", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def stable_seed(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**32)


def split_group_id(sample_id: str) -> str:
    if sample_id.lower().startswith("view_") and "__" in sample_id:
        return sample_id.split("__", 1)[1]
    return sample_id


def prepare_output_root(output_root: Path, overwrite: bool) -> None:
    resolved = output_root.resolve()
    if overwrite and resolved.exists():
        if len(resolved.parts) < 4:
            raise ValueError(f"Refusing to remove shallow output path: {resolved}")
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def group_gate_images(class_dir: Path) -> dict[str, dict[int, Path]]:
    grouped: dict[str, dict[int, Path]] = {}
    for path in sorted(class_dir.rglob("*.png")):
        match = GATE_PATTERN.match(path.name)
        if not match:
            continue
        grouped.setdefault(match.group("base"), {})[int(match.group("gate"))] = path
    return grouped


def load_stack(paths: dict[int, Path], expected_num_slices: int) -> np.ndarray:
    missing = sorted(set(range(expected_num_slices)) - set(paths))
    if missing:
        raise ValueError(f"Missing gate images: {missing}")
    images = [np.asarray(Image.open(paths[gate]).convert("L"), dtype=np.float32) for gate in range(expected_num_slices)]
    return np.stack(images, axis=0)


def low_frequency_field(rng: np.random.Generator, shape: tuple[int, int], grid_size: int = 9) -> np.ndarray:
    h, w = shape
    coarse = rng.normal(0.0, 1.0, size=(grid_size, grid_size)).astype(np.float32)
    img = Image.fromarray(coarse, mode="F").resize((w, h), resample=Image.Resampling.BICUBIC)
    field = np.asarray(img, dtype=np.float32)
    field = field - float(field.mean())
    std = float(field.std())
    if std > 1e-6:
        field = field / std
    return field


def build_occlusion_mask(
    rng: np.random.Generator,
    shape: tuple[int, int],
    foreground: np.ndarray,
    probability: float,
    min_fraction: float,
    max_fraction: float,
) -> tuple[np.ndarray, float]:
    h, w = shape
    mask = np.ones((h, w), dtype=np.float32)
    if rng.random() > probability or not foreground.any():
        return mask, 0.0

    ys, xs = np.where(foreground)
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    fg_h = max(y_max - y_min + 1, 1)
    fg_w = max(x_max - x_min + 1, 1)
    fraction = float(rng.uniform(min_fraction, max_fraction))
    aspect = float(rng.uniform(0.55, 1.8))
    occ_h = max(4, int(round((fg_h * fraction / max(aspect, 1e-3)) ** 0.5 * fg_h**0.5)))
    occ_w = max(4, int(round(occ_h * aspect)))
    occ_h = min(occ_h, fg_h)
    occ_w = min(occ_w, fg_w)
    cy = int(rng.integers(y_min, y_max + 1))
    cx = int(rng.integers(x_min, x_max + 1))
    y0 = max(y_min, cy - occ_h // 2)
    y1 = min(y_max + 1, y0 + occ_h)
    x0 = max(x_min, cx - occ_w // 2)
    x1 = min(x_max + 1, x0 + occ_w)
    mask[y0:y1, x0:x1] = 0.0
    actual_fraction = float(((mask == 0.0) & foreground).sum() / max(int(foreground.sum()), 1))
    return mask, actual_fraction


def apply_nuisance(stack: np.ndarray, sample_id: str, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, float | str]]:
    nuisance_key = split_group_id(sample_id)
    rng = np.random.default_rng(stable_seed(args.seed, nuisance_key))
    foreground = stack.max(axis=0) > args.foreground_threshold
    h, w = foreground.shape

    reflectance_field = low_frequency_field(rng, (h, w), grid_size=9)
    reflectance = np.clip(1.0 + args.reflectance_strength * reflectance_field, 0.45, 1.65).astype(np.float32)

    background_field = low_frequency_field(rng, (h, w), grid_size=7)
    background = args.background_strength * np.clip(0.5 + 0.22 * background_field, 0.0, 1.0)

    occlusion_mask, occlusion_fraction = build_occlusion_mask(
        rng,
        (h, w),
        foreground,
        args.occlusion_probability,
        args.occlusion_min_fraction,
        args.occlusion_max_fraction,
    )
    occlusion = args.occlusion_alpha + (1.0 - args.occlusion_alpha) * occlusion_mask

    out = []
    input_max = float(stack.max())
    for gate_idx, gate in enumerate(stack):
        gate_scale = 1.0 + float(rng.uniform(-args.background_gate_jitter, args.background_gate_jitter))
        degraded = gate * reflectance * occlusion + background * gate_scale
        if args.preserve_input_max:
            max_value = float(degraded.max())
            if max_value > 1e-6:
                degraded = degraded * (input_max / max_value)
        out.append(np.clip(np.rint(degraded), 0, 255).astype(np.uint8))

    metadata = {
        "nuisance_key": nuisance_key,
        "foreground_fraction": float(foreground.mean()),
        "occlusion_fraction": float(occlusion_fraction),
        "reflectance_strength": float(args.reflectance_strength),
        "background_strength": float(args.background_strength),
        "occlusion_alpha": float(args.occlusion_alpha),
    }
    return np.stack(out, axis=0), metadata


def write_manifest(path: Path, rows: list[NuisanceSampleRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(NuisanceSampleRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def build_dataset(args: argparse.Namespace) -> list[NuisanceSampleRow]:
    rows: list[NuisanceSampleRow] = []
    for class_name in args.classes:
        class_dir = args.input_root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        output_class_dir = args.output_root / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)
        grouped = group_gate_images(class_dir)
        for sample_id, paths in sorted(grouped.items()):
            if len(paths) != args.expected_num_slices:
                continue
            stack = load_stack(paths, args.expected_num_slices)
            out_stack, metadata = apply_nuisance(stack, sample_id, args)
            for gate_idx, arr in enumerate(out_stack):
                Image.fromarray(arr, mode="L").save(output_class_dir / f"{sample_id}_gate_{gate_idx}.png")
            with (output_class_dir / f"{sample_id}_hard_nuisance.json").open("w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            rows.append(
                NuisanceSampleRow(
                    class_name=class_name,
                    sample_id=sample_id,
                    nuisance_key=str(metadata["nuisance_key"]),
                    images_written=args.expected_num_slices,
                    reflectance_strength=float(metadata["reflectance_strength"]),
                    background_strength=float(metadata["background_strength"]),
                    occlusion_alpha=float(metadata["occlusion_alpha"]),
                    occlusion_fraction=float(metadata["occlusion_fraction"]),
                    foreground_fraction=float(metadata["foreground_fraction"]),
                    source_paths=";".join(str(paths[gate]) for gate in sorted(paths)),
                )
            )
    return rows


def main() -> int:
    args = parse_args()
    if args.expected_num_slices <= 0:
        raise ValueError("--expected-num-slices must be positive")
    if not 0.0 <= args.occlusion_probability <= 1.0:
        raise ValueError("--occlusion-probability must be in [0, 1]")
    if not 0.0 <= args.occlusion_alpha <= 1.0:
        raise ValueError("--occlusion-alpha must be in [0, 1]")
    if args.occlusion_min_fraction < 0.0 or args.occlusion_max_fraction < args.occlusion_min_fraction:
        raise ValueError("Invalid occlusion fraction range")
    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {args.input_root}")

    prepare_output_root(args.output_root, args.overwrite)
    rows = build_dataset(args)
    if not rows:
        raise RuntimeError(f"No complete samples found under {args.input_root}")
    manifest = args.manifest_out or args.output_root / "hard_nuisance_manifest.csv"
    write_manifest(manifest, rows)
    print(f"samples={len(rows)}")
    print(f"gate_images={sum(row.images_written for row in rows)}")
    print(f"output_root={args.output_root}")
    print(f"manifest={manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

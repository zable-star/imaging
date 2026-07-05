from __future__ import annotations

import argparse
import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


DEFAULT_DATASET_ROOT = Path(__file__).resolve().parent / "dataset"
DEFAULT_SOURCE_CLASSES = ["chair", "desk", "sofa", "bed", "toilet"]
DEFAULT_TARGET_CLASS = "image2d"
DEFAULT_SAMPLES = 100
DEFAULT_SEED = 42
GATE_COUNT = 3
MODES = ["single-active", "flat-echo"]


@dataclass(frozen=True)
class SourceImage:
    class_name: str
    sample_id: str
    gate_index: int
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a sixth image2d/flat-target class for gated-slice experiments."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--source-classes", nargs="+", default=DEFAULT_SOURCE_CLASSES)
    parser.add_argument("--target-class", default=DEFAULT_TARGET_CLASS)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--gate-count", type=int, default=GATE_COUNT)
    parser.add_argument("--informative-gate", type=int, default=None)
    parser.add_argument("--mode", choices=MODES, default="single-active")
    parser.add_argument(
        "--gate-centers",
        type=str,
        default="0,1,2",
        help="Comma-separated gate centers used by flat-echo mode.",
    )
    parser.add_argument(
        "--receiver-gate-width",
        type=float,
        default=1.0,
        help="Receiver gate width used by flat-echo mode.",
    )
    parser.add_argument(
        "--laser-pulse-width",
        type=float,
        default=0.45,
        help="Laser pulse width used by flat-echo mode.",
    )
    parser.add_argument(
        "--flat-depth-jitter",
        type=float,
        default=0.1,
        help="Random flat target depth jitter as a fraction of gate spacing.",
    )
    parser.add_argument(
        "--min-gate-intensity",
        type=float,
        default=0.20,
        help="Residual response outside the best gate for flat-echo mode.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one value is required.")
    return values


def parse_gate_path(path: Path) -> tuple[str, int] | None:
    stem = path.stem
    marker = "_gate_"
    if marker not in stem:
        return None
    sample_id, gate_text = stem.rsplit(marker, 1)
    if not gate_text.isdigit():
        return None
    return sample_id, int(gate_text)


def collect_source_images(dataset_root: Path, source_classes: list[str], gate_count: int) -> list[SourceImage]:
    images: list[SourceImage] = []
    for class_name in source_classes:
        class_dir = dataset_root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing source class directory: {class_dir}")

        for path in sorted(class_dir.rglob("*.png")):
            parsed = parse_gate_path(path)
            if parsed is None:
                continue
            sample_id, gate_index = parsed
            if 0 <= gate_index < gate_count:
                images.append(SourceImage(class_name, sample_id, gate_index, path))

    if not images:
        raise RuntimeError("No source gate images were found.")
    return images


def make_black_image(reference_path: Path, out_path: Path) -> None:
    with Image.open(reference_path) as img:
        mode = img.mode
        size = img.size
    Image.new(mode, size, color=0).save(out_path)


def gate_response(depth: float, gate_center: float, gate_width: float, pulse_width: float) -> float:
    half_sum = 0.5 * (gate_width + pulse_width)
    overlap_cap = min(gate_width, pulse_width)
    overlap = min(max(half_sum - abs(depth - gate_center), 0.0), overlap_cap)
    return overlap / max(overlap_cap, 1e-6)


def save_scaled_image(source_path: Path, out_path: Path, scale: float) -> None:
    scale = max(0.0, min(float(scale), 1.0))
    with Image.open(source_path) as img:
        gray = img.convert("L")
        out = gray.point(lambda value: int(round(value * scale)))
        out.save(out_path)


def create_image2d_class(
    dataset_root: Path,
    source_classes: list[str],
    target_class: str,
    num_samples: int,
    seed: int,
    gate_count: int = GATE_COUNT,
    informative_gate: int | None = None,
    mode: str = "single-active",
    gate_centers: list[float] | None = None,
    receiver_gate_width: float = 1.0,
    laser_pulse_width: float = 0.45,
    flat_depth_jitter: float = 0.1,
    min_gate_intensity: float = 0.20,
    overwrite: bool = False,
) -> Path:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if informative_gate is not None and not 0 <= informative_gate < gate_count:
        raise ValueError(f"informative_gate must be in [0, {gate_count - 1}], got {informative_gate}")
    if mode not in MODES:
        raise ValueError(f"Unsupported mode: {mode}")
    if gate_centers is None:
        gate_centers = [float(idx) for idx in range(gate_count)]
    if len(gate_centers) < gate_count:
        raise ValueError(f"Expected at least {gate_count} gate centers, got {len(gate_centers)}")
    if receiver_gate_width <= 0 or laser_pulse_width <= 0:
        raise ValueError("receiver_gate_width and laser_pulse_width must be positive")

    target_dir = dataset_root / target_class
    if target_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{target_dir} already exists. Use --overwrite to recreate it.")
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)

    source_images = collect_source_images(dataset_root, source_classes, gate_count)
    rng = random.Random(seed)
    selected = rng.sample(source_images, k=min(num_samples, len(source_images)))
    if len(selected) < num_samples:
        selected.extend(rng.choices(source_images, k=num_samples - len(selected)))

    manifest_path = target_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "target_sample_id",
            "mode",
            "informative_gate",
            "flat_depth",
            "gate_responses",
            "source_class",
            "source_sample_id",
            "source_gate",
            "source_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, source in enumerate(selected):
            active_gate = informative_gate if informative_gate is not None else idx % gate_count
            target_sample_id = f"{target_class}_{idx:04d}"
            flat_depth = ""
            gate_responses: list[float] = []
            if mode == "flat-echo":
                if gate_count > 1:
                    spacing = abs(gate_centers[min(active_gate + 1, gate_count - 1)] - gate_centers[max(active_gate - 1, 0)])
                    spacing = spacing if spacing > 0 else 1.0
                else:
                    spacing = 1.0
                flat_depth = gate_centers[active_gate] + rng.uniform(-flat_depth_jitter, flat_depth_jitter) * spacing
                raw_responses = [
                    gate_response(float(flat_depth), gate_centers[gate_index], receiver_gate_width, laser_pulse_width)
                    for gate_index in range(gate_count)
                ]
                max_response = max(raw_responses) if raw_responses else 1.0
                gate_responses = [
                    max(min_gate_intensity, response / max(max_response, 1e-6))
                    for response in raw_responses
                ]

            for gate_index in range(gate_count):
                out_path = target_dir / f"{target_sample_id}_gate_{gate_index}.png"
                if mode == "flat-echo":
                    save_scaled_image(source.path, out_path, gate_responses[gate_index])
                elif gate_index == active_gate:
                    shutil.copy2(source.path, out_path)
                else:
                    make_black_image(source.path, out_path)

            writer.writerow(
                {
                    "target_sample_id": target_sample_id,
                    "mode": mode,
                    "informative_gate": active_gate,
                    "flat_depth": flat_depth,
                    "gate_responses": ";".join(f"{value:.4f}" for value in gate_responses),
                    "source_class": source.class_name,
                    "source_sample_id": source.sample_id,
                    "source_gate": source.gate_index,
                    "source_path": str(source.path),
                }
            )

    return target_dir


def main() -> int:
    args = parse_args()
    target_dir = create_image2d_class(
        dataset_root=args.dataset_root,
        source_classes=args.source_classes,
        target_class=args.target_class,
        num_samples=args.num_samples,
        seed=args.seed,
        gate_count=args.gate_count,
        informative_gate=args.informative_gate,
        mode=args.mode,
        gate_centers=parse_float_list(args.gate_centers),
        receiver_gate_width=args.receiver_gate_width,
        laser_pulse_width=args.laser_pulse_width,
        flat_depth_jitter=args.flat_depth_jitter,
        min_gate_intensity=args.min_gate_intensity,
        overwrite=args.overwrite,
    )
    print(f"Created {args.num_samples} samples in {target_dir}")
    print(f"Manifest: {target_dir / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

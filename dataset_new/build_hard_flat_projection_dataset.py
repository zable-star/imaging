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
class HardFlatSampleRow:
    source_class: str
    source_sample_id: str
    output_sample_id: str
    projection_mode: str
    response_mode: str
    flat_center_gate: float
    response_sigma: float
    pulse_width: float
    gate_width: float
    gate_spacing: float
    min_response: float
    reflectance: float
    gate_responses: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build harder flat false-target gate stacks from true 3D gated renders. "
            "The false target uses the max/mean projection of the true stack as one planar silhouette, "
            "then applies a flat-depth gate response curve."
        )
    )
    parser.add_argument("--true-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-num-slices", type=int, default=3)
    parser.add_argument("--projection-mode", choices=["max", "mean"], default="max")
    parser.add_argument(
        "--response-mode",
        choices=["gaussian", "rectangular-overlap"],
        default="gaussian",
        help=(
            "Gate response model. gaussian keeps the original smooth response; "
            "rectangular-overlap uses the overlap length between a rectangular laser pulse and a rectangular gate."
        ),
    )
    parser.add_argument("--response-sigma", type=float, default=0.65)
    parser.add_argument("--pulse-width", type=float, default=1.15, help="Rectangular laser pulse width in gate-index units.")
    parser.add_argument("--gate-width", type=float, default=1.35, help="Rectangular receiver gate width in gate-index units.")
    parser.add_argument("--gate-spacing", type=float, default=1.0, help="Spacing between adjacent gate centers.")
    parser.add_argument("--min-response", type=float, default=0.08)
    parser.add_argument("--reflectance-min", type=float, default=0.85)
    parser.add_argument("--reflectance-max", type=float, default=1.15)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--manifest-out", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def stable_seed(seed: int, class_name: str, sample_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{class_name}:{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**32)


def prepare_output_root(output_root: Path, overwrite: bool) -> None:
    resolved = output_root.resolve()
    if overwrite and resolved.exists():
        if len(resolved.parts) < 4:
            raise ValueError(f"Refusing to remove a shallow output path: {resolved}")
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
    arrays = [np.asarray(Image.open(paths[gate]).convert("L"), dtype=np.float32) for gate in range(expected_num_slices)]
    return np.stack(arrays, axis=0)


def make_projection(stack: np.ndarray, mode: str) -> np.ndarray:
    if mode == "max":
        return stack.max(axis=0)
    return stack.mean(axis=0)


def gaussian_gate_response(num_gates: int, center: float, sigma: float, min_response: float) -> np.ndarray:
    gates = np.arange(num_gates, dtype=np.float32)
    response = np.exp(-0.5 * ((gates - center) / sigma) ** 2)
    response = response / max(float(response.max()), 1e-6)
    return min_response + (1.0 - min_response) * response


def interval_overlap(left_center: float, left_width: float, right_center: float, right_width: float) -> float:
    left_start = left_center - left_width / 2.0
    left_end = left_center + left_width / 2.0
    right_start = right_center - right_width / 2.0
    right_end = right_center + right_width / 2.0
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def rectangular_overlap_gate_response(
    num_gates: int,
    center: float,
    pulse_width: float,
    gate_width: float,
    gate_spacing: float,
    min_response: float,
) -> np.ndarray:
    echo_center = center * gate_spacing
    max_overlap = min(pulse_width, gate_width)
    values = []
    for gate in range(num_gates):
        gate_center = gate * gate_spacing
        overlap = interval_overlap(echo_center, pulse_width, gate_center, gate_width)
        values.append(overlap / max(max_overlap, 1e-6))
    response = np.asarray(values, dtype=np.float32)
    return min_response + (1.0 - min_response) * response


def gate_response(num_gates: int, center: float, args: argparse.Namespace) -> np.ndarray:
    if args.response_mode == "gaussian":
        return gaussian_gate_response(num_gates, center, args.response_sigma, args.min_response)
    return rectangular_overlap_gate_response(
        num_gates,
        center,
        args.pulse_width,
        args.gate_width,
        args.gate_spacing,
        args.min_response,
    )


def write_manifest(path: Path, rows: list[HardFlatSampleRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(HardFlatSampleRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def build_sample(
    class_name: str,
    sample_id: str,
    paths: dict[int, Path],
    output_dir: Path,
    args: argparse.Namespace,
) -> HardFlatSampleRow:
    rng = np.random.default_rng(stable_seed(args.seed, class_name, sample_id))
    stack = load_stack(paths, args.expected_num_slices)
    projection = make_projection(stack, args.projection_mode)
    center = float(rng.uniform(0.0, args.expected_num_slices - 1))
    reflectance = float(rng.uniform(args.reflectance_min, args.reflectance_max))
    responses = gate_response(args.expected_num_slices, center, args)

    output_dir.mkdir(parents=True, exist_ok=True)
    for gate, response in enumerate(responses):
        out_arr = np.clip(np.rint(projection * float(response) * reflectance), 0, 255).astype(np.uint8)
        Image.fromarray(out_arr, mode="L").save(output_dir / f"{sample_id}_gate_{gate}.png")

    metadata = {
        "source_class": class_name,
        "source_sample_id": sample_id,
        "projection_mode": args.projection_mode,
        "response_mode": args.response_mode,
        "flat_center_gate": center,
        "response_sigma": args.response_sigma,
        "pulse_width": args.pulse_width,
        "gate_width": args.gate_width,
        "gate_spacing": args.gate_spacing,
        "min_response": args.min_response,
        "reflectance": reflectance,
        "gate_responses": [float(value) for value in responses],
    }
    with (output_dir / f"{sample_id}_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return HardFlatSampleRow(
        source_class=class_name,
        source_sample_id=sample_id,
        output_sample_id=sample_id,
        projection_mode=args.projection_mode,
        response_mode=args.response_mode,
        flat_center_gate=center,
        response_sigma=args.response_sigma,
        pulse_width=args.pulse_width,
        gate_width=args.gate_width,
        gate_spacing=args.gate_spacing,
        min_response=args.min_response,
        reflectance=reflectance,
        gate_responses=";".join(f"g{gate}:{float(value):.6f}" for gate, value in enumerate(responses)),
    )


def main() -> int:
    args = parse_args()
    if args.expected_num_slices <= 0:
        raise ValueError("--expected-num-slices must be positive")
    if args.response_sigma <= 0.0:
        raise ValueError("--response-sigma must be positive")
    if args.pulse_width <= 0.0:
        raise ValueError("--pulse-width must be positive")
    if args.gate_width <= 0.0:
        raise ValueError("--gate-width must be positive")
    if args.gate_spacing <= 0.0:
        raise ValueError("--gate-spacing must be positive")
    if not 0.0 <= args.min_response <= 1.0:
        raise ValueError("--min-response must be in [0, 1]")
    if args.reflectance_min <= 0.0 or args.reflectance_max < args.reflectance_min:
        raise ValueError("Reflectance range must be positive and ordered")
    if not args.true_root.exists():
        raise FileNotFoundError(f"True root does not exist: {args.true_root}")

    prepare_output_root(args.output_root, args.overwrite)
    rows: list[HardFlatSampleRow] = []
    for class_dir in sorted(path for path in args.true_root.iterdir() if path.is_dir() and not path.name.startswith("_")):
        grouped = group_gate_images(class_dir)
        out_class_dir = args.output_root / class_dir.name
        for sample_id, paths in sorted(grouped.items()):
            if len(paths) != args.expected_num_slices:
                continue
            rows.append(build_sample(class_dir.name, sample_id, paths, out_class_dir, args))

    if not rows:
        raise RuntimeError(f"No complete {args.expected_num_slices}-gate samples found under {args.true_root}")

    manifest_out = args.manifest_out or args.output_root / "hard_flat_manifest.csv"
    write_manifest(manifest_out, rows)
    print(f"samples={len(rows)}")
    print(f"output_root={args.output_root}")
    print(f"manifest={manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

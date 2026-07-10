from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


GATE_PATTERN = re.compile(r"^(?P<base>.+?)_gate_(?P<gate>\d+)\.png$")


@dataclass(frozen=True)
class SampleGateDiagnostic:
    class_name: str
    sample_id: str
    num_gates: int
    gate_max: str
    gate_mean: str
    gate_active_fraction: str
    mean_pair_corr_raw: float
    mean_pair_corr_maxnorm: float
    mean_pair_mask_iou: float
    mean_pair_absdiff_maxnorm: float
    active_fraction_std: float
    max_mean_ratio: float


@dataclass(frozen=True)
class ClassGateDiagnostic:
    class_name: str
    num_samples: int
    mean_pair_corr_raw: float
    mean_pair_corr_maxnorm: float
    mean_pair_mask_iou: float
    mean_pair_absdiff_maxnorm: float
    mean_active_fraction_std: float
    mean_max_mean_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether gated image stacks contain depth-varying structure. "
            "Flat false targets should have higher inter-gate structural similarity than true 3D targets."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sample-csv-out", type=Path, required=True)
    parser.add_argument("--class-csv-out", type=Path, required=True)
    parser.add_argument("--active-threshold", type=int, default=5)
    parser.add_argument("--expected-gates", type=int, default=3)
    return parser.parse_args()


def iter_grouped_gate_images(root: Path) -> dict[tuple[str, str], dict[int, Path]]:
    grouped: dict[tuple[str, str], dict[int, Path]] = {}
    for class_dir in sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith("_")):
        for path in sorted(class_dir.rglob("*.png")):
            match = GATE_PATTERN.match(path.name)
            if not match:
                continue
            key = (class_dir.name, match.group("base"))
            grouped.setdefault(key, {})[int(match.group("gate"))] = path
    return grouped


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    af = a.reshape(-1).astype(np.float64)
    bf = b.reshape(-1).astype(np.float64)
    if float(af.std()) == 0.0 or float(bf.std()) == 0.0:
        return 0.0
    return float(np.corrcoef(af, bf)[0, 1])


def max_normalize(arr: np.ndarray) -> np.ndarray:
    max_value = float(arr.max())
    if max_value <= 0.0:
        return arr.astype(np.float32)
    return arr.astype(np.float32) / max_value


def mask_iou(a: np.ndarray, b: np.ndarray, threshold: int) -> float:
    am = a > threshold
    bm = b > threshold
    union = int(np.logical_or(am, bm).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(am, bm).sum() / union)


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def analyze_sample(
    class_name: str,
    sample_id: str,
    gate_paths: dict[int, Path],
    active_threshold: int,
) -> SampleGateDiagnostic:
    arrays = {gate: np.asarray(Image.open(path).convert("L"), dtype=np.float32) for gate, path in sorted(gate_paths.items())}
    gates = sorted(arrays)
    max_values = [float(arrays[gate].max()) for gate in gates]
    mean_values = [float(arrays[gate].mean()) for gate in gates]
    active_values = [float((arrays[gate] > active_threshold).mean()) for gate in gates]

    raw_corrs: list[float] = []
    norm_corrs: list[float] = []
    ious: list[float] = []
    absdiffs: list[float] = []
    for left_idx, left_gate in enumerate(gates):
        for right_gate in gates[left_idx + 1 :]:
            left = arrays[left_gate]
            right = arrays[right_gate]
            left_norm = max_normalize(left)
            right_norm = max_normalize(right)
            raw_corrs.append(safe_corr(left, right))
            norm_corrs.append(safe_corr(left_norm, right_norm))
            ious.append(mask_iou(left, right, active_threshold))
            absdiffs.append(float(np.abs(left_norm - right_norm).mean()))

    nonzero_means = [value for value in mean_values if value > 0.0]
    max_mean_ratio = (max(nonzero_means) / min(nonzero_means)) if nonzero_means else 0.0

    return SampleGateDiagnostic(
        class_name=class_name,
        sample_id=sample_id,
        num_gates=len(gates),
        gate_max=";".join(f"g{gate}:{max_values[idx]:.3f}" for idx, gate in enumerate(gates)),
        gate_mean=";".join(f"g{gate}:{mean_values[idx]:.6f}" for idx, gate in enumerate(gates)),
        gate_active_fraction=";".join(f"g{gate}:{active_values[idx]:.6f}" for idx, gate in enumerate(gates)),
        mean_pair_corr_raw=mean(raw_corrs),
        mean_pair_corr_maxnorm=mean(norm_corrs),
        mean_pair_mask_iou=mean(ious),
        mean_pair_absdiff_maxnorm=mean(absdiffs),
        active_fraction_std=float(np.std(np.asarray(active_values, dtype=np.float32))),
        max_mean_ratio=float(max_mean_ratio),
    )


def summarize_by_class(rows: list[SampleGateDiagnostic]) -> list[ClassGateDiagnostic]:
    out: list[ClassGateDiagnostic] = []
    classes = sorted({row.class_name for row in rows})
    for class_name in classes:
        class_rows = [row for row in rows if row.class_name == class_name]
        out.append(
            ClassGateDiagnostic(
                class_name=class_name,
                num_samples=len(class_rows),
                mean_pair_corr_raw=mean([row.mean_pair_corr_raw for row in class_rows]),
                mean_pair_corr_maxnorm=mean([row.mean_pair_corr_maxnorm for row in class_rows]),
                mean_pair_mask_iou=mean([row.mean_pair_mask_iou for row in class_rows]),
                mean_pair_absdiff_maxnorm=mean([row.mean_pair_absdiff_maxnorm for row in class_rows]),
                mean_active_fraction_std=mean([row.active_fraction_std for row in class_rows]),
                mean_max_mean_ratio=mean([row.max_mean_ratio for row in class_rows]),
            )
        )
    return out


def write_csv(path: Path, rows: list[object]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def main() -> int:
    args = parse_args()
    if not args.root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {args.root}")

    grouped = iter_grouped_gate_images(args.root)
    rows = [
        analyze_sample(class_name, sample_id, paths, args.active_threshold)
        for (class_name, sample_id), paths in grouped.items()
        if len(paths) == args.expected_gates
    ]
    if not rows:
        raise RuntimeError(f"No complete {args.expected_gates}-gate samples found under {args.root}")

    class_rows = summarize_by_class(rows)
    write_csv(args.sample_csv_out, rows)
    write_csv(args.class_csv_out, class_rows)
    print(f"samples={len(rows)}")
    print(f"classes={len(class_rows)}")
    print(f"sample_csv={args.sample_csv_out}")
    print(f"class_csv={args.class_csv_out}")
    for row in class_rows:
        print(
            f"{row.class_name}: samples={row.num_samples} "
            f"corr_norm={row.mean_pair_corr_maxnorm:.4f} "
            f"mask_iou={row.mean_pair_mask_iou:.4f} "
            f"absdiff_norm={row.mean_pair_absdiff_maxnorm:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

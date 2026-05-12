from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image


GATE_PATTERN = re.compile(r"_gate_(\d+)\.png$")


def analyze(dataset_root: Path, black_threshold: float) -> list[dict[str, float | int]]:
    means: dict[int, list[float]] = {}
    active_fracs: dict[int, list[float]] = {}

    for path in dataset_root.rglob("*.png"):
        match = GATE_PATTERN.search(path.name)
        if not match:
            continue
        gate = int(match.group(1))
        arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        means.setdefault(gate, []).append(float(arr.mean()))
        active_fracs.setdefault(gate, []).append(float((arr > black_threshold).mean()))

    rows = []
    for gate in sorted(means):
        gate_means = np.asarray(means[gate], dtype=np.float32)
        gate_active = np.asarray(active_fracs[gate], dtype=np.float32)
        rows.append(
            {
                "gate": gate,
                "count": int(gate_means.size),
                "mean_intensity": float(gate_means.mean()),
                "mean_active_fraction": float(gate_active.mean()),
                "blank_images": int((gate_active < 0.001).sum()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize black/active area statistics for gated PNG slices.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--black-threshold", type=float, default=0.01)
    args = parser.parse_args()

    for row in analyze(args.dataset_root, args.black_threshold):
        print(
            f"gate_{row['gate']}: count={row['count']} "
            f"mean_intensity={row['mean_intensity']:.6f} "
            f"active_fraction={row['mean_active_fraction']:.6f} "
            f"blank_images={row['blank_images']}"
        )


if __name__ == "__main__":
    main()

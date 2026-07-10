from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from dataset_new import build_hard_nuisance_dataset as builder


def write_stack(root: Path, class_name: str, sample_id: str) -> None:
    class_dir = root / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[:16, :16]
    base = ((xx + yy) % 16).astype(np.float32) + 20
    for gate in range(3):
        arr = np.clip(base + gate * 15, 0, 255).astype(np.uint8)
        Image.fromarray(arr, mode="L").save(class_dir / f"{sample_id}_gate_{gate}.png")


def make_args(input_root: Path, output_root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        input_root=input_root,
        output_root=output_root,
        classes=["true3d", "flat_false"],
        expected_num_slices=3,
        seed=123,
        foreground_threshold=4,
        reflectance_strength=0.08,
        background_strength=3.0,
        background_gate_jitter=0.18,
        occlusion_probability=1.0,
        occlusion_min_fraction=0.04,
        occlusion_max_fraction=0.08,
        occlusion_alpha=0.6,
        preserve_input_max=True,
        manifest_out=output_root / "manifest.csv",
        overwrite=False,
    )


def test_hard_nuisance_builder_writes_paired_complete_stacks(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    sample_id = "view_z000__tank__model001"
    write_stack(input_root, "true3d", sample_id)
    write_stack(input_root, "flat_false", sample_id)

    args = make_args(input_root, output_root)
    builder.prepare_output_root(output_root, overwrite=False)
    rows = builder.build_dataset(args)
    builder.write_manifest(args.manifest_out, rows)

    assert len(rows) == 2
    assert {row.class_name for row in rows} == {"true3d", "flat_false"}
    assert len({row.nuisance_key for row in rows}) == 1
    assert len(list((output_root / "true3d").glob("*_gate_*.png"))) == 3
    assert len(list((output_root / "flat_false").glob("*_gate_*.png"))) == 3

    with args.manifest_out.open(newline="", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))
    assert len(manifest_rows) == 2
    assert manifest_rows[0]["nuisance_key"] == "tank__model001"

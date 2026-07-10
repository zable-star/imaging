from __future__ import annotations

import numpy as np
from PIL import Image

from dataset_new.match_false_target_exposure import (
    copy_true_image,
    scale_false_image,
    target_stats,
)


def save_gate(path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8), value, dtype=np.uint8), mode="L").save(path)


def read_mean(path) -> float:
    return float(np.asarray(Image.open(path).convert("L"), dtype=np.float32).mean() / 255.0)


def test_false_exposure_matching_scales_to_true_gate_mean(tmp_path) -> None:
    true_dir = tmp_path / "input" / "true3d"
    false_dir = tmp_path / "input" / "flat_false"
    output_dir = tmp_path / "matched"

    save_gate(true_dir / "sample_gate_0.png", 100)
    save_gate(false_dir / "sample_gate_0.png", 50)

    targets = target_stats(true_dir, stat="mean_all", foreground_threshold=8.0 / 255.0)
    row = scale_false_image(
        false_dir / "sample_gate_0.png",
        output_dir / "flat_false" / "sample_gate_0.png",
        target_stat=targets[0],
        match_stat="mean_all",
        match_scope="per-image",
        foreground_threshold=8.0 / 255.0,
        max_scale=8.0,
    )

    assert row.copied is False
    assert row.scale == 2.0
    assert abs(read_mean(output_dir / "flat_false" / "sample_gate_0.png") - targets[0]) < 1e-6


def test_false_exposure_matching_can_use_max_value_stat(tmp_path) -> None:
    true_dir = tmp_path / "input" / "true3d"
    false_dir = tmp_path / "input" / "flat_false"
    output_dir = tmp_path / "matched"

    save_gate(true_dir / "sample_gate_0.png", 100)
    save_gate(false_dir / "sample_gate_0.png", 200)

    targets = target_stats(true_dir, stat="max_value", foreground_threshold=8.0 / 255.0)
    row = scale_false_image(
        false_dir / "sample_gate_0.png",
        output_dir / "flat_false" / "sample_gate_0.png",
        target_stat=targets[0],
        match_stat="max_value",
        match_scope="per-image",
        foreground_threshold=8.0 / 255.0,
        max_scale=8.0,
    )

    assert row.scale == 0.5
    assert abs(read_mean(output_dir / "flat_false" / "sample_gate_0.png") - targets[0]) < 1e-6


def test_class_gate_fixed_scale_can_be_used_for_false_images(tmp_path) -> None:
    false_dir = tmp_path / "input" / "flat_false"
    output_dir = tmp_path / "matched"
    save_gate(false_dir / "sample_gate_0.png", 80)

    row = scale_false_image(
        false_dir / "sample_gate_0.png",
        output_dir / "flat_false" / "sample_gate_0.png",
        target_stat=0.0,
        match_stat="mean_all",
        match_scope="class-gate",
        foreground_threshold=8.0 / 255.0,
        max_scale=8.0,
        fixed_scale=0.5,
    )

    assert row.scale == 0.5
    assert abs(read_mean(output_dir / "flat_false" / "sample_gate_0.png") - (40.0 / 255.0)) < 1e-6


def test_true_images_are_copied_without_scaling(tmp_path) -> None:
    source = tmp_path / "input" / "true3d" / "sample_gate_0.png"
    output = tmp_path / "matched" / "true3d" / "sample_gate_0.png"
    save_gate(source, 77)

    row = copy_true_image(source, output, match_stat="mean_all", foreground_threshold=8.0 / 255.0)

    assert row.copied is True
    assert row.scale == 1.0
    assert read_mean(output) == read_mean(source)

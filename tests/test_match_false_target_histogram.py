from __future__ import annotations

import numpy as np
from PIL import Image

from dataset_new.match_false_target_histogram import (
    copy_true_image,
    foreground_values,
    match_false_image,
    quantile_match_values,
)


def save_gate(path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(values, dtype=np.uint8), mode="L").save(path)


def read_gray(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def test_quantile_match_values_maps_source_range_to_target_range() -> None:
    source = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    target = np.asarray([0.5, 0.6, 0.7], dtype=np.float32)

    matched = quantile_match_values(source, target, num_quantiles=8)

    assert np.allclose(matched, target, atol=1e-6)


def test_quantile_match_values_handles_constant_source() -> None:
    source = np.asarray([0.2, 0.2, 0.2], dtype=np.float32)
    target = np.asarray([0.4, 0.6, 0.8], dtype=np.float32)

    matched = quantile_match_values(source, target, num_quantiles=8)

    assert np.allclose(matched, np.full_like(source, 0.6), atol=1e-6)


def test_match_false_image_preserves_background_and_matches_foreground_mean(tmp_path) -> None:
    source = tmp_path / "input" / "flat_false" / "sample_gate_0.png"
    output = tmp_path / "output" / "flat_false" / "sample_gate_0.png"
    save_gate(source, [[0, 20], [40, 60]])
    target_values = np.asarray([120 / 255.0, 180 / 255.0, 240 / 255.0], dtype=np.float32)

    row = match_false_image(
        source,
        output,
        target_values=target_values,
        target_path=None,
        match_scope="class-gate",
        foreground_threshold=8.0 / 255.0,
        num_quantiles=8,
    )

    out = read_gray(output)
    out_foreground = foreground_values(out, 8.0 / 255.0)
    assert out[0, 0] == 0.0
    assert row.copied is False
    assert abs(float(out_foreground.mean()) - float(target_values.mean())) < 1 / 255.0


def test_copy_true_image_copies_without_histogram_matching(tmp_path) -> None:
    source = tmp_path / "input" / "true3d" / "sample_gate_0.png"
    output = tmp_path / "output" / "true3d" / "sample_gate_0.png"
    save_gate(source, [[0, 100], [150, 200]])

    row = copy_true_image(source, output, foreground_threshold=8.0 / 255.0)

    assert row.copied is True
    assert np.array_equal(read_gray(source), read_gray(output))

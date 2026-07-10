from __future__ import annotations

import numpy as np
from PIL import Image

from dataset_new.clip_false_target_intensity import clip_false_image, copy_true_image, target_clip_values


def save_gate(path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(values, dtype=np.uint8), mode="L").save(path)


def read_gray(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def test_target_clip_values_uses_true_gate_mean_max(tmp_path) -> None:
    true_dir = tmp_path / "input" / "true3d"
    save_gate(true_dir / "a_gate_0.png", [[0, 100]])
    save_gate(true_dir / "b_gate_0.png", [[0, 200]])

    targets = target_clip_values(true_dir, "max_value")

    assert abs(targets[0] - (150.0 / 255.0)) < 1e-6


def test_clip_false_image_clips_only_high_pixels(tmp_path) -> None:
    source = tmp_path / "input" / "flat_false" / "sample_gate_0.png"
    output = tmp_path / "output" / "flat_false" / "sample_gate_0.png"
    save_gate(source, [[0, 50, 250]])

    row = clip_false_image(
        source,
        output,
        clip_value=100.0 / 255.0,
        clip_stat="max_value",
        clip_scope="class-gate-mean",
    )

    out = read_gray(output)
    assert row.copied is False
    assert out[0, 0] == 0.0
    assert abs(out[0, 1] - (50.0 / 255.0)) < 1e-6
    assert abs(out[0, 2] - (100.0 / 255.0)) < 1e-6
    assert row.clipped_pixel_fraction == 1 / 3


def test_copy_true_image_preserves_image(tmp_path) -> None:
    source = tmp_path / "input" / "true3d" / "sample_gate_0.png"
    output = tmp_path / "output" / "true3d" / "sample_gate_0.png"
    save_gate(source, [[0, 100], [150, 200]])

    row = copy_true_image(source, output, clip_stat="max_value")

    assert row.copied is True
    assert np.array_equal(read_gray(source), read_gray(output))

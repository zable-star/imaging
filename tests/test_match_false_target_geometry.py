from __future__ import annotations

import numpy as np
from PIL import Image

from dataset_new.match_false_target_geometry import (
    area_matched_mask,
    chessboard_distance,
    copy_true_image,
    match_false_image,
)


def save_gate(path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(values, dtype=np.uint8), mode="L").save(path)


def read_gray(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def test_chessboard_distance_prefers_object_center() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True

    distance = chessboard_distance(mask)

    assert distance[2, 2] == 2
    assert distance[1, 1] == 1
    assert distance[0, 0] == 0


def test_area_matched_mask_keeps_requested_number_of_pixels() -> None:
    arr = np.zeros((5, 5), dtype=np.float32)
    arr[1:4, 1:4] = 0.5

    mask = area_matched_mask(arr, foreground_threshold=8.0 / 255.0, target_pixels=5)

    assert int(mask.sum()) == 5
    assert mask[2, 2]


def test_match_false_image_reduces_foreground_area(tmp_path) -> None:
    source = tmp_path / "input" / "flat_false" / "sample_gate_0.png"
    output = tmp_path / "output" / "flat_false" / "sample_gate_0.png"
    values = np.zeros((6, 6), dtype=np.uint8)
    values[1:5, 1:5] = 100
    save_gate(source, values)

    row = match_false_image(
        source,
        output,
        target_pixels=6,
        target_path=None,
        match_scope="class-gate-mean",
        foreground_threshold=8.0 / 255.0,
        allow_grow=False,
    )

    out = read_gray(output)
    assert row.copied is False
    assert row.source_foreground_pixels == 16
    assert row.output_foreground_pixels == 6
    assert int((out > 8.0 / 255.0).sum()) == 6


def test_copy_true_image_preserves_image(tmp_path) -> None:
    source = tmp_path / "input" / "true3d" / "sample_gate_0.png"
    output = tmp_path / "output" / "true3d" / "sample_gate_0.png"
    save_gate(source, [[0, 30], [60, 90]])

    row = copy_true_image(source, output, foreground_threshold=8.0 / 255.0)

    assert row.copied is True
    assert np.array_equal(read_gray(source), read_gray(output))

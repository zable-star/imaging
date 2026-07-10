from __future__ import annotations

import numpy as np
from PIL import Image

from dataset_new.diagnose_single_gate_features import (
    analyze_image,
    separability,
    threshold_accuracy,
)


def save_image(path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), mode="L").save(path)


def test_analyze_image_reports_foreground_and_bbox_features(tmp_path) -> None:
    arr = np.zeros((10, 10), dtype=np.uint8)
    arr[2:6, 3:8] = 120
    path = tmp_path / "true3d" / "sample_gate_1.png"
    save_image(path, arr)

    row = analyze_image("true3d", path, foreground_threshold=8.0 / 255.0, edge_threshold=0.04)

    assert row.sample_id == "sample"
    assert row.gate == 1
    assert row.foreground_ratio == 0.2
    assert row.bbox_area_ratio == 0.2
    assert row.bbox_aspect == 1.25
    assert row.edge_density > 0.0


def test_threshold_accuracy_finds_best_direction() -> None:
    values0 = np.asarray([0.1, 0.2, 0.3])
    values1 = np.asarray([0.7, 0.8, 0.9])

    threshold, accuracy, direction = threshold_accuracy(values0, values1)

    assert 0.3 < threshold < 0.7
    assert accuracy == 1.0
    assert direction == "class1_if_ge"


def test_separability_detects_simple_feature_split(tmp_path) -> None:
    rows = []
    for idx, value in enumerate([30, 35, 40]):
        path = tmp_path / "true3d" / f"true_{idx}_gate_0.png"
        save_image(path, np.full((8, 8), value, dtype=np.uint8))
        rows.append(analyze_image("true3d", path, foreground_threshold=8.0 / 255.0, edge_threshold=0.04))
    for idx, value in enumerate([150, 160, 170]):
        path = tmp_path / "flat_false" / f"false_{idx}_gate_0.png"
        save_image(path, np.full((8, 8), value, dtype=np.uint8))
        rows.append(analyze_image("flat_false", path, foreground_threshold=8.0 / 255.0, edge_threshold=0.04))

    sep = separability(rows, ["true3d", "flat_false"])
    mean_all = next(row for row in sep if row.feature == "mean_all")

    assert mean_all.best_accuracy == 1.0
    assert mean_all.cohen_d > 5.0

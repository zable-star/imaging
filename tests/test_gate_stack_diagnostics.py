from __future__ import annotations

import numpy as np
from PIL import Image

from dataset_new.diagnose_gate_stack import (
    analyze_sample,
    iter_grouped_gate_images,
    summarize_by_class,
)


def save_gate(path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), mode="L").save(path)


def test_gate_stack_diagnostics_separate_flat_and_depth_varying_samples(tmp_path) -> None:
    flat_dir = tmp_path / "flat_false"
    true_dir = tmp_path / "true3d"

    base_mask = np.zeros((16, 16), dtype=np.uint8)
    base_mask[4:12, 4:12] = 1
    for gate, scale in enumerate([40, 90, 140]):
        save_gate(flat_dir / f"flat_sample_gate_{gate}.png", base_mask * scale)

    true_masks = []
    for start in [2, 5, 8]:
        arr = np.zeros((16, 16), dtype=np.uint8)
        arr[start : start + 5, start : start + 5] = 120
        true_masks.append(arr)
    for gate, arr in enumerate(true_masks):
        save_gate(true_dir / f"true_sample_gate_{gate}.png", arr)

    grouped = iter_grouped_gate_images(tmp_path)
    rows = [
        analyze_sample(class_name, sample_id, paths, active_threshold=5)
        for (class_name, sample_id), paths in grouped.items()
    ]
    summary = {row.class_name: row for row in summarize_by_class(rows)}

    assert summary["flat_false"].mean_pair_corr_maxnorm > 0.99
    assert summary["flat_false"].mean_pair_mask_iou == 1.0
    assert summary["true3d"].mean_pair_corr_maxnorm < 0.5
    assert summary["true3d"].mean_pair_mask_iou < 0.5

from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("torch")

from dataset import MultiSliceObjectDataset


def test_dataset_groups_gate_images_recursively(tmp_path: Path) -> None:
    class_dir = tmp_path / "chair"
    nested = class_dir / "train"
    nested.mkdir(parents=True)
    for gate in range(3):
        Image.new("L", (4, 4), color=gate * 20).save(nested / f"train_chair_0001_gate_{gate}.png")

    dataset = MultiSliceObjectDataset({"chair": class_dir}, expected_num_slices=3)

    assert len(dataset) == 1
    x, label, meta = dataset[0]
    assert x.shape == (3, 1, 4, 4)
    assert label == 0
    assert meta["sample_id"] == "train_chair_0001"

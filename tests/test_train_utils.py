from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from train import (
    SliceDegradationDataset,
    SliceInputViewDataset,
    ema,
    split_group_id,
    stratified_group_split,
    stratified_split,
)


class DummySample:
    def __init__(self, label: int, sample_id=None) -> None:
        self.label = label
        self.sample_id = sample_id if sample_id is not None else f"sample_{label}"


class DummyDataset:
    def __init__(self, labels: list[int], sample_ids: list[str] | None = None) -> None:
        if sample_ids is None:
            sample_ids = [f"sample_{idx}" for idx in range(len(labels))]
        self.samples = [DummySample(label, sample_id) for label, sample_id in zip(labels, sample_ids)]


class DummySliceDataset:
    def __init__(self) -> None:
        self.samples = [DummySample(0)]
        self.class_names = ["chair"]
        self.num_slices = 3

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        x = torch.stack(
            [
                torch.full((1, 2, 2), 0.25),
                torch.full((1, 2, 2), 0.50),
                torch.full((1, 2, 2), 0.75),
            ]
        )
        return x, 0, {"sample_id": "sample_0", "class_name": "chair"}


def test_ema_smooths_values_with_alpha() -> None:
    assert ema([1.0, 3.0, 5.0], alpha=0.5) == [1.0, 2.0, 3.5]


def test_ema_handles_empty_values() -> None:
    assert ema([], alpha=0.5) == []


def test_ema_clamps_alpha() -> None:
    assert ema([1.0, 3.0], alpha=2.0) == [1.0, 3.0]
    assert ema([1.0, 3.0], alpha=-1.0) == [1.0, 1.0]


def test_stratified_split_keeps_class_counts_balanced() -> None:
    dataset = DummyDataset(labels=[0] * 10 + [1] * 10 + [2] * 10)

    train_set, val_set = stratified_split(dataset, val_ratio=0.2, seed=42)

    val_labels = [dataset.samples[index].label for index in val_set.indices]
    train_labels = [dataset.samples[index].label for index in train_set.indices]

    assert len(val_set) == 6
    assert len(train_set) == 24
    assert {label: val_labels.count(label) for label in range(3)} == {0: 2, 1: 2, 2: 2}
    assert {label: train_labels.count(label) for label in range(3)} == {0: 8, 1: 8, 2: 8}


def test_stratified_split_rejects_invalid_ratio() -> None:
    dataset = DummyDataset(labels=[0, 0, 1, 1])

    with pytest.raises(ValueError, match="val_ratio"):
        stratified_split(dataset, val_ratio=0.0, seed=42)


def test_stratified_group_split_keeps_sample_ids_together() -> None:
    labels = []
    sample_ids = []
    for idx in range(10):
        labels.extend([0, 1])
        sample_ids.extend([f"source_{idx}", f"source_{idx}"])
    dataset = DummyDataset(labels=labels, sample_ids=sample_ids)

    train_set, val_set = stratified_group_split(dataset, val_ratio=0.3, seed=42)

    train_ids = {dataset.samples[index].sample_id for index in train_set.indices}
    val_ids = {dataset.samples[index].sample_id for index in val_set.indices}
    assert train_ids.isdisjoint(val_ids)
    assert len(val_set) == 6


def test_split_group_id_removes_multiview_prefix() -> None:
    assert split_group_id("view_z000__source_001") == "source_001"
    assert split_group_id("view_z090__01_Main_Battle_Tank__source_001") == "01_Main_Battle_Tank__source_001"
    assert split_group_id("source_001") == "source_001"


def test_split_group_id_removes_domain_and_multiview_prefixes() -> None:
    assert split_group_id("domain_norm__view_z000__source_001") == "source_001"
    assert split_group_id("domain_hardv3__view_z090__01_Main_Battle_Tank__source_001") == (
        "01_Main_Battle_Tank__source_001"
    )


def test_stratified_group_split_keeps_multiview_source_ids_together() -> None:
    labels = []
    sample_ids = []
    for idx in range(6):
        for view in ("view_z000", "view_z090"):
            labels.extend([0, 1])
            sample_ids.extend([f"{view}__source_{idx}", f"{view}__source_{idx}"])
    dataset = DummyDataset(labels=labels, sample_ids=sample_ids)

    train_set, val_set = stratified_group_split(dataset, val_ratio=0.33, seed=42)

    train_groups = {split_group_id(dataset.samples[index].sample_id) for index in train_set.indices}
    val_groups = {split_group_id(dataset.samples[index].sample_id) for index in val_set.indices}
    assert train_groups.isdisjoint(val_groups)


def test_stratified_group_split_keeps_domain_variants_together() -> None:
    labels = []
    sample_ids = []
    for idx in range(6):
        for domain in ("domain_norm", "domain_hardv3"):
            for view in ("view_z000", "view_z090"):
                labels.extend([0, 1])
                sample_ids.extend([f"{domain}__{view}__source_{idx}", f"{domain}__{view}__source_{idx}"])
    dataset = DummyDataset(labels=labels, sample_ids=sample_ids)

    train_set, val_set = stratified_group_split(dataset, val_ratio=0.33, seed=42)

    train_groups = {split_group_id(dataset.samples[index].sample_id) for index in train_set.indices}
    val_groups = {split_group_id(dataset.samples[index].sample_id) for index in val_set.indices}
    assert train_groups.isdisjoint(val_groups)


def test_slice_input_view_dataset_can_return_single_2d_gate() -> None:
    dataset = SliceInputViewDataset(DummySliceDataset(), input_mode="single-gate", single_gate_index=1)

    x, label, meta = dataset[0]

    assert x.shape == (1, 1, 2, 2)
    assert torch.all(x == 0.50)
    assert label == 0
    assert meta["input_mode"] == "single-gate"
    assert meta["single_gate_index"] == 1


def test_slice_input_view_dataset_can_black_out_other_gates() -> None:
    dataset = SliceInputViewDataset(DummySliceDataset(), input_mode="single-gate-black", single_gate_index=2)

    x, _, meta = dataset[0]

    assert x.shape == (3, 1, 2, 2)
    assert torch.all(x[0] == 0.0)
    assert torch.all(x[1] == 0.0)
    assert torch.all(x[2] == 0.75)
    assert meta["input_mode"] == "single-gate-black"


def test_slice_input_view_dataset_rejects_invalid_gate_index() -> None:
    with pytest.raises(ValueError, match="single_gate_index"):
        SliceInputViewDataset(DummySliceDataset(), input_mode="single-gate", single_gate_index=3)


def test_slice_degradation_dataset_applies_fixed_gate_dropout() -> None:
    base = SliceInputViewDataset(DummySliceDataset(), input_mode="multi", single_gate_index=0)
    dataset = SliceDegradationDataset(base, seed=5, gate_dropout_mode="fixed", gate_dropout_index=1)

    x, _, meta = dataset[0]

    assert torch.all(x[0] == 0.25)
    assert torch.all(x[1] == 0.0)
    assert torch.all(x[2] == 0.75)
    assert meta["degradation_enabled"] is True
    assert meta["resolved_gate_dropout_index"] == 1


def test_slice_degradation_dataset_records_clean_condition() -> None:
    base = SliceInputViewDataset(DummySliceDataset(), input_mode="multi", single_gate_index=0)
    dataset = SliceDegradationDataset(base, seed=5)

    x, _, meta = dataset[0]

    assert torch.all(x[0] == 0.25)
    assert torch.all(x[1] == 0.50)
    assert torch.all(x[2] == 0.75)
    assert meta["degradation_enabled"] is False
    assert meta["resolved_gate_dropout_index"] == -1


def test_slice_degradation_probability_can_leave_configured_sample_clean() -> None:
    base = SliceInputViewDataset(DummySliceDataset(), input_mode="multi", single_gate_index=0)
    dataset = SliceDegradationDataset(
        base,
        seed=5,
        gate_dropout_mode="fixed",
        gate_dropout_index=1,
        degradation_probability=0.0,
    )

    x, _, meta = dataset[0]

    assert torch.all(x[0] == 0.25)
    assert torch.all(x[1] == 0.50)
    assert torch.all(x[2] == 0.75)
    assert meta["degradation_configured"] is True
    assert meta["degradation_enabled"] is False
    assert meta["degradation_probability"] == 0.0
    assert meta["resolved_gate_dropout_index"] == -1


def test_slice_degradation_dataset_is_deterministic_for_noise() -> None:
    base = SliceInputViewDataset(DummySliceDataset(), input_mode="multi", single_gate_index=0)
    dataset = SliceDegradationDataset(base, seed=5, gaussian_noise_std=0.1, background_scatter=0.05)

    x1, _, _ = dataset[0]
    x2, _, _ = dataset[0]

    torch.testing.assert_close(x1, x2)


def test_slice_degradation_dataset_applies_structured_nuisance_deterministically() -> None:
    base = SliceInputViewDataset(DummySliceDataset(), input_mode="multi", single_gate_index=0)
    dataset = SliceDegradationDataset(
        base,
        seed=5,
        structured_reflectance_strength=0.1,
        structured_background_strength=0.02,
        structured_nuisance_grid_size=3,
        occlusion_probability=1.0,
        occlusion_min_fraction=0.2,
        occlusion_max_fraction=0.2,
        occlusion_alpha=0.5,
        preserve_input_max=True,
    )

    original, _, _ = base[0]
    x1, _, meta = dataset[0]
    x2, _, _ = dataset[0]

    torch.testing.assert_close(x1, x2)
    assert not torch.allclose(original, x1)
    assert float(torch.max(x1)) == pytest.approx(float(torch.max(original)))
    assert meta["degradation_enabled"] is True
    assert meta["structured_reflectance_strength"] == 0.1
    assert meta["occlusion_probability"] == 1.0

import pytest

torch = pytest.importorskip("torch")

from train import SliceInputViewDataset, ema, stratified_split


class DummySample:
    def __init__(self, label: int) -> None:
        self.label = label


class DummyDataset:
    def __init__(self, labels: list[int]) -> None:
        self.samples = [DummySample(label) for label in labels]


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
                torch.full((1, 2, 2), 1.0),
                torch.full((1, 2, 2), 2.0),
                torch.full((1, 2, 2), 3.0),
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


def test_slice_input_view_dataset_can_return_single_2d_gate() -> None:
    dataset = SliceInputViewDataset(DummySliceDataset(), input_mode="single-gate", single_gate_index=1)

    x, label, meta = dataset[0]

    assert x.shape == (1, 1, 2, 2)
    assert torch.all(x == 2.0)
    assert label == 0
    assert meta["input_mode"] == "single-gate"
    assert meta["single_gate_index"] == 1


def test_slice_input_view_dataset_can_black_out_other_gates() -> None:
    dataset = SliceInputViewDataset(DummySliceDataset(), input_mode="single-gate-black", single_gate_index=2)

    x, _, meta = dataset[0]

    assert x.shape == (3, 1, 2, 2)
    assert torch.all(x[0] == 0.0)
    assert torch.all(x[1] == 0.0)
    assert torch.all(x[2] == 3.0)
    assert meta["input_mode"] == "single-gate-black"


def test_slice_input_view_dataset_rejects_invalid_gate_index() -> None:
    with pytest.raises(ValueError, match="single_gate_index"):
        SliceInputViewDataset(DummySliceDataset(), input_mode="single-gate", single_gate_index=3)

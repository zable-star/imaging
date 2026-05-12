from pathlib import Path

from convert_off_to_obj_dataset import (
    DEFAULT_TEST_PER_CLASS,
    DEFAULT_TRAIN_PER_CLASS,
    DatasetSelection,
    convert_off_to_obj,
    select_modelnet_files,
)


def test_convert_off_to_obj_writes_vertices_and_faces(tmp_path: Path) -> None:
    off_path = tmp_path / "sample.off"
    obj_path = tmp_path / "sample.obj"
    off_path.write_text(
        "\n".join(
            [
                "OFF",
                "4 2 0",
                "0 0 0",
                "1 0 0",
                "1 1 0",
                "0 1 0",
                "3 0 1 2",
                "4 0 1 2 3",
            ]
        ),
        encoding="utf-8",
    )

    convert_off_to_obj(off_path, obj_path)

    assert obj_path.read_text(encoding="utf-8").splitlines() == [
        "v 0 0 0",
        "v 1 0 0",
        "v 1 1 0",
        "v 0 1 0",
        "f 1 2 3",
        "f 1 2 3 4",
    ]


def test_select_modelnet_files_limits_each_split(tmp_path: Path) -> None:
    root = tmp_path / "ModelNet10"
    for split, count in {"train": 3, "test": 2}.items():
        split_dir = root / "chair" / split
        split_dir.mkdir(parents=True)
        for idx in range(count):
            (split_dir / f"chair_{idx:04d}.off").write_text("OFF\n0 0 0\n", encoding="utf-8")

    selected = select_modelnet_files(
        root,
        DatasetSelection(classes=("chair",), train_per_class=2, test_per_class=1),
    )

    assert [item.split for item in selected] == ["test", "train", "train"]
    assert [item.class_name for item in selected] == ["chair", "chair", "chair"]


def test_default_dataset_size_is_500_for_five_classes() -> None:
    assert DEFAULT_TRAIN_PER_CLASS == 80
    assert DEFAULT_TEST_PER_CLASS == 20
    assert len(DatasetSelection().classes) == 5
    assert len(DatasetSelection().classes) * (DEFAULT_TRAIN_PER_CLASS + DEFAULT_TEST_PER_CLASS) == 500

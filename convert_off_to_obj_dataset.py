from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CLASSES = ("chair", "desk", "sofa", "bed", "toilet")
DEFAULT_TRAIN_PER_CLASS = 80
DEFAULT_TEST_PER_CLASS = 20


@dataclass(frozen=True)
class DatasetSelection:
    classes: tuple[str, ...] = DEFAULT_CLASSES
    train_per_class: int = DEFAULT_TRAIN_PER_CLASS
    test_per_class: int = DEFAULT_TEST_PER_CLASS


@dataclass(frozen=True)
class SelectedOffFile:
    class_name: str
    split: str
    path: Path


def _non_comment_lines(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if line and not line.startswith("#"):
                yield line


def _read_counts(lines) -> tuple[int, int]:
    first = next(lines)
    if first == "OFF":
        counts = next(lines).split()
    elif first.startswith("OFF"):
        counts = first[3:].strip().split()
    else:
        raise ValueError("OFF file must start with OFF")

    if len(counts) < 2:
        raise ValueError("OFF counts line must contain vertex and face counts")
    return int(counts[0]), int(counts[1])


def convert_off_to_obj(off_path: str | Path, obj_path: str | Path) -> None:
    off_path = Path(off_path)
    obj_path = Path(obj_path)
    lines = _non_comment_lines(off_path)
    vertex_count, face_count = _read_counts(lines)

    vertices = [next(lines).split()[:3] for _ in range(vertex_count)]
    faces = []
    for _ in range(face_count):
        parts = next(lines).split()
        if not parts:
            continue
        face_size = int(parts[0])
        indices = [str(int(idx) + 1) for idx in parts[1 : 1 + face_size]]
        faces.append(indices)

    obj_path.parent.mkdir(parents=True, exist_ok=True)
    with obj_path.open("w", encoding="utf-8", newline="\n") as f:
        for vertex in vertices:
            f.write(f"v {' '.join(vertex)}\n")
        for face in faces:
            f.write(f"f {' '.join(face)}\n")


def select_modelnet_files(modelnet_root: str | Path, selection: DatasetSelection) -> list[SelectedOffFile]:
    modelnet_root = Path(modelnet_root)
    selected: list[SelectedOffFile] = []
    split_limits = {"test": selection.test_per_class, "train": selection.train_per_class}

    for class_name in selection.classes:
        for split, limit in split_limits.items():
            split_dir = modelnet_root / class_name / split
            files = sorted(split_dir.glob("*.off"))[:limit]
            if len(files) < limit:
                raise RuntimeError(f"{split_dir} has only {len(files)} OFF files, expected {limit}.")
            selected.extend(SelectedOffFile(class_name=class_name, split=split, path=path) for path in files)

    return selected


def convert_selection(
    modelnet_root: str | Path,
    output_root: str | Path,
    selection: DatasetSelection,
) -> int:
    output_root = Path(output_root)
    selected_files = select_modelnet_files(modelnet_root, selection)
    for item in selected_files:
        obj_path = output_root / item.class_name / item.split / f"{item.path.stem}.obj"
        convert_off_to_obj(item.path, obj_path)
    return len(selected_files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select five ModelNet10 classes and convert OFF files to OBJ.")
    parser.add_argument(
        "--modelnet-root",
        type=Path,
        default=Path("origindataset") / "ModelNet10" / "ModelNet10",
        help="Path containing ModelNet10 class folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("obj_dataset"),
        help="Output folder for converted OBJ files.",
    )
    parser.add_argument("--classes", nargs="+", default=list(DEFAULT_CLASSES), help="Class names to convert.")
    parser.add_argument("--train-per-class", type=int, default=DEFAULT_TRAIN_PER_CLASS)
    parser.add_argument("--test-per-class", type=int, default=DEFAULT_TEST_PER_CLASS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection = DatasetSelection(
        classes=tuple(args.classes),
        train_per_class=args.train_per_class,
        test_per_class=args.test_per_class,
    )
    total = convert_selection(args.modelnet_root, args.output_root, selection)
    print(f"Converted {total} OFF files to OBJ under {args.output_root}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset


GATE_PATTERN = re.compile(r"^(?P<base>.+?)_gate_(?P<gate>\d+)\.png$")


@dataclass
class SliceSample:
    sample_id: str
    label: int
    class_name: str
    paths: list[Path]


def pil_to_tensor_gray(img: Image.Image) -> torch.Tensor:
    img = img.convert("L")
    data = torch.as_tensor(bytearray(img.tobytes()), dtype=torch.uint8)
    data = data.view(img.size[1], img.size[0]).float() / 255.0
    return data.unsqueeze(0)


class MultiSliceObjectDataset(Dataset):
    def __init__(
        self,
        class_dirs: dict[str, str | Path],
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        expected_num_slices: int | None = None,
    ) -> None:
        self.transform = transform or pil_to_tensor_gray
        self.expected_num_slices = expected_num_slices
        self.class_names = list(class_dirs.keys())
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.samples: list[SliceSample] = []

        for class_name, class_dir in class_dirs.items():
            grouped = self._group_paths(Path(class_dir))
            for sample_id, path_map in sorted(grouped.items()):
                ordered_gates = sorted(path_map.keys())
                paths = [path_map[g] for g in ordered_gates]
                if self.expected_num_slices is not None and len(paths) != self.expected_num_slices:
                    continue
                self.samples.append(
                    SliceSample(
                        sample_id=sample_id,
                        label=self.class_to_idx[class_name],
                        class_name=class_name,
                        paths=paths,
                    )
                )

        if not self.samples:
            raise RuntimeError("No valid multi-slice samples were found.")

        slice_counts = {len(sample.paths) for sample in self.samples}
        if len(slice_counts) != 1:
            raise RuntimeError(
                f"Inconsistent slice counts detected: {sorted(slice_counts)}. "
                "Please fix the dataset or set expected_num_slices."
            )

        self.num_slices = len(self.samples[0].paths)

    @staticmethod
    def _group_paths(class_dir: Path) -> dict[str, dict[int, Path]]:
        grouped: dict[str, dict[int, Path]] = {}
        for path in sorted(class_dir.rglob("*.png")):
            match = GATE_PATTERN.match(path.name)
            if not match:
                continue
            base = match.group("base")
            gate = int(match.group("gate"))
            grouped.setdefault(base, {})[gate] = path
        return grouped

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        slices = []
        for path in sample.paths:
            img = Image.open(path)
            tensor = self.transform(img)
            slices.append(tensor)

        x = torch.stack(slices, dim=0)  # [S, 1, H, W]
        meta = {
            "sample_id": sample.sample_id,
            "class_name": sample.class_name,
            "paths": [str(p) for p in sample.paths],
        }
        return x, sample.label, meta

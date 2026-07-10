from __future__ import annotations

import csv
import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset, Subset

from dataset import MultiSliceObjectDataset
from model import FUSION_MODES, SliceAttentionClassifier


ROOT = Path(r"E:\wjz\test1\dataset\dataset_obj")
BASELINE_ROOT = ROOT / "slice_attention_baseline"
DATASET_ROOT = BASELINE_ROOT / "dataset"
ARTIFACT_DIR = BASELINE_ROOT / "artifacts"
MODEL_PATH = BASELINE_ROOT / "slice_attention_model.pth"
DEFAULT_CLASSES = ["chair", "desk", "sofa", "bed", "toilet", "image2d"]
INPUT_MODES = ["multi", "single-gate", "single-gate-black"]
GATE_DROPOUT_MODES = ["none", "fixed", "random"]

IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 30
LR = 3e-4
MIN_LR = 1e-5
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
GRAD_CLIP = 1.0
EMA_ALPHA = 0.35
VAL_RATIO = 0.2
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Train slice-attention classifier on gated 3D object slices.")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    parser.add_argument("--expected-num-slices", type=int, default=3)
    parser.add_argument("--input-mode", choices=INPUT_MODES, default="multi")
    parser.add_argument("--single-gate-index", type=int, default=0)
    parser.add_argument("--fusion-mode", choices=FUSION_MODES, default="attention")
    parser.add_argument("--gaussian-noise-std", type=float, default=0.0)
    parser.add_argument("--poisson-peak", type=float, default=0.0)
    parser.add_argument("--background-scatter", type=float, default=0.0)
    parser.add_argument("--background-sigma", type=float, default=24.0)
    parser.add_argument("--gate-attenuation-index", type=int, default=-1)
    parser.add_argument("--gate-attenuation-factor", type=float, default=1.0)
    parser.add_argument("--gate-dropout-mode", choices=GATE_DROPOUT_MODES, default="none")
    parser.add_argument("--gate-dropout-index", type=int, default=0)
    parser.add_argument("--structured-reflectance-strength", type=float, default=0.0)
    parser.add_argument("--structured-background-strength", type=float, default=0.0)
    parser.add_argument("--structured-nuisance-grid-size", type=int, default=9)
    parser.add_argument("--occlusion-probability", type=float, default=0.0)
    parser.add_argument("--occlusion-min-fraction", type=float, default=0.04)
    parser.add_argument("--occlusion-max-fraction", type=float, default=0.12)
    parser.add_argument("--occlusion-alpha", type=float, default=0.6)
    parser.add_argument("--preserve-input-max", action="store_true")
    parser.add_argument(
        "--degradation-probability",
        type=float,
        default=1.0,
        help="Deterministic per-sample probability of applying configured image/gate degradations.",
    )
    parser.add_argument(
        "--pretrained-model-path",
        type=Path,
        default=None,
        help="Optional checkpoint used to initialize the model for small-sample transfer learning.",
    )
    parser.add_argument(
        "--pretrained-include-classifier",
        action="store_true",
        help="Also load classifier weights when they are shape-compatible. By default, classifier weights are skipped.",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze the shared SliceEncoder after loading optional pretrained weights.",
    )
    parser.add_argument(
        "--freeze-attention",
        action="store_true",
        help="Freeze the gate attention scorer. Usually keep this false for military fine-tuning.",
    )
    parser.add_argument(
        "--freeze-residual",
        action="store_true",
        help="Freeze the residual projection used by attention_residual fusion.",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count. Use 2-4 on the lab 3090 machine if disk I/O becomes the bottleneck.",
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Use CUDA automatic mixed precision when a GPU is available.",
    )
    parser.add_argument(
        "--cudnn-benchmark",
        action="store_true",
        help="Enable cuDNN benchmark for faster fixed-size GPU training; disables deterministic cuDNN mode.",
    )
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--min-lr", type=float, default=MIN_LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--label-smoothing", type=float, default=LABEL_SMOOTHING)
    parser.add_argument("--grad-clip", type=float, default=GRAD_CLIP)
    parser.add_argument("--ema-alpha", type=float, default=EMA_ALPHA)
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument(
        "--split-group-by-sample-id",
        action="store_true",
        help=(
            "Keep samples with the same sample_id in the same split. "
            "Use this for paired true/false datasets generated from the same source model."
        ),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def build_class_dirs(dataset_root: Path, class_names: list[str]) -> dict[str, Path]:
    class_dirs = {class_name: dataset_root / class_name for class_name in class_names}
    missing = [str(path) for path in class_dirs.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing class slice directories:\n" + "\n".join(missing))
    return class_dirs


def set_seed(seed: int, cudnn_benchmark: bool = False) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        # 优先保证训练曲线可复现，而不是使用 cuDNN 中最快但可能不确定的卷积算法。
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resize_tensor_img(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] == (IMAGE_SIZE, IMAGE_SIZE):
        return x
    return torch.nn.functional.interpolate(
        x.unsqueeze(0), size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
    ).squeeze(0)


def transform_gray(img):
    from dataset import pil_to_tensor_gray

    return resize_tensor_img(pil_to_tensor_gray(img))


def collate_meta(batch):
    xs, ys, metas = zip(*batch)
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long), list(metas)


def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=enabled)
    return nullcontext()


def make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


class SliceInputViewDataset(Dataset):
    """Convert a multi-slice sample into a selected experimental input view."""

    def __init__(self, dataset: Dataset, input_mode: str, single_gate_index: int) -> None:
        if input_mode not in INPUT_MODES:
            raise ValueError(f"Unsupported input_mode: {input_mode}")
        self.dataset = dataset
        self.input_mode = input_mode
        self.single_gate_index = single_gate_index
        self.samples = dataset.samples
        self.class_names = dataset.class_names
        self.num_slices = self._infer_num_slices()
        if not 0 <= single_gate_index < self.num_slices:
            raise ValueError(
                f"single_gate_index must be in [0, {self.num_slices - 1}], got {single_gate_index}"
            )
        self.effective_num_input_slices = 1 if input_mode == "single-gate" else self.num_slices

    def _infer_num_slices(self) -> int:
        if hasattr(self.dataset, "num_slices"):
            return int(self.dataset.num_slices)
        x, _, _ = self.dataset[0]
        return int(x.shape[0])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        x, y, meta = self.dataset[index]
        meta = dict(meta)
        meta["input_mode"] = self.input_mode
        meta["single_gate_index"] = self.single_gate_index

        if self.input_mode == "multi":
            return x, y, meta

        selected = x[self.single_gate_index : self.single_gate_index + 1]
        if self.input_mode == "single-gate":
            return selected, y, meta

        black_input = torch.zeros_like(x)
        black_input[self.single_gate_index] = x[self.single_gate_index]
        return black_input, y, meta


class SliceDegradationDataset(Dataset):
    """Apply deterministic gated-image degradations for robustness experiments."""

    def __init__(
        self,
        dataset: Dataset,
        seed: int,
        gaussian_noise_std: float = 0.0,
        poisson_peak: float = 0.0,
        background_scatter: float = 0.0,
        background_sigma: float = 24.0,
        gate_attenuation_index: int = -1,
        gate_attenuation_factor: float = 1.0,
        gate_dropout_mode: str = "none",
        gate_dropout_index: int = 0,
        structured_reflectance_strength: float = 0.0,
        structured_background_strength: float = 0.0,
        structured_nuisance_grid_size: int = 9,
        occlusion_probability: float = 0.0,
        occlusion_min_fraction: float = 0.04,
        occlusion_max_fraction: float = 0.12,
        occlusion_alpha: float = 0.6,
        preserve_input_max: bool = False,
        degradation_probability: float = 1.0,
    ) -> None:
        if gate_dropout_mode not in GATE_DROPOUT_MODES:
            raise ValueError(f"Unsupported gate_dropout_mode: {gate_dropout_mode}")
        if gaussian_noise_std < 0.0:
            raise ValueError("gaussian_noise_std must be non-negative")
        if poisson_peak < 0.0:
            raise ValueError("poisson_peak must be non-negative")
        if background_scatter < 0.0:
            raise ValueError("background_scatter must be non-negative")
        if background_sigma <= 0.0:
            raise ValueError("background_sigma must be positive")
        if not 0.0 <= gate_attenuation_factor <= 1.0:
            raise ValueError("gate_attenuation_factor must be in [0, 1]")
        if structured_reflectance_strength < 0.0:
            raise ValueError("structured_reflectance_strength must be non-negative")
        if structured_background_strength < 0.0:
            raise ValueError("structured_background_strength must be non-negative")
        if structured_nuisance_grid_size < 2:
            raise ValueError("structured_nuisance_grid_size must be at least 2")
        if not 0.0 <= occlusion_probability <= 1.0:
            raise ValueError("occlusion_probability must be in [0, 1]")
        if occlusion_min_fraction < 0.0 or occlusion_max_fraction < occlusion_min_fraction:
            raise ValueError("occlusion fraction range must be non-negative and ordered")
        if not 0.0 <= occlusion_alpha <= 1.0:
            raise ValueError("occlusion_alpha must be in [0, 1]")
        if not 0.0 <= degradation_probability <= 1.0:
            raise ValueError("degradation_probability must be in [0, 1]")

        self.dataset = dataset
        self.seed = seed
        self.gaussian_noise_std = gaussian_noise_std
        self.poisson_peak = poisson_peak
        self.background_scatter = background_scatter
        self.background_sigma = background_sigma
        self.gate_attenuation_index = gate_attenuation_index
        self.gate_attenuation_factor = gate_attenuation_factor
        self.gate_dropout_mode = gate_dropout_mode
        self.gate_dropout_index = gate_dropout_index
        self.structured_reflectance_strength = structured_reflectance_strength
        self.structured_background_strength = structured_background_strength
        self.structured_nuisance_grid_size = structured_nuisance_grid_size
        self.occlusion_probability = occlusion_probability
        self.occlusion_min_fraction = occlusion_min_fraction
        self.occlusion_max_fraction = occlusion_max_fraction
        self.occlusion_alpha = occlusion_alpha
        self.preserve_input_max = preserve_input_max
        self.degradation_probability = degradation_probability
        self.samples = dataset.samples
        self.class_names = dataset.class_names
        self.num_slices = dataset.num_slices
        self.effective_num_input_slices = dataset.effective_num_input_slices

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        x, y, meta = self.dataset[index]
        meta = dict(meta)
        configured = self._enabled()
        enabled = configured and self.should_apply_degradation(index)
        if enabled:
            x = self.apply_degradation(x, index)
        meta.update(self.degradation_meta(index, x.shape[0], configured, enabled))
        return x, y, meta

    def _enabled(self) -> bool:
        return any(
            [
                self.gaussian_noise_std > 0.0,
                self.poisson_peak > 0.0,
                self.background_scatter > 0.0,
                self.gate_attenuation_index >= 0 and self.gate_attenuation_factor < 1.0,
                self.gate_dropout_mode != "none",
                self.structured_reflectance_strength > 0.0,
                self.structured_background_strength > 0.0,
                self.occlusion_probability > 0.0,
            ]
        )

    def _generator(self, index: int, offset: int = 0) -> torch.Generator:
        return torch.Generator().manual_seed(self.seed + index * 1009 + offset)

    def should_apply_degradation(self, index: int) -> bool:
        if self.degradation_probability >= 1.0:
            return True
        if self.degradation_probability <= 0.0:
            return False
        draw = torch.rand((), generator=self._generator(index, offset=73)).item()
        return bool(draw < self.degradation_probability)

    def degradation_meta(
        self,
        index: int,
        num_slices: int,
        configured: bool,
        enabled: bool,
    ) -> dict[str, object]:
        drop_gate = self.resolve_dropout_gate(index, num_slices) if enabled else -1
        return {
            "degradation_configured": configured,
            "degradation_enabled": enabled,
            "degradation_probability": self.degradation_probability,
            "gaussian_noise_std": self.gaussian_noise_std,
            "poisson_peak": self.poisson_peak,
            "background_scatter": self.background_scatter,
            "background_sigma": self.background_sigma,
            "gate_attenuation_index": self.gate_attenuation_index,
            "gate_attenuation_factor": self.gate_attenuation_factor,
            "gate_dropout_mode": self.gate_dropout_mode,
            "gate_dropout_index": self.gate_dropout_index,
            "resolved_gate_dropout_index": drop_gate,
            "structured_reflectance_strength": self.structured_reflectance_strength,
            "structured_background_strength": self.structured_background_strength,
            "structured_nuisance_grid_size": self.structured_nuisance_grid_size,
            "occlusion_probability": self.occlusion_probability,
            "occlusion_min_fraction": self.occlusion_min_fraction,
            "occlusion_max_fraction": self.occlusion_max_fraction,
            "occlusion_alpha": self.occlusion_alpha,
            "preserve_input_max": self.preserve_input_max,
        }

    def resolve_dropout_gate(self, index: int, num_slices: int) -> int:
        if self.gate_dropout_mode == "none":
            return -1
        if self.gate_dropout_mode == "fixed":
            return self.gate_dropout_index % num_slices
        generator = self._generator(index, offset=17)
        return int(torch.randint(0, num_slices, (1,), generator=generator).item())

    def apply_degradation(self, x: torch.Tensor, index: int) -> torch.Tensor:
        out = x.clone()
        num_slices = out.shape[0]
        input_max = torch.max(out).detach()

        if 0 <= self.gate_attenuation_index < num_slices and self.gate_attenuation_factor < 1.0:
            out[self.gate_attenuation_index] *= self.gate_attenuation_factor

        drop_gate = self.resolve_dropout_gate(index, num_slices)
        if drop_gate >= 0:
            out[drop_gate] = 0.0

        structured_enabled = any(
            [
                self.structured_reflectance_strength > 0.0,
                self.structured_background_strength > 0.0,
                self.occlusion_probability > 0.0,
            ]
        )
        if structured_enabled:
            out = self.apply_structured_nuisance(out, index)
            if self.preserve_input_max and float(input_max) > 0.0:
                max_value = torch.max(out).detach()
                if float(max_value) > 0.0:
                    out = out * (input_max / max_value)

        if self.background_scatter > 0.0:
            out = out + self._background_like(out, index) * self.background_scatter

        if self.poisson_peak > 0.0:
            scaled = torch.clamp(out, 0.0, 1.0) * self.poisson_peak
            sampled = torch.poisson(scaled, generator=self._generator(index, offset=31))
            out = sampled / self.poisson_peak

        if self.gaussian_noise_std > 0.0:
            noise = torch.randn(
                out.shape,
                generator=self._generator(index, offset=47),
                dtype=out.dtype,
                device=out.device,
            )
            out = out + noise * self.gaussian_noise_std

        return torch.clamp(out, 0.0, 1.0)

    def apply_structured_nuisance(self, x: torch.Tensor, index: int) -> torch.Tensor:
        out = x
        if self.structured_reflectance_strength > 0.0:
            field = self._low_frequency_field(x, index, offset=83)
            reflectance = torch.clamp(1.0 + field * self.structured_reflectance_strength, 0.45, 1.65)
            out = out * reflectance

        if self.occlusion_probability > 0.0 and self._should_apply_occlusion(index):
            out = out * self._occlusion_multiplier(out, index)

        if self.structured_background_strength > 0.0:
            field = self._low_frequency_field(x, index, offset=97)
            background = torch.clamp(0.5 + 0.22 * field, 0.0, 1.0)
            out = out + background * self.structured_background_strength
        return out

    def _low_frequency_field(self, reference: torch.Tensor, index: int, offset: int) -> torch.Tensor:
        h, w = reference.shape[-2:]
        grid = self.structured_nuisance_grid_size
        coarse = torch.randn(
            (1, 1, grid, grid),
            generator=self._generator(index, offset=offset),
            dtype=reference.dtype,
            device=reference.device,
        )
        field = torch.nn.functional.interpolate(coarse, size=(h, w), mode="bilinear", align_corners=False)
        field = field.squeeze(0)
        field = field - field.mean()
        std = field.std()
        if float(std) > 1e-6:
            field = field / std
        return field

    def _should_apply_occlusion(self, index: int) -> bool:
        if self.occlusion_probability >= 1.0:
            return True
        draw = torch.rand((), generator=self._generator(index, offset=109)).item()
        return bool(draw < self.occlusion_probability)

    def _occlusion_multiplier(self, reference: torch.Tensor, index: int) -> torch.Tensor:
        foreground = reference.max(dim=0).values.squeeze(0) > 1e-4
        multiplier = torch.ones_like(foreground, dtype=reference.dtype)
        if not bool(foreground.any()):
            return multiplier.unsqueeze(0)

        coords = foreground.nonzero()
        y_min = int(coords[:, 0].min().item())
        y_max = int(coords[:, 0].max().item())
        x_min = int(coords[:, 1].min().item())
        x_max = int(coords[:, 1].max().item())
        fg_h = max(y_max - y_min + 1, 1)
        fg_w = max(x_max - x_min + 1, 1)
        fraction_draw = torch.rand((), generator=self._generator(index, offset=127)).item()
        fraction = self.occlusion_min_fraction + (self.occlusion_max_fraction - self.occlusion_min_fraction) * fraction_draw
        aspect = 0.55 + 1.25 * torch.rand((), generator=self._generator(index, offset=131)).item()
        occ_h = max(2, min(fg_h, int(round(fg_h * max(fraction / max(aspect, 1e-3), 1e-3) ** 0.5))))
        occ_w = max(2, min(fg_w, int(round(occ_h * aspect))))
        cy = int(torch.randint(y_min, y_max + 1, (1,), generator=self._generator(index, offset=137)).item())
        cx = int(torch.randint(x_min, x_max + 1, (1,), generator=self._generator(index, offset=139)).item())
        y0 = max(y_min, min(y_max + 1 - occ_h, cy - occ_h // 2))
        x0 = max(x_min, min(x_max + 1 - occ_w, cx - occ_w // 2))
        multiplier[y0 : y0 + occ_h, x0 : x0 + occ_w] = self.occlusion_alpha
        return multiplier.unsqueeze(0)

    def _background_like(self, reference: torch.Tensor, index: int) -> torch.Tensor:
        noise = torch.rand(
            reference.shape,
            generator=self._generator(index, offset=61),
            dtype=reference.dtype,
            device=reference.device,
        )
        kernel_size = max(3, int(round(self.background_sigma)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        pad = kernel_size // 2
        pooled = torch.nn.functional.avg_pool2d(
            noise.reshape(-1, 1, noise.shape[-2], noise.shape[-1]),
            kernel_size=kernel_size,
            stride=1,
            padding=pad,
        )
        return pooled.reshape_as(reference)


def stratified_split(dataset: MultiSliceObjectDataset, val_ratio: float, seed: int) -> tuple[Subset, Subset]:
    """按类别分层划分训练集和验证集，避免验证集中某些类别过多或过少。"""

    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")

    indices_by_label: dict[int, list[int]] = {}
    for index, sample in enumerate(dataset.samples):
        indices_by_label.setdefault(sample.label, []).append(index)

    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []

    for label in sorted(indices_by_label):
        class_indices = indices_by_label[label][:]
        rng.shuffle(class_indices)
        val_count = int(round(len(class_indices) * val_ratio))
        if len(class_indices) > 1:
            val_count = min(max(val_count, 1), len(class_indices) - 1)

        val_indices.extend(class_indices[:val_count])
        train_indices.extend(class_indices[val_count:])

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def split_group_id(sample_id: str) -> str:
    """Return the leakage-control group id for paired and multi-view samples."""

    if sample_id.lower().startswith("domain_") and "__" in sample_id:
        sample_id = sample_id.split("__", 1)[1]
    if sample_id.lower().startswith("view_") and "__" in sample_id:
        return sample_id.split("__", 1)[1]
    return sample_id


def stratified_group_split(dataset: MultiSliceObjectDataset, val_ratio: float, seed: int) -> tuple[Subset, Subset]:
    """Split by sample_id groups to avoid paired true/false source-model leakage."""

    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")

    groups: dict[str, list[int]] = {}
    for index, sample in enumerate(dataset.samples):
        groups.setdefault(split_group_id(sample.sample_id), []).append(index)

    groups_by_primary_label: dict[int, list[list[int]]] = {}
    for indices in groups.values():
        labels = sorted({dataset.samples[index].label for index in indices})
        primary_label = labels[0]
        groups_by_primary_label.setdefault(primary_label, []).append(indices)

    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for primary_label in sorted(groups_by_primary_label):
        label_groups = groups_by_primary_label[primary_label][:]
        rng.shuffle(label_groups)
        val_count = int(round(len(label_groups) * val_ratio))
        if len(label_groups) > 1:
            val_count = min(max(val_count, 1), len(label_groups) - 1)
        for group in label_groups[:val_count]:
            val_indices.extend(group)
        for group in label_groups[val_count:]:
            train_indices.extend(group)

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def ema(values: list[float], alpha: float) -> list[float]:
    """指数滑动平均，只用于让曲线图更容易观察趋势。"""

    if not values:
        return []
    alpha = min(max(alpha, 0.0), 1.0)
    smoothed = [values[0]]
    for value in values[1:]:
        smoothed.append(alpha * value + (1.0 - alpha) * smoothed[-1])
    return smoothed


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def set_module_trainable(module: nn.Module | None, trainable: bool) -> None:
    if module is None:
        return
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def configure_finetuning(
    model: SliceAttentionClassifier,
    freeze_encoder: bool = False,
    freeze_attention: bool = False,
    freeze_residual: bool = False,
) -> None:
    """Apply lightweight transfer-learning freezes for small military datasets."""

    set_module_trainable(model.encoder, not freeze_encoder)
    set_module_trainable(model.attention, not freeze_attention)
    set_module_trainable(model.residual_projection, not freeze_residual)


def _extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            nested = checkpoint.get(key)
            if isinstance(nested, dict):
                return nested
        if all(isinstance(key, str) for key in checkpoint):
            return checkpoint
    raise ValueError("Checkpoint does not contain a PyTorch state_dict.")


def load_pretrained_weights(
    model: SliceAttentionClassifier,
    checkpoint_path: Path,
    device: torch.device,
    include_classifier: bool = False,
) -> dict[str, object]:
    """Load shape-compatible weights while allowing a new classifier head."""

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    pretrained_state = _extract_state_dict(checkpoint)
    current_state = model.state_dict()
    compatible_state = {}
    loaded_keys: list[str] = []
    skipped_keys: list[str] = []

    for key, value in pretrained_state.items():
        normalized_key = key[7:] if key.startswith("module.") else key
        if normalized_key.startswith("classifier.") and not include_classifier:
            skipped_keys.append(normalized_key)
            continue
        if normalized_key not in current_state:
            skipped_keys.append(normalized_key)
            continue
        if tuple(current_state[normalized_key].shape) != tuple(value.shape):
            skipped_keys.append(normalized_key)
            continue
        compatible_state[normalized_key] = value
        loaded_keys.append(normalized_key)

    current_state.update(compatible_state)
    model.load_state_dict(current_state)

    return {
        "path": str(checkpoint_path),
        "include_classifier": include_classifier,
        "loaded_keys": loaded_keys,
        "skipped_keys": skipped_keys,
        "loaded_key_count": len(loaded_keys),
        "skipped_key_count": len(skipped_keys),
    }


def plot_curves(history: dict[str, list[float]], out_path: Path) -> None:
    epochs = list(range(1, len(history["train_loss"]) + 1))
    # 保留原始指标曲线，同时增加 EMA 平滑曲线，方便观察整体趋势。
    train_loss_smooth = ema(history["train_loss"], history["ema_alpha"][0])
    val_loss_smooth = ema(history["val_loss"], history["ema_alpha"][0])
    val_acc_smooth = ema(history["val_acc"], history["ema_alpha"][0])

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], color="tab:blue", alpha=0.25, label="train raw")
    plt.plot(epochs, history["val_loss"], color="tab:orange", alpha=0.25, label="val raw")
    plt.plot(epochs, train_loss_smooth, color="tab:blue", linewidth=2.0, label="train EMA")
    plt.plot(epochs, val_loss_smooth, color="tab:orange", linewidth=2.0, label="val EMA")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_acc"], color="tab:green", alpha=0.25, label="val acc raw")
    plt.plot(epochs, val_acc_smooth, color="tab:green", linewidth=2.0, label="val acc EMA")
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_confusion(cm, class_names, out_path: Path) -> None:
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xticks(range(len(class_names)), class_names)
    plt.yticks(range(len(class_names)), class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp: bool = False):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    y_true = []
    y_pred = []
    rows = []

    for x, y, metas in loader:
        x = x.to(device)
        y = y.to(device)
        with autocast_context(device, use_amp):
            logits, attn_weights, _ = model(x)
            loss = criterion(logits, y)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
        correct += (preds == y).sum().item()

        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

        for meta, prob, pred, gt, attn in zip(metas, probs.cpu(), preds.cpu(), y.cpu(), attn_weights.cpu()):
            row = {
                "sample_id": meta["sample_id"],
                "class_name": meta["class_name"],
                "pred": int(pred.item()),
                "gt": int(gt.item()),
            }
            for key in (
                "input_mode",
                "single_gate_index",
                "degradation_configured",
                "degradation_enabled",
                "degradation_probability",
                "gaussian_noise_std",
                "poisson_peak",
                "background_scatter",
                "background_sigma",
                "gate_attenuation_index",
                "gate_attenuation_factor",
                "gate_dropout_mode",
                "resolved_gate_dropout_index",
                "structured_reflectance_strength",
                "structured_background_strength",
                "structured_nuisance_grid_size",
                "occlusion_probability",
                "occlusion_min_fraction",
                "occlusion_max_fraction",
                "occlusion_alpha",
                "preserve_input_max",
            ):
                if key in meta:
                    row[key] = meta[key]
            for idx, value in enumerate(attn.tolist()):
                row[f"attn_gate_{idx}"] = float(value)
            for idx, value in enumerate(prob.tolist()):
                row[f"prob_class_{idx}"] = float(value)
            rows.append(row)

    return {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
        "y_true": y_true,
        "y_pred": y_pred,
        "rows": rows,
    }


def save_attention_csv(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_history_csv(history: dict[str, list[float]], out_path: Path) -> None:
    alpha = history["ema_alpha"][0]
    # CSV 同时保存原始值和平滑值，后续重新画图或写论文时可以直接使用。
    fieldnames = [
        "epoch",
        "lr",
        "train_loss",
        "val_loss",
        "val_acc",
        "train_loss_ema",
        "val_loss_ema",
        "val_acc_ema",
    ]
    rows = []
    train_loss_smooth = ema(history["train_loss"], alpha)
    val_loss_smooth = ema(history["val_loss"], alpha)
    val_acc_smooth = ema(history["val_acc"], alpha)
    for idx, (train_loss, val_loss, val_acc, lr) in enumerate(
        zip(history["train_loss"], history["val_loss"], history["val_acc"], history["lr"]),
        start=1,
    ):
        rows.append(
            {
                "epoch": idx,
                "lr": lr,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "train_loss_ema": train_loss_smooth[idx - 1],
                "val_loss_ema": val_loss_smooth[idx - 1],
                "val_acc_ema": val_acc_smooth[idx - 1],
            }
        )

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    set_seed(args.seed, cudnn_benchmark=args.cudnn_benchmark)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)

    base_dataset = MultiSliceObjectDataset(
        class_dirs=build_class_dirs(args.dataset_root, args.classes),
        transform=transform_gray,
        expected_num_slices=args.expected_num_slices,
    )
    dataset = SliceInputViewDataset(base_dataset, args.input_mode, args.single_gate_index)
    dataset = SliceDegradationDataset(
        dataset,
        seed=args.seed,
        gaussian_noise_std=args.gaussian_noise_std,
        poisson_peak=args.poisson_peak,
        background_scatter=args.background_scatter,
        background_sigma=args.background_sigma,
        gate_attenuation_index=args.gate_attenuation_index,
        gate_attenuation_factor=args.gate_attenuation_factor,
        gate_dropout_mode=args.gate_dropout_mode,
        gate_dropout_index=args.gate_dropout_index,
        structured_reflectance_strength=args.structured_reflectance_strength,
        structured_background_strength=args.structured_background_strength,
        structured_nuisance_grid_size=args.structured_nuisance_grid_size,
        occlusion_probability=args.occlusion_probability,
        occlusion_min_fraction=args.occlusion_min_fraction,
        occlusion_max_fraction=args.occlusion_max_fraction,
        occlusion_alpha=args.occlusion_alpha,
        preserve_input_max=args.preserve_input_max,
        degradation_probability=args.degradation_probability,
    )

    if args.split_group_by_sample_id:
        train_set, val_set = stratified_group_split(dataset, args.val_ratio, args.seed)
    else:
        train_set, val_set = stratified_split(dataset, args.val_ratio, args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_meta,
        generator=generator,
        pin_memory=device.type == "cuda",
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_meta,
        pin_memory=device.type == "cuda",
        num_workers=args.num_workers,
    )

    model = SliceAttentionClassifier(
        num_classes=len(dataset.class_names),
        fusion_mode=args.fusion_mode,
        num_slices=dataset.effective_num_input_slices,
    ).to(device)
    pretrained_report = None
    if args.pretrained_model_path is not None:
        pretrained_report = load_pretrained_weights(
            model,
            args.pretrained_model_path,
            device,
            include_classifier=args.pretrained_include_classifier,
        )
        print(
            "Loaded pretrained weights: "
            f"{pretrained_report['loaded_key_count']} keys loaded, "
            f"{pretrained_report['skipped_key_count']} keys skipped."
        )

    configure_finetuning(
        model,
        freeze_encoder=args.freeze_encoder,
        freeze_attention=args.freeze_attention,
        freeze_residual=args.freeze_residual,
    )
    total_parameters, trainable_parameters = count_parameters(model)
    if trainable_parameters == 0:
        raise RuntimeError("No trainable parameters remain after applying freeze options.")

    # label smoothing 能降低模型在小数据集上的过度自信。
    train_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    # 验证集 loss 不使用 smoothing，这样数值仍然对应标准交叉熵。
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    # 余弦退火会逐渐降低学习率，减少训练后期的曲线震荡。
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": [], "ema_alpha": [args.ema_alpha]}
    best_acc = -1.0
    best_rows = []
    best_cm = None
    amp_enabled = bool(args.use_amp and device.type == "cuda")
    scaler = make_grad_scaler(enabled=amp_enabled)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_total = 0

        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp_enabled):
                logits, _, _ = model(x)
                loss = train_criterion(logits, y)
            scaler.scale(loss).backward()
            # 梯度裁剪限制偶发的大梯度，避免某个 batch 导致 loss 突然尖峰。
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * x.size(0)
            running_total += x.size(0)

        train_loss = running_loss / max(running_total, 1)
        metrics = evaluate(model, val_loader, eval_criterion, device, use_amp=amp_enabled)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(metrics["loss"])
        history["val_acc"].append(metrics["acc"])
        history["lr"].append(current_lr)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"lr={current_lr:.6f} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={metrics['loss']:.4f} | "
            f"val_acc={metrics['acc']:.4f}"
        )

        if metrics["acc"] > best_acc:
            best_acc = metrics["acc"]
            best_rows = metrics["rows"]
            best_cm = confusion_matrix(metrics["y_true"], metrics["y_pred"], labels=list(range(len(dataset.class_names))))
            torch.save(model.state_dict(), args.model_path)

        # 每个 epoch 的参数更新结束后再调整学习率，和日志中记录的 lr 保持一致。
        scheduler.step()

    plot_curves(history, args.artifact_dir / "training_curves.png")
    save_history_csv(history, args.artifact_dir / "training_history.csv")

    if best_cm is not None:
        plot_confusion(best_cm, dataset.class_names, args.artifact_dir / "best_confusion_matrix.png")

    save_attention_csv(best_rows, args.artifact_dir / "val_attention_weights.csv")

    summary = {
        "num_samples": len(dataset),
        "num_slices_per_object": dataset.num_slices,
        "effective_num_input_slices": dataset.effective_num_input_slices,
        "classes": dataset.class_names,
        "input_mode": args.input_mode,
        "single_gate_index": args.single_gate_index,
        "fusion_mode": args.fusion_mode,
        "gaussian_noise_std": args.gaussian_noise_std,
        "poisson_peak": args.poisson_peak,
        "background_scatter": args.background_scatter,
        "background_sigma": args.background_sigma,
        "gate_attenuation_index": args.gate_attenuation_index,
        "gate_attenuation_factor": args.gate_attenuation_factor,
        "gate_dropout_mode": args.gate_dropout_mode,
        "gate_dropout_index": args.gate_dropout_index,
        "structured_reflectance_strength": args.structured_reflectance_strength,
        "structured_background_strength": args.structured_background_strength,
        "structured_nuisance_grid_size": args.structured_nuisance_grid_size,
        "occlusion_probability": args.occlusion_probability,
        "occlusion_min_fraction": args.occlusion_min_fraction,
        "occlusion_max_fraction": args.occlusion_max_fraction,
        "occlusion_alpha": args.occlusion_alpha,
        "preserve_input_max": args.preserve_input_max,
        "degradation_probability": args.degradation_probability,
        "pretrained_model_path": str(args.pretrained_model_path) if args.pretrained_model_path is not None else "",
        "pretrained_include_classifier": args.pretrained_include_classifier,
        "pretrained_loaded_key_count": pretrained_report["loaded_key_count"] if pretrained_report else 0,
        "pretrained_skipped_key_count": pretrained_report["skipped_key_count"] if pretrained_report else 0,
        "pretrained_loaded_keys": pretrained_report["loaded_keys"] if pretrained_report else [],
        "pretrained_skipped_keys": pretrained_report["skipped_keys"] if pretrained_report else [],
        "freeze_encoder": args.freeze_encoder,
        "freeze_attention": args.freeze_attention,
        "freeze_residual": args.freeze_residual,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "best_val_acc": best_acc,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "use_amp": args.use_amp,
        "amp_enabled": amp_enabled,
        "cudnn_benchmark": args.cudnn_benchmark,
        "val_ratio": args.val_ratio,
        "split_group_by_sample_id": args.split_group_by_sample_id,
        "expected_num_slices": args.expected_num_slices,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "grad_clip": args.grad_clip,
        "ema_alpha": args.ema_alpha,
        "dataset_root": str(args.dataset_root),
        "artifact_dir": str(args.artifact_dir),
        "model_path": str(args.model_path),
    }
    with (args.artifact_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Training finished.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

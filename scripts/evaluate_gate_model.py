from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import MultiSliceObjectDataset
from model import FUSION_MODES, SliceAttentionClassifier
from train import (
    GATE_DROPOUT_MODES,
    SliceDegradationDataset,
    SliceInputViewDataset,
    build_class_dirs,
    collate_meta,
    evaluate,
    save_attention_csv,
    stratified_group_split,
    stratified_split,
    transform_gray,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained gated-slice classifier on a deterministic validation split, "
            "optionally applying test-time gate/image degradations."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--classes", nargs="+", default=["true3d", "flat_false"])
    parser.add_argument("--expected-num-slices", type=int, default=3)
    parser.add_argument("--input-mode", choices=["multi", "single-gate", "single-gate-black"], default="multi")
    parser.add_argument("--single-gate-index", type=int, default=0)
    parser.add_argument("--fusion-mode", choices=FUSION_MODES, default="attention")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--split-group-by-sample-id", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-amp", action="store_true")
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
    parser.add_argument("--degradation-probability", type=float, default=1.0)
    parser.add_argument("--predictions-csv", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser.parse_args()


def load_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            nested = checkpoint.get(key)
            if isinstance(nested, dict):
                return nested
        if all(isinstance(key, str) for key in checkpoint):
            return checkpoint
    raise ValueError(f"Checkpoint does not contain a state_dict: {path}")


def main() -> int:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    predictions_csv = args.predictions_csv or args.artifact_dir / "eval_predictions.csv"
    summary_json = args.summary_json or args.artifact_dir / "eval_summary.json"

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
        _, val_set = stratified_group_split(dataset, args.val_ratio, args.seed)
    else:
        _, val_set = stratified_split(dataset, args.val_ratio, args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.use_amp and device.type == "cuda")
    loader = DataLoader(
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
    model.load_state_dict(load_state_dict(args.model_path, device))

    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(model, loader, criterion, device, use_amp=amp_enabled)
    save_attention_csv(metrics["rows"], predictions_csv)

    summary = {
        "dataset_root": str(args.dataset_root),
        "model_path": str(args.model_path),
        "classes": dataset.class_names,
        "input_mode": args.input_mode,
        "single_gate_index": args.single_gate_index,
        "fusion_mode": args.fusion_mode,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "split_group_by_sample_id": args.split_group_by_sample_id,
        "num_val_samples": len(val_set),
        "loss": metrics["loss"],
        "acc": metrics["acc"],
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
        "predictions_csv": str(predictions_csv),
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

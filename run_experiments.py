from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


BASELINE_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = BASELINE_ROOT / "dataset"
DEFAULT_EXPERIMENT_ROOT = BASELINE_ROOT / "experiments"
DEFAULT_TRAIN_SCRIPT = BASELINE_ROOT / "train.py"
DEFAULT_CLASSES = ["chair", "desk", "sofa", "bed", "toilet", "image2d"]
DEFAULT_SEEDS = [42, 332, 2026]

RESULT_FIELDNAMES = [
    "experiment",
    "seed",
    "status",
    "best_val_acc",
    "num_samples",
    "num_slices_per_object",
    "effective_num_input_slices",
    "classes",
    "dataset_root",
    "artifact_dir",
    "model_path",
    "summary_path",
    "input_mode",
    "single_gate_index",
    "fusion_mode",
    "gaussian_noise_std",
    "poisson_peak",
    "background_scatter",
    "background_sigma",
    "gate_attenuation_index",
    "gate_attenuation_factor",
    "gate_dropout_mode",
    "gate_dropout_index",
    "structured_reflectance_strength",
    "structured_background_strength",
    "structured_nuisance_grid_size",
    "occlusion_probability",
    "occlusion_min_fraction",
    "occlusion_max_fraction",
    "occlusion_alpha",
    "preserve_input_max",
    "pretrained_model_path",
    "pretrained_include_classifier",
    "pretrained_loaded_key_count",
    "pretrained_skipped_key_count",
    "freeze_encoder",
    "freeze_attention",
    "freeze_residual",
    "total_parameters",
    "trainable_parameters",
    "epochs",
    "batch_size",
    "num_workers",
    "use_amp",
    "amp_enabled",
    "cudnn_benchmark",
    "lr",
    "min_lr",
    "weight_decay",
    "label_smoothing",
    "grad_clip",
    "ema_alpha",
    "val_ratio",
    "split_group_by_sample_id",
    "expected_num_slices",
]

AGGREGATE_FIELDNAMES = [
    "experiment",
    "num_runs",
    "mean_best_val_acc",
    "std_best_val_acc",
    "min_best_val_acc",
    "max_best_val_acc",
    "seeds",
]


@dataclass(frozen=True)
class ExperimentSpec:
    experiment: str
    seed: int
    artifact_dir: Path
    model_path: Path


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    slug = re.sub(r"_+", "_", slug).strip("._-")
    return slug or "experiment"


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run repeated slice-attention experiments and summarize each run's "
            "summary.json into experiments/results.csv."
        )
    )
    parser.add_argument("--experiment-name", default="baseline")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--train-script", type=Path, default=DEFAULT_TRAIN_SCRIPT)
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    parser.add_argument("--expected-num-slices", type=int, default=3)
    parser.add_argument("--input-mode", choices=["multi", "single-gate", "single-gate-black"], default="multi")
    parser.add_argument("--single-gate-index", type=int, default=0)
    parser.add_argument(
        "--fusion-mode",
        choices=["attention", "mean", "concat", "attention_residual"],
        default="attention",
    )
    parser.add_argument("--gaussian-noise-std", type=float, default=0.0)
    parser.add_argument("--poisson-peak", type=float, default=0.0)
    parser.add_argument("--background-scatter", type=float, default=0.0)
    parser.add_argument("--background-sigma", type=float, default=24.0)
    parser.add_argument("--gate-attenuation-index", type=int, default=-1)
    parser.add_argument("--gate-attenuation-factor", type=float, default=1.0)
    parser.add_argument("--gate-dropout-mode", choices=["none", "fixed", "random"], default="none")
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
    parser.add_argument("--pretrained-model-path", type=Path, default=None)
    parser.add_argument("--pretrained-include-classifier", action="store_true")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--freeze-attention", action="store_true")
    parser.add_argument("--freeze-residual", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-alpha", type=float, default=0.35)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--split-group-by-sample-id", action="store_true")
    parser.add_argument("--results-csv", type=Path, default=None)
    parser.add_argument("--aggregate-csv", type=Path, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")

    args, extra_train_args = parser.parse_known_args(argv)
    if extra_train_args[:1] == ["--"]:
        extra_train_args = extra_train_args[1:]
    return args, extra_train_args


def normalize_paths(args: argparse.Namespace) -> None:
    args.experiment_root = args.experiment_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.train_script = args.train_script.resolve()
    if args.results_csv is not None:
        args.results_csv = args.results_csv.resolve()
    if args.aggregate_csv is not None:
        args.aggregate_csv = args.aggregate_csv.resolve()
    if args.pretrained_model_path is not None:
        args.pretrained_model_path = args.pretrained_model_path.resolve()


def build_experiment_specs(args: argparse.Namespace) -> list[ExperimentSpec]:
    experiment_slug = slugify(args.experiment_name)
    specs = []
    for seed in args.seeds:
        run_name = f"{experiment_slug}_seed{seed}"
        artifact_dir = args.experiment_root / run_name
        specs.append(
            ExperimentSpec(
                experiment=args.experiment_name,
                seed=seed,
                artifact_dir=artifact_dir,
                model_path=artifact_dir / "slice_attention_model.pth",
            )
        )
    return specs


def build_train_command(
    args: argparse.Namespace,
    spec: ExperimentSpec,
    extra_train_args: Sequence[str],
) -> list[str]:
    return [
        sys.executable,
        str(args.train_script),
        "--dataset-root",
        str(args.dataset_root),
        "--artifact-dir",
        str(spec.artifact_dir),
        "--model-path",
        str(spec.model_path),
        "--classes",
        *args.classes,
        "--expected-num-slices",
        str(args.expected_num_slices),
        "--input-mode",
        str(args.input_mode),
        "--single-gate-index",
        str(args.single_gate_index),
        "--fusion-mode",
        str(args.fusion_mode),
        "--gaussian-noise-std",
        str(args.gaussian_noise_std),
        "--poisson-peak",
        str(args.poisson_peak),
        "--background-scatter",
        str(args.background_scatter),
        "--background-sigma",
        str(args.background_sigma),
        "--gate-attenuation-index",
        str(args.gate_attenuation_index),
        "--gate-attenuation-factor",
        str(args.gate_attenuation_factor),
        "--gate-dropout-mode",
        str(args.gate_dropout_mode),
        "--gate-dropout-index",
        str(args.gate_dropout_index),
        "--structured-reflectance-strength",
        str(args.structured_reflectance_strength),
        "--structured-background-strength",
        str(args.structured_background_strength),
        "--structured-nuisance-grid-size",
        str(args.structured_nuisance_grid_size),
        "--occlusion-probability",
        str(args.occlusion_probability),
        "--occlusion-min-fraction",
        str(args.occlusion_min_fraction),
        "--occlusion-max-fraction",
        str(args.occlusion_max_fraction),
        "--occlusion-alpha",
        str(args.occlusion_alpha),
        *(["--preserve-input-max"] if args.preserve_input_max else []),
        "--degradation-probability",
        str(args.degradation_probability),
        *(
            ["--pretrained-model-path", str(args.pretrained_model_path)]
            if args.pretrained_model_path is not None
            else []
        ),
        *(["--pretrained-include-classifier"] if args.pretrained_include_classifier else []),
        *(["--freeze-encoder"] if args.freeze_encoder else []),
        *(["--freeze-attention"] if args.freeze_attention else []),
        *(["--freeze-residual"] if args.freeze_residual else []),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        *(["--use-amp"] if args.use_amp else []),
        *(["--cudnn-benchmark"] if args.cudnn_benchmark else []),
        "--lr",
        str(args.lr),
        "--min-lr",
        str(args.min_lr),
        "--weight-decay",
        str(args.weight_decay),
        "--label-smoothing",
        str(args.label_smoothing),
        "--grad-clip",
        str(args.grad_clip),
        "--ema-alpha",
        str(args.ema_alpha),
        "--val-ratio",
        str(args.val_ratio),
        *(["--split-group-by-sample-id"] if args.split_group_by_sample_id else []),
        "--seed",
        str(spec.seed),
        *extra_train_args,
    ]


def load_summary(summary_path: Path) -> dict:
    if not summary_path.exists():
        raise FileNotFoundError(f"Expected training summary was not created: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_result_row(
    args: argparse.Namespace,
    spec: ExperimentSpec,
    summary: dict,
    status: str = "ok",
) -> dict[str, object]:
    summary_path = spec.artifact_dir / "summary.json"
    return {
        "experiment": spec.experiment,
        "seed": summary.get("seed", spec.seed),
        "status": status,
        "best_val_acc": summary.get("best_val_acc", ""),
        "num_samples": summary.get("num_samples", ""),
        "num_slices_per_object": summary.get("num_slices_per_object", ""),
        "effective_num_input_slices": summary.get("effective_num_input_slices", ""),
        "classes": " ".join(summary.get("classes", args.classes)),
        "dataset_root": summary.get("dataset_root", str(args.dataset_root)),
        "artifact_dir": summary.get("artifact_dir", str(spec.artifact_dir)),
        "model_path": summary.get("model_path", str(spec.model_path)),
        "summary_path": str(summary_path),
        "input_mode": summary.get("input_mode", args.input_mode),
        "single_gate_index": summary.get("single_gate_index", args.single_gate_index),
        "fusion_mode": summary.get("fusion_mode", args.fusion_mode),
        "gaussian_noise_std": summary.get("gaussian_noise_std", args.gaussian_noise_std),
        "poisson_peak": summary.get("poisson_peak", args.poisson_peak),
        "background_scatter": summary.get("background_scatter", args.background_scatter),
        "background_sigma": summary.get("background_sigma", args.background_sigma),
        "gate_attenuation_index": summary.get("gate_attenuation_index", args.gate_attenuation_index),
        "gate_attenuation_factor": summary.get("gate_attenuation_factor", args.gate_attenuation_factor),
        "gate_dropout_mode": summary.get("gate_dropout_mode", args.gate_dropout_mode),
        "gate_dropout_index": summary.get("gate_dropout_index", args.gate_dropout_index),
        "structured_reflectance_strength": summary.get(
            "structured_reflectance_strength", args.structured_reflectance_strength
        ),
        "structured_background_strength": summary.get(
            "structured_background_strength", args.structured_background_strength
        ),
        "structured_nuisance_grid_size": summary.get(
            "structured_nuisance_grid_size", args.structured_nuisance_grid_size
        ),
        "occlusion_probability": summary.get("occlusion_probability", args.occlusion_probability),
        "occlusion_min_fraction": summary.get("occlusion_min_fraction", args.occlusion_min_fraction),
        "occlusion_max_fraction": summary.get("occlusion_max_fraction", args.occlusion_max_fraction),
        "occlusion_alpha": summary.get("occlusion_alpha", args.occlusion_alpha),
        "preserve_input_max": summary.get("preserve_input_max", args.preserve_input_max),
        "pretrained_model_path": summary.get(
            "pretrained_model_path",
            str(args.pretrained_model_path) if args.pretrained_model_path is not None else "",
        ),
        "pretrained_include_classifier": summary.get(
            "pretrained_include_classifier",
            args.pretrained_include_classifier,
        ),
        "pretrained_loaded_key_count": summary.get("pretrained_loaded_key_count", ""),
        "pretrained_skipped_key_count": summary.get("pretrained_skipped_key_count", ""),
        "freeze_encoder": summary.get("freeze_encoder", args.freeze_encoder),
        "freeze_attention": summary.get("freeze_attention", args.freeze_attention),
        "freeze_residual": summary.get("freeze_residual", args.freeze_residual),
        "total_parameters": summary.get("total_parameters", ""),
        "trainable_parameters": summary.get("trainable_parameters", ""),
        "epochs": summary.get("epochs", args.epochs),
        "batch_size": summary.get("batch_size", args.batch_size),
        "num_workers": summary.get("num_workers", args.num_workers),
        "use_amp": summary.get("use_amp", args.use_amp),
        "amp_enabled": summary.get("amp_enabled", ""),
        "cudnn_benchmark": summary.get("cudnn_benchmark", args.cudnn_benchmark),
        "lr": summary.get("lr", args.lr),
        "min_lr": summary.get("min_lr", args.min_lr),
        "weight_decay": summary.get("weight_decay", args.weight_decay),
        "label_smoothing": summary.get("label_smoothing", args.label_smoothing),
        "grad_clip": summary.get("grad_clip", args.grad_clip),
        "ema_alpha": summary.get("ema_alpha", args.ema_alpha),
        "val_ratio": summary.get("val_ratio", args.val_ratio),
        "split_group_by_sample_id": summary.get("split_group_by_sample_id", args.split_group_by_sample_id),
        "expected_num_slices": summary.get("expected_num_slices", args.expected_num_slices),
    }


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_aggregate_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        if row.get("status") != "ok" or row.get("best_val_acc") in ("", None):
            continue
        grouped.setdefault(str(row["experiment"]), []).append(row)

    aggregate_rows = []
    for experiment, experiment_rows in sorted(grouped.items()):
        accs = [float(row["best_val_acc"]) for row in experiment_rows]
        seeds = [str(row["seed"]) for row in experiment_rows]
        aggregate_rows.append(
            {
                "experiment": experiment,
                "num_runs": len(accs),
                "mean_best_val_acc": statistics.mean(accs),
                "std_best_val_acc": statistics.stdev(accs) if len(accs) > 1 else 0.0,
                "min_best_val_acc": min(accs),
                "max_best_val_acc": max(accs),
                "seeds": " ".join(seeds),
            }
        )
    return aggregate_rows


def run_training(command: Sequence[str], cwd: Path) -> None:
    subprocess.run(command, cwd=str(cwd), check=True)


def main(argv: Sequence[str] | None = None) -> int:
    args, extra_train_args = parse_args(argv)
    normalize_paths(args)
    args.experiment_root.mkdir(parents=True, exist_ok=True)
    results_csv = args.results_csv or args.experiment_root / "results.csv"
    aggregate_csv = args.aggregate_csv or args.experiment_root / "aggregate_results.csv"
    specs = build_experiment_specs(args)
    rows: list[dict[str, object]] = []

    for spec in specs:
        summary_path = spec.artifact_dir / "summary.json"
        command = build_train_command(args, spec, extra_train_args)
        print(f"\n=== {spec.experiment} seed={spec.seed} ===", flush=True)
        print(" ".join(command), flush=True)

        if args.dry_run:
            continue

        try:
            if args.skip_existing and summary_path.exists():
                print(f"Skipping existing run: {summary_path}")
            else:
                spec.artifact_dir.mkdir(parents=True, exist_ok=True)
                run_training(command, cwd=args.train_script.parent)

            summary = load_summary(summary_path)
            rows.append(make_result_row(args, spec, summary))
            write_csv(results_csv, RESULT_FIELDNAMES, rows)
            write_csv(aggregate_csv, AGGREGATE_FIELDNAMES, build_aggregate_rows(rows))
        except Exception as exc:
            if not args.keep_going:
                raise
            print(f"Run failed: {exc}", file=sys.stderr)
            failed_summary = {
                "seed": spec.seed,
                "classes": args.classes,
                "dataset_root": str(args.dataset_root),
                "artifact_dir": str(spec.artifact_dir),
                "model_path": str(spec.model_path),
            }
            rows.append(make_result_row(args, spec, failed_summary, status="failed"))
            write_csv(results_csv, RESULT_FIELDNAMES, rows)

    if not args.dry_run:
        write_csv(results_csv, RESULT_FIELDNAMES, rows)
        write_csv(aggregate_csv, AGGREGATE_FIELDNAMES, build_aggregate_rows(rows))
        print(f"\nWrote per-run results: {results_csv}")
        print(f"Wrote aggregate results: {aggregate_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

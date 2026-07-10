from pathlib import Path

import pytest

from run_experiments import (
    build_aggregate_rows,
    build_experiment_specs,
    build_train_command,
    normalize_paths,
    parse_args,
    slugify,
)


def test_slugify_keeps_experiment_names_filesystem_friendly() -> None:
    assert slugify("baseline gate width=1.0") == "baseline_gate_width_1.0"
    assert slugify("???") == "experiment"


def test_build_experiment_specs_creates_seed_specific_paths(tmp_path: Path) -> None:
    args, _ = parse_args(
        [
            "--experiment-name",
            "baseline",
            "--seeds",
            "42",
            "123",
            "--experiment-root",
            str(tmp_path),
        ]
    )

    specs = build_experiment_specs(args)

    assert [spec.seed for spec in specs] == [42, 123]
    assert specs[0].artifact_dir == tmp_path / "baseline_seed42"
    assert specs[0].model_path == tmp_path / "baseline_seed42" / "slice_attention_model.pth"


def test_build_train_command_passes_run_specific_outputs(tmp_path: Path) -> None:
    args, extra_args = parse_args(
        [
            "--experiment-root",
            str(tmp_path),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--train-script",
            str(tmp_path / "train.py"),
            "--seeds",
            "42",
            "--input-mode",
            "single-gate-black",
            "--single-gate-index",
            "2",
            "--fusion-mode",
            "mean",
            "--gaussian-noise-std",
            "0.05",
            "--gate-dropout-mode",
            "fixed",
            "--gate-dropout-index",
            "1",
            "--structured-reflectance-strength",
            "0.08",
            "--structured-background-strength",
            "0.02",
            "--occlusion-probability",
            "0.4",
            "--preserve-input-max",
            "--pretrained-model-path",
            str(tmp_path / "pretrained.pth"),
            "--freeze-encoder",
            "--freeze-attention",
            "--split-group-by-sample-id",
            "--epochs",
            "2",
            "--batch-size",
            "16",
            "--num-workers",
            "4",
            "--use-amp",
            "--cudnn-benchmark",
            "--",
            "--some-future-train-flag",
            "value",
        ]
    )
    spec = build_experiment_specs(args)[0]

    command = build_train_command(args, spec, extra_args)

    assert "--seed" in command
    assert command[command.index("--seed") + 1] == "42"
    assert "--artifact-dir" in command
    assert command[command.index("--artifact-dir") + 1] == str(spec.artifact_dir)
    assert "--model-path" in command
    assert command[command.index("--model-path") + 1] == str(spec.model_path)
    assert "--input-mode" in command
    assert command[command.index("--input-mode") + 1] == "single-gate-black"
    assert "--single-gate-index" in command
    assert command[command.index("--single-gate-index") + 1] == "2"
    assert "--fusion-mode" in command
    assert command[command.index("--fusion-mode") + 1] == "mean"
    assert "--gaussian-noise-std" in command
    assert command[command.index("--gaussian-noise-std") + 1] == "0.05"
    assert "--gate-dropout-mode" in command
    assert command[command.index("--gate-dropout-mode") + 1] == "fixed"
    assert "--gate-dropout-index" in command
    assert command[command.index("--gate-dropout-index") + 1] == "1"
    assert "--structured-reflectance-strength" in command
    assert command[command.index("--structured-reflectance-strength") + 1] == "0.08"
    assert "--structured-background-strength" in command
    assert command[command.index("--structured-background-strength") + 1] == "0.02"
    assert "--occlusion-probability" in command
    assert command[command.index("--occlusion-probability") + 1] == "0.4"
    assert "--preserve-input-max" in command
    assert "--pretrained-model-path" in command
    assert command[command.index("--pretrained-model-path") + 1] == str(tmp_path / "pretrained.pth")
    assert "--freeze-encoder" in command
    assert "--freeze-attention" in command
    assert "--freeze-residual" not in command
    assert "--split-group-by-sample-id" in command
    assert "--batch-size" in command
    assert command[command.index("--batch-size") + 1] == "16"
    assert "--num-workers" in command
    assert command[command.index("--num-workers") + 1] == "4"
    assert "--use-amp" in command
    assert "--cudnn-benchmark" in command
    assert command[-2:] == ["--some-future-train-flag", "value"]


def test_normalize_paths_resolves_runner_paths(tmp_path: Path) -> None:
    args, _ = parse_args(
        [
            "--experiment-root",
            str(tmp_path / "experiments"),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--train-script",
            str(tmp_path / "train.py"),
            "--results-csv",
            str(tmp_path / "experiments" / "results.csv"),
        ]
    )

    normalize_paths(args)

    assert args.experiment_root.is_absolute()
    assert args.dataset_root.is_absolute()
    assert args.train_script.is_absolute()
    assert args.results_csv.is_absolute()


def test_build_aggregate_rows_summarizes_successful_runs_only() -> None:
    rows = [
        {"experiment": "baseline", "seed": 42, "status": "ok", "best_val_acc": 0.7},
        {"experiment": "baseline", "seed": 123, "status": "ok", "best_val_acc": 0.9},
        {"experiment": "baseline", "seed": 2025, "status": "failed", "best_val_acc": ""},
    ]

    aggregate_rows = build_aggregate_rows(rows)

    assert len(aggregate_rows) == 1
    aggregate = aggregate_rows[0]
    assert aggregate["experiment"] == "baseline"
    assert aggregate["num_runs"] == 2
    assert aggregate["mean_best_val_acc"] == pytest.approx(0.8)
    assert aggregate["std_best_val_acc"] == pytest.approx(0.14142135623730953)
    assert aggregate["seeds"] == "42 123"

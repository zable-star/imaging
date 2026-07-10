from __future__ import annotations

import csv
from pathlib import Path

from dataset_new import build_multiview_true_false_dataset as builder


def write_gate_stack(root: Path, view: str, class_name: str, sample_id: str, gates: int = 3) -> None:
    class_dir = root / view / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    for gate in range(gates):
        (class_dir / f"{sample_id}_gate_{gate}.png").write_bytes(b"png")


def test_multiview_builder_keeps_same_model_views_as_distinct_samples(tmp_path: Path) -> None:
    true_root = tmp_path / "true"
    false_root = tmp_path / "false"
    out_root = tmp_path / "combined"

    for view in ["view_z000", "view_z090"]:
        write_gate_stack(true_root, view, "tank", "same_model")
        write_gate_stack(false_root, view, "tank", "same_model")

    args = builder.parse_args(
        [
            "--true-root",
            str(true_root),
            "--false-root",
            str(false_root),
            "--output-root",
            str(out_root),
            "--expected-views",
            "2",
        ]
    )
    true_samples = builder.collect_samples(args.true_root, "true3d", args.expected_num_slices, args.expected_views)
    false_samples = builder.collect_samples(args.false_root, "flat_false", args.expected_num_slices, args.expected_views)
    builder.prepare_output_root(args.output_root, args.overwrite)
    rows = []
    rows.extend(builder.copy_samples(true_samples, args.true_class_name, args.output_root))
    rows.extend(builder.copy_samples(false_samples, args.false_class_name, args.output_root))
    builder.write_manifest(args.output_root, rows)

    true_pngs = sorted((out_root / "true3d").glob("*.png"))
    false_pngs = sorted((out_root / "flat_false").glob("*.png"))
    assert len(true_pngs) == 6
    assert len(false_pngs) == 6
    assert any("view_z000" in path.name for path in true_pngs)
    assert any("view_z090" in path.name for path in true_pngs)

    with (out_root / "binary_manifest.csv").open(newline="", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))
    assert len(manifest_rows) == 4
    assert sorted({row["view_id"] for row in manifest_rows}) == ["view_z000", "view_z090"]


def test_discover_view_dirs_accepts_single_view_root(tmp_path: Path) -> None:
    root = tmp_path / "single"
    class_dir = root / "tank"
    class_dir.mkdir(parents=True)
    for gate in range(3):
        (class_dir / f"model_gate_{gate}.png").write_bytes(b"png")

    assert builder.discover_view_dirs(root) == [("view0", root)]

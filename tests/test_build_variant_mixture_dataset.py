from __future__ import annotations

import csv
from pathlib import Path

from dataset_new import build_variant_mixture_dataset as builder


def write_gate_stack(root: Path, class_name: str, sample_id: str, gates: int = 3) -> None:
    class_dir = root / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    for gate in range(gates):
        (class_dir / f"{sample_id}_gate_{gate}.png").write_bytes(f"{sample_id}:{gate}".encode("ascii"))


def test_variant_mixture_builder_prefixes_domains_and_writes_manifest(tmp_path: Path) -> None:
    norm_root = tmp_path / "norm"
    hard_root = tmp_path / "hard"
    out_root = tmp_path / "mixed"

    for source_root in [norm_root, hard_root]:
        write_gate_stack(source_root, "true3d", "view_z000__source_001")
        write_gate_stack(source_root, "flat_false", "view_z000__source_001")

    args = builder.parse_args(
        [
            "--sources",
            f"norm={norm_root}",
            f"hardv3={hard_root}",
            "--output-root",
            str(out_root),
            "--overwrite",
        ]
    )
    rows = builder.build_dataset(args)

    assert len(rows) == 4
    true_pngs = sorted((out_root / "true3d").glob("*.png"))
    false_pngs = sorted((out_root / "flat_false").glob("*.png"))
    assert len(true_pngs) == 6
    assert len(false_pngs) == 6
    assert any(path.name.startswith("domain_norm__view_z000__source_001") for path in true_pngs)
    assert any(path.name.startswith("domain_hardv3__view_z000__source_001") for path in true_pngs)

    with (out_root / "variant_mixture_manifest.csv").open(newline="", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))
    assert sorted({row["domain"] for row in manifest_rows}) == ["hardv3", "norm"]
    assert sorted({row["mixed_sample_id"] for row in manifest_rows}) == [
        "domain_hardv3__view_z000__source_001",
        "domain_norm__view_z000__source_001",
    ]

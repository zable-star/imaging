from __future__ import annotations

import argparse
import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


GATE_PATTERN = re.compile(r"^(?P<base>.+?)_gate_(?P<gate>\d+)\.png$")
TAG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass(frozen=True)
class MixtureRow:
    domain: str
    class_name: str
    source_sample_id: str
    mixed_sample_id: str
    images_written: int
    source_paths: str


def parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Sources must use the form tag=path")
    tag, raw_path = value.split("=", 1)
    tag = tag.strip()
    if not TAG_PATTERN.match(tag):
        raise argparse.ArgumentTypeError(f"Invalid source tag: {tag!r}")
    path = Path(raw_path)
    return tag, path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine multiple gated-image dataset variants into one domain-prefixed dataset."
    )
    parser.add_argument("--sources", nargs="+", type=parse_source, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--classes", nargs="+", default=["true3d", "flat_false"])
    parser.add_argument("--expected-num-slices", type=int, default=3)
    parser.add_argument("--manifest-out", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def prepare_output_root(output_root: Path, overwrite: bool) -> None:
    resolved = output_root.resolve()
    if overwrite and resolved.exists():
        if len(resolved.parts) < 4:
            raise ValueError(f"Refusing to remove shallow output path: {resolved}")
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def group_gate_images(class_dir: Path) -> dict[str, dict[int, Path]]:
    grouped: dict[str, dict[int, Path]] = {}
    for path in sorted(class_dir.rglob("*.png")):
        match = GATE_PATTERN.match(path.name)
        if not match:
            continue
        grouped.setdefault(match.group("base"), {})[int(match.group("gate"))] = path
    return grouped


def copy_variant(
    domain: str,
    source_root: Path,
    output_root: Path,
    classes: Sequence[str],
    expected_num_slices: int,
) -> list[MixtureRow]:
    rows: list[MixtureRow] = []
    for class_name in classes:
        class_dir = source_root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        output_class_dir = output_root / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        grouped = group_gate_images(class_dir)
        for sample_id, paths in sorted(grouped.items()):
            if len(paths) != expected_num_slices:
                continue
            mixed_sample_id = f"domain_{domain}__{sample_id}"
            written_paths: list[str] = []
            for gate_idx in range(expected_num_slices):
                src = paths[gate_idx]
                dst = output_class_dir / f"{mixed_sample_id}_gate_{gate_idx}.png"
                if dst.exists():
                    raise FileExistsError(f"Duplicate mixed sample path: {dst}")
                shutil.copy2(src, dst)
                written_paths.append(str(src))
            rows.append(
                MixtureRow(
                    domain=domain,
                    class_name=class_name,
                    source_sample_id=sample_id,
                    mixed_sample_id=mixed_sample_id,
                    images_written=expected_num_slices,
                    source_paths=";".join(written_paths),
                )
            )
    return rows


def write_manifest(path: Path, rows: Sequence[MixtureRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(MixtureRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def build_dataset(args: argparse.Namespace) -> list[MixtureRow]:
    prepare_output_root(args.output_root, args.overwrite)
    rows: list[MixtureRow] = []
    seen_domains: set[str] = set()
    for domain, source_root in args.sources:
        if domain in seen_domains:
            raise ValueError(f"Duplicate domain tag: {domain}")
        seen_domains.add(domain)
        rows.extend(copy_variant(domain, source_root, args.output_root, args.classes, args.expected_num_slices))

    manifest = args.manifest_out or args.output_root / "variant_mixture_manifest.csv"
    write_manifest(manifest, rows)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = build_dataset(args)
    print(f"domains={len(args.sources)}")
    print(f"samples={len(rows)}")
    print(f"gate_images={sum(row.images_written for row in rows)}")
    print(f"output_root={args.output_root}")
    print(f"manifest={args.manifest_out or args.output_root / 'variant_mixture_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

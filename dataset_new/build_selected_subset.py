from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_KEEP_VALUES = {"1", "true", "yes", "y", "keep", "selected"}


@dataclass(frozen=True)
class SelectedModel:
    category: str
    source_file: Path
    output_file: Path
    name: str = ""
    uid: str = ""
    reason: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a clean selected-model subset from a manual review CSV. "
            "Use --expected-count 44 to prevent accidentally copying all candidates."
        )
    )
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--keep-column",
        default="keep",
        help="Column used to decide whether a row is selected.",
    )
    parser.add_argument(
        "--file-column",
        default="file",
        help="Column containing the source model file path.",
    )
    parser.add_argument(
        "--category-column",
        default="category",
        help="Column containing the class/category name.",
    )
    parser.add_argument(
        "--keep-values",
        nargs="+",
        default=sorted(DEFAULT_KEEP_VALUES),
        help="Accepted selected values in the keep column.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=0,
        help="If positive, fail unless exactly this many selected rows are found.",
    )
    parser.add_argument("--manifest-csv", type=Path, default=None)
    parser.add_argument("--manifest-json", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_keep(value: object) -> str:
    return str(value or "").strip().lower()


def selected_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    keep_values = {normalize_keep(value) for value in args.keep_values}
    with args.review_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    missing_columns = [
        column
        for column in (args.keep_column, args.file_column, args.category_column)
        if not rows or column not in rows[0]
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns in review CSV: {missing_columns}")

    return [row for row in rows if normalize_keep(row.get(args.keep_column)) in keep_values]


def unique_output_path(output_dir: Path, source_file: Path, used: set[Path]) -> Path:
    stem = source_file.stem
    suffix = source_file.suffix
    candidate = output_dir / source_file.name
    index = 2
    while candidate in used:
        candidate = output_dir / f"{stem}_{index}{suffix}"
        index += 1
    used.add(candidate)
    return candidate


def plan_copy(args: argparse.Namespace, rows: list[dict[str, str]]) -> list[SelectedModel]:
    used_outputs: set[Path] = set()
    selected: list[SelectedModel] = []
    for row in rows:
        source_file = Path(row[args.file_column])
        if not source_file.exists():
            raise FileNotFoundError(f"Selected source file does not exist: {source_file}")
        category = row[args.category_column].strip()
        if not category:
            raise ValueError(f"Selected row has empty category: {source_file}")
        output_dir = args.output_root / category
        output_file = unique_output_path(output_dir, source_file, used_outputs)
        selected.append(
            SelectedModel(
                category=category,
                source_file=source_file,
                output_file=output_file,
                name=row.get("name", ""),
                uid=row.get("uid", ""),
                reason=row.get("reason", ""),
            )
        )
    return selected


def write_manifest_csv(path: Path, selected: list[SelectedModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["category", "source_file", "output_file", "name", "uid", "reason"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in selected:
            row = asdict(item)
            row["source_file"] = str(item.source_file)
            row["output_file"] = str(item.output_file)
            writer.writerow(row)


def write_manifest_json(path: Path, selected: list[SelectedModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            **asdict(item),
            "source_file": str(item.source_file),
            "output_file": str(item.output_file),
        }
        for item in selected
    ]
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def copy_selected(selected: list[SelectedModel], overwrite: bool) -> None:
    for item in selected:
        item.output_file.parent.mkdir(parents=True, exist_ok=True)
        if item.output_file.exists() and not overwrite:
            raise FileExistsError(f"Output file already exists: {item.output_file}")
        shutil.copy2(item.source_file, item.output_file)


def print_summary(selected: list[SelectedModel]) -> None:
    counts: dict[str, int] = {}
    for item in selected:
        counts[item.category] = counts.get(item.category, 0) + 1
    print(f"Selected models: {len(selected)}")
    for category in sorted(counts):
        print(f"  {category}: {counts[category]}")


def main() -> int:
    args = parse_args()
    rows = selected_rows(args)
    if args.expected_count > 0 and len(rows) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} selected rows, found {len(rows)}. "
            "Refusing to build subset to avoid using the wrong military candidates."
        )

    selected = plan_copy(args, rows)
    print_summary(selected)
    if args.dry_run:
        print("Dry run only; no files copied.")
        return 0

    copy_selected(selected, overwrite=args.overwrite)
    manifest_csv = args.manifest_csv or args.output_root / "selected_manifest.csv"
    manifest_json = args.manifest_json or args.output_root / "selected_manifest.json"
    write_manifest_csv(manifest_csv, selected)
    write_manifest_json(manifest_json, selected)
    print(f"Wrote selected subset: {args.output_root}")
    print(f"Wrote manifest CSV: {manifest_csv}")
    print(f"Wrote manifest JSON: {manifest_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

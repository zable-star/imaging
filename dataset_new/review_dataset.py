"""Create and apply a manual review table for the Objaverse subset."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "Military_3D_Dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review copied Objaverse dataset files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Dataset directory with one folder per class.",
    )
    parser.add_argument(
        "--candidate-review",
        type=Path,
        default=None,
        help="CSV produced by generate.py. Defaults to <output-dir>/candidate_review.csv.",
    )
    parser.add_argument(
        "--review-table",
        type=Path,
        default=None,
        help="Manual review CSV. Defaults to <output-dir>/manual_review.csv.",
    )
    parser.add_argument(
        "--reject-uids",
        type=Path,
        default=None,
        help="Rejected UID text file. Defaults to <output-dir>/rejected_uids.txt.",
    )
    parser.add_argument(
        "--rejected-dir",
        type=Path,
        default=None,
        help="Directory where rejected files are moved. Defaults to <output-dir>/_rejected.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Move files whose keep column is 0/no/false/reject and record their UIDs.",
    )
    return parser.parse_args()


def uid_prefix_from_name(filename: str) -> str | None:
    stem = Path(filename).stem
    parts = stem.rsplit("_", 1)
    if len(parts) != 2:
        return None
    prefix = parts[1].lower()
    if len(prefix) == 8 and all(char in "0123456789abcdef" for char in prefix):
        return prefix
    return None


def load_metadata(candidate_review_path: Path) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    if not candidate_review_path.exists():
        return metadata
    with candidate_review_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            uid = row.get("uid", "")
            if uid:
                metadata[uid] = row
    return metadata


def iter_dataset_files(output_dir: Path):
    for category_dir in sorted(path for path in output_dir.iterdir() if path.is_dir()):
        if category_dir.name.startswith("_"):
            continue
        for model_path in sorted(path for path in category_dir.iterdir() if path.is_file()):
            yield category_dir.name, model_path


def make_review_table(output_dir: Path, candidate_review_path: Path, review_table_path: Path) -> None:
    metadata = load_metadata(candidate_review_path)
    rows: list[dict[str, str]] = []

    for category, model_path in iter_dataset_files(output_dir):
        prefix = uid_prefix_from_name(model_path.name)
        matches = [uid for uid in metadata if prefix and uid.startswith(prefix)]
        uid = matches[0] if matches else ""
        meta = metadata.get(uid, {})
        rows.append(
            {
                "keep": "1",
                "category": category,
                "file": str(model_path),
                "uid": uid,
                "name": meta.get("name", ""),
                "tags": meta.get("tags", ""),
                "categories": meta.get("categories", ""),
                "description": meta.get("description", ""),
                "reason": "",
            }
        )

    review_table_path.parent.mkdir(parents=True, exist_ok=True)
    with review_table_path.open("w", encoding="utf-8-sig", newline="") as file:
        fieldnames = ["keep", "category", "file", "uid", "name", "tags", "categories", "description", "reason"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[review] Wrote {len(rows)} rows to {review_table_path}")


def load_existing_rejections(reject_uids_path: Path) -> set[str]:
    if not reject_uids_path.exists():
        return set()
    with reject_uids_path.open("r", encoding="utf-8") as file:
        return {line.strip() for line in file if line.strip() and not line.startswith("#")}


def apply_review(review_table_path: Path, reject_uids_path: Path, rejected_dir: Path) -> None:
    rejected_uids = load_existing_rejections(reject_uids_path)
    moved = 0

    with review_table_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            keep = row.get("keep", "").strip().lower()
            if keep not in {"0", "no", "n", "false", "reject", "bad"}:
                continue

            uid = row.get("uid", "").strip()
            if uid:
                rejected_uids.add(uid)

            model_path = Path(row.get("file", ""))
            if not model_path.exists():
                continue

            category = row.get("category", model_path.parent.name)
            target_dir = rejected_dir / category
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / model_path.name
            if target_path.exists():
                target_path = target_dir / f"{model_path.stem}_{moved + 1}{model_path.suffix}"
            shutil.move(str(model_path), str(target_path))
            moved += 1

    reject_uids_path.parent.mkdir(parents=True, exist_ok=True)
    with reject_uids_path.open("w", encoding="utf-8") as file:
        for uid in sorted(rejected_uids):
            file.write(f"{uid}\n")

    print(f"[review] Moved {moved} rejected files to {rejected_dir}")
    print(f"[review] Wrote {len(rejected_uids)} rejected UIDs to {reject_uids_path}")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    candidate_review_path = (
        args.candidate_review.resolve()
        if args.candidate_review is not None
        else output_dir / "candidate_review.csv"
    )
    review_table_path = (
        args.review_table.resolve()
        if args.review_table is not None
        else output_dir / "manual_review.csv"
    )
    reject_uids_path = (
        args.reject_uids.resolve()
        if args.reject_uids is not None
        else output_dir / "rejected_uids.txt"
    )
    rejected_dir = (
        args.rejected_dir.resolve()
        if args.rejected_dir is not None
        else output_dir / "_rejected"
    )

    if args.apply:
        apply_review(review_table_path, reject_uids_path, rejected_dir)
    else:
        make_review_table(output_dir, candidate_review_path, review_table_path)


if __name__ == "__main__":
    main()

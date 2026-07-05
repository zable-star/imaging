"""Download a small military-object subset from Objaverse 1.0.

The script searches Objaverse annotations with category-specific keywords, then
downloads and copies the selected objects into one folder per class.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CategorySpec:
    strong_keywords: tuple[str, ...]
    weak_keywords: tuple[str, ...] = ()
    context_keywords: tuple[str, ...] = ()
    negative_keywords: tuple[str, ...] = ()


CATEGORIES: dict[str, CategorySpec] = {
    "01_Main_Battle_Tank": CategorySpec(
        strong_keywords=(
            "main battle tank",
            "battle tank",
            "military tank",
            "m1a2",
            "abrams",
            "t-72",
            "t72",
            "t-90",
            "leopard 2",
            "merkava",
        ),
        weak_keywords=("tank",),
        context_keywords=("army", "military", "tracked", "turret", "cannon", "armored", "armoured"),
        negative_keywords=(
            "aquarium",
            "barrel",
            "container",
            "fish tank",
            "fuel tank",
            "gas tank",
            "oil tank",
            "septic tank",
            "storage tank",
            "water tank",
        ),
    ),
    "02_Fighter_Jet": CategorySpec(
        strong_keywords=(
            "fighter jet",
            "fighter aircraft",
            "military aircraft",
            "f-15",
            "f15",
            "f-16",
            "f16",
            "f-18",
            "f18",
            "f-22",
            "f22",
            "f-35",
            "f35",
            "su-27",
            "su27",
            "mig-29",
            "mig29",
        ),
        weak_keywords=("jet", "aircraft", "plane"),
        context_keywords=("fighter", "military", "air force", "weapon", "missile"),
        negative_keywords=("airliner", "boeing", "cessna", "cartoon", "paper airplane", "toy"),
    ),
    "03_Attack_Helicopter": CategorySpec(
        strong_keywords=("attack helicopter", "apache", "ah-64", "ah64", "mi-24", "mi24", "ka-52", "ka52"),
        weak_keywords=("helicopter",),
        context_keywords=("attack", "military", "gunship", "army", "weapon", "missile"),
        negative_keywords=("toy", "cartoon", "civilian", "rescue", "police"),
    ),
    "04_Armored_Vehicle": CategorySpec(
        strong_keywords=(
            "armored vehicle",
            "armoured vehicle",
            "armored personnel carrier",
            "armoured personnel carrier",
            "apc",
            "ifv",
            "stryker",
            "btr",
            "brdm",
            "mrap",
        ),
        weak_keywords=("armored", "armoured"),
        context_keywords=("vehicle", "military", "army", "wheeled", "personnel carrier"),
        negative_keywords=("fantasy", "toy", "cartoon"),
    ),
    "05_Military_Truck_SAM": CategorySpec(
        strong_keywords=(
            "air defense",
            "missile launcher",
            "radar truck",
            "rocket launcher",
            "sam launcher",
            "surface to air missile",
            "surface-to-air missile",
            "military truck",
        ),
        weak_keywords=("truck", "launcher", "radar"),
        context_keywords=("military", "army", "missile", "sam", "air defense", "rocket"),
        negative_keywords=("pickup", "dump truck", "fire truck", "food truck", "toy", "cartoon"),
    ),
}

EXCLUDE_KEYWORDS = (
    "cartoon",
    "fantasy",
    "lego",
    "low poly",
    "low-poly",
    "minecraft",
    "paper",
    "toy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a keyword-filtered Objaverse military 3D dataset."
    )
    parser.add_argument(
        "--target-per-class",
        type=int,
        default=100,
        help="Number of objects to select for each category.",
    )
    parser.add_argument(
        "--candidate-multiplier",
        type=float,
        default=2.0,
        help="Select this many candidate UIDs per requested object before download.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "Military_3D_Dataset",
        help="Directory where categorized model files will be copied.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Selection manifest path. Defaults to <output-dir>/selection_manifest.json.",
    )
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=None,
        help="Candidate review CSV path. Defaults to <output-dir>/candidate_review.csv.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20_000,
        help="Number of UID annotations loaded per scan batch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to shuffle Objaverse UIDs before scanning.",
    )
    parser.add_argument(
        "--download-processes",
        type=int,
        default=1,
        help="Parallel download workers passed to objaverse.load_objects.",
    )
    parser.add_argument(
        "--download-batch-size",
        type=int,
        default=50,
        help="Number of selected UIDs to download per objaverse.load_objects call.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Retry count for unstable network calls.",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=15.0,
        help="Initial seconds to sleep before retrying network calls.",
    )
    parser.add_argument(
        "--reuse-manifest",
        action="store_true",
        help="Reuse an existing manifest instead of scanning annotations again.",
    )
    parser.add_argument(
        "--reject-uids",
        type=Path,
        default=None,
        help="Text file with one rejected UID per line. Defaults to <output-dir>/rejected_uids.txt.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only scan/select UIDs and write the manifest; do not download objects.",
    )
    parser.add_argument(
        "--skip-failed-batches",
        action="store_true",
        help="Skip annotation batches that still fail after retries.",
    )
    return parser.parse_args()


def annotation_text(annotation: dict[str, Any]) -> str:
    """Create a lowercase searchable string from an Objaverse annotation."""
    fields: list[str] = [
        str(annotation.get("name", "")),
        str(annotation.get("description", "")),
    ]
    fields.extend(str(tag.get("name", "")) for tag in annotation.get("tags", []))
    fields.extend(str(category) for category in annotation.get("categories", []))
    return " ".join(fields).lower()


def keyword_present(text: str, keyword: str) -> bool:
    pattern = r"(?<![a-z0-9])" + re.escape(keyword.lower()) + r"(?![a-z0-9])"
    return re.search(pattern, text) is not None


def keyword_hits(text: str, keywords: tuple[str, ...]) -> list[str]:
    return [keyword for keyword in keywords if keyword_present(text, keyword)]


def score_annotation(annotation: dict[str, Any], spec: CategorySpec) -> tuple[int, list[str]]:
    text = annotation_text(annotation)
    global_negatives = keyword_hits(text, EXCLUDE_KEYWORDS)
    category_negatives = keyword_hits(text, spec.negative_keywords)
    if global_negatives or category_negatives:
        return 0, [f"reject:{','.join(global_negatives + category_negatives)}"]

    strong_hits = keyword_hits(text, spec.strong_keywords)
    weak_hits = keyword_hits(text, spec.weak_keywords)
    context_hits = keyword_hits(text, spec.context_keywords)

    score = 0
    reasons: list[str] = []
    if strong_hits:
        score += 100 + 10 * len(strong_hits)
        reasons.append("strong=" + ",".join(strong_hits))
    if weak_hits and context_hits:
        score += 40 + 5 * len(context_hits)
        reasons.append("weak=" + ",".join(weak_hits))
        reasons.append("context=" + ",".join(context_hits))
    if weak_hits and not context_hits and not strong_hits:
        reasons.append("weak_without_context=" + ",".join(weak_hits))
        return 0, reasons
    return score, reasons


def enough_uids(category_uids: dict[str, list[str]], target_per_class: int) -> bool:
    return all(len(uids) >= target_per_class for uids in category_uids.values())


def enough_candidates(category_uids: dict[str, list[str]], candidates_per_class: int) -> bool:
    return all(len(uids) >= candidates_per_class for uids in category_uids.values())


def require_objaverse():
    try:
        import objaverse
    except ImportError as error:
        raise RuntimeError(
            "The 'objaverse' package is not installed in this Python environment. "
            "Install it with: pip install objaverse"
        ) from error
    return objaverse


def with_retries(label: str, attempts: int, initial_sleep: float, func):
    """Run a network operation with simple exponential backoff."""
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as error:  # Objaverse may raise urllib, requests, or fsspec errors.
            last_error = error
            if attempt >= attempts:
                break
            sleep_seconds = initial_sleep * (2 ** (attempt - 1))
            print(
                f"  [retry] {label} failed on attempt {attempt}/{attempts}: {error}. "
                f"Sleeping {sleep_seconds:.1f}s..."
            )
            time.sleep(sleep_seconds)
    raise RuntimeError(f"{label} failed after {attempts} attempts") from last_error


def load_manifest(manifest_path: Path) -> dict[str, list[str]]:
    with manifest_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    categories = payload.get("categories", payload)
    return {cat: list(uids) for cat, uids in categories.items()}


def load_rejected_uids(reject_uids_path: Path) -> set[str]:
    if not reject_uids_path.exists():
        return set()
    rejected: set[str] = set()
    with reject_uids_path.open("r", encoding="utf-8") as file:
        for line in file:
            uid = line.strip()
            if uid and not uid.startswith("#"):
                rejected.add(uid)
    return rejected


def filter_rejected_uids(
    category_uids: dict[str, list[str]],
    rejected_uids: set[str],
) -> dict[str, list[str]]:
    if not rejected_uids:
        return category_uids
    filtered = {
        category: [uid for uid in uids if uid not in rejected_uids]
        for category, uids in category_uids.items()
    }
    removed = sum(len(category_uids[category]) - len(filtered[category]) for category in category_uids)
    print(f"[review] Excluded {removed} rejected UIDs from selection.")
    return filtered


def save_manifest(
    manifest_path: Path,
    category_uids: dict[str, list[str]],
    target_per_class: int,
    seed: int,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "objaverse-1.0",
        "target_per_class": target_per_class,
        "seed": seed,
        "categories": category_uids,
    }
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def annotation_summary(annotation: dict[str, Any]) -> dict[str, str]:
    tags = ", ".join(str(tag.get("name", "")) for tag in annotation.get("tags", []))
    categories = ", ".join(str(category) for category in annotation.get("categories", []))
    return {
        "name": str(annotation.get("name", "")),
        "description": str(annotation.get("description", ""))[:300],
        "tags": tags[:500],
        "categories": categories[:300],
    }


def save_review_csv(review_csv_path: Path, rows: list[dict[str, Any]]) -> None:
    review_csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "category",
        "rank",
        "uid",
        "score",
        "reasons",
        "name",
        "tags",
        "categories",
        "description",
    ]
    with review_csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def select_uids(
    args: argparse.Namespace,
    manifest_path: Path,
    review_csv_path: Path,
) -> dict[str, list[str]]:
    objaverse = require_objaverse()
    if args.retries <= 0:
        raise ValueError("--retries must be greater than 0")
    if args.retry_sleep < 0:
        raise ValueError("--retry-sleep must be greater than or equal to 0")
    if args.candidate_multiplier < 1.0:
        raise ValueError("--candidate-multiplier must be greater than or equal to 1.0")

    if args.reuse_manifest and manifest_path.exists():
        print(f"[manifest] Reusing selected UIDs from {manifest_path}")
        return load_manifest(manifest_path)

    print("[1/4] Loading Objaverse UID list. This may take a while...")
    all_uids = with_retries(
        "load_uids",
        args.retries,
        args.retry_sleep,
        objaverse.load_uids,
    )
    print(f"[1/4] Loaded {len(all_uids):,} UIDs.")

    rng = random.Random(args.seed)
    rng.shuffle(all_uids)

    candidates_per_class = max(args.target_per_class, int(args.target_per_class * args.candidate_multiplier))
    category_records: dict[str, list[dict[str, Any]]] = {cat: [] for cat in CATEGORIES}
    seen_uids: set[str] = set()

    print("[2/4] Scanning annotations and selecting matching objects...")
    for start in range(0, len(all_uids), args.batch_size):
        batch_uids = all_uids[start : start + args.batch_size]
        try:
            annotations = with_retries(
                f"load_annotations batch {start:,}-{start + len(batch_uids):,}",
                args.retries,
                args.retry_sleep,
                lambda batch_uids=batch_uids: objaverse.load_annotations(batch_uids),
            )
        except RuntimeError as error:
            if not args.skip_failed_batches:
                raise
            print(f"  [warn] Skipping failed annotation batch: {error}")
            continue

        for uid, annotation in annotations.items():
            if uid in seen_uids:
                continue

            for category, spec in CATEGORIES.items():
                if len(category_records[category]) >= candidates_per_class:
                    continue
                score, reasons = score_annotation(annotation, spec)
                if score > 0:
                    row = {
                        "category": category,
                        "uid": uid,
                        "score": score,
                        "reasons": "; ".join(reasons),
                        **annotation_summary(annotation),
                    }
                    category_records[category].append(row)
                    seen_uids.add(uid)
                    break

        progress = {cat: len(rows) for cat, rows in category_records.items()}
        print(f"  scanned {start + len(batch_uids):,}/{len(all_uids):,}: {progress}")

        if enough_candidates({cat: [row["uid"] for row in rows] for cat, rows in category_records.items()}, candidates_per_class):
            print("[2/4] All categories reached the candidate count.")
            break

    review_rows: list[dict[str, Any]] = []
    category_uids: dict[str, list[str]] = {}
    for category, rows in category_records.items():
        sorted_rows = sorted(rows, key=lambda row: (-int(row["score"]), str(row["uid"])))
        category_uids[category] = [str(row["uid"]) for row in sorted_rows[:candidates_per_class]]
        for rank, row in enumerate(sorted_rows, start=1):
            review_rows.append({**row, "rank": rank})

    save_manifest(manifest_path, category_uids, args.target_per_class, args.seed)
    print(f"[manifest] Saved selected UIDs to {manifest_path}")
    save_review_csv(review_csv_path, review_rows)
    print(f"[review] Saved candidate review CSV to {review_csv_path}")
    return category_uids


def copy_downloads(
    downloaded_objects: dict[str, str],
    category_uids: dict[str, list[str]],
    output_dir: Path,
    target_per_class: int,
) -> None:
    print("[4/4] Copying downloaded files into category folders...")
    output_dir.mkdir(parents=True, exist_ok=True)

    for category, uids in category_uids.items():
        category_dir = output_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        existing_files = sorted(path for path in category_dir.iterdir() if path.is_file())
        existing_prefixes = {uid_prefix_from_name(path.name) for path in existing_files}
        existing_prefixes.discard(None)
        copied = len(existing_files)

        for uid in uids:
            if copied >= target_per_class:
                break
            if uid[:8] in existing_prefixes:
                continue
            src = downloaded_objects.get(uid)
            if not src:
                print(f"  [warn] Download missing for UID {uid} in {category}")
                continue

            src_path = Path(src)
            dst_name = f"{category}_{copied + 1:03d}_{uid[:8]}{src_path.suffix}"
            shutil.copy2(src_path, category_dir / dst_name)
            existing_prefixes.add(uid[:8])
            copied += 1

        status = "ok" if copied == target_per_class else "short"
        print(f"  {category}: present {copied}/{target_per_class} objects ({status}; candidates={len(uids)})")


def uid_prefix_from_name(filename: str) -> str | None:
    stem = Path(filename).stem
    match = re.search(r"_([0-9a-f]{8})$", stem, re.IGNORECASE)
    return match.group(1).lower() if match else None


def existing_uid_prefixes(category_dir: Path) -> set[str]:
    if not category_dir.exists():
        return set()
    prefixes = {uid_prefix_from_name(path.name) for path in category_dir.iterdir() if path.is_file()}
    prefixes.discard(None)
    return prefixes


def pending_uids_for_resume(
    category_uids: dict[str, list[str]],
    output_dir: Path,
    target_per_class: int,
) -> list[str]:
    pending: list[str] = []
    seen: set[str] = set()

    for category, uids in category_uids.items():
        category_dir = output_dir / category
        existing_count = len([path for path in category_dir.iterdir() if path.is_file()]) if category_dir.exists() else 0
        if existing_count >= target_per_class:
            print(f"  [resume] {category}: already has {existing_count}/{target_per_class}; skipping downloads.")
            continue

        existing_prefixes = existing_uid_prefixes(category_dir)
        needed = target_per_class - existing_count
        added = 0
        for uid in uids:
            if uid[:8].lower() in existing_prefixes or uid in seen:
                continue
            pending.append(uid)
            seen.add(uid)
            added += 1
            if added >= needed:
                break

        print(
            f"  [resume] {category}: has {existing_count}/{target_per_class}; "
            f"queued {added} candidate downloads."
        )

    return pending


def download_objects_in_batches(
    objaverse,
    selected_uids: list[str],
    download_processes: int,
    batch_size: int,
    retries: int,
    retry_sleep: float,
) -> dict[str, str]:
    downloaded_objects: dict[str, str] = {}
    total = len(selected_uids)

    for start in range(0, total, batch_size):
        batch_uids = selected_uids[start : start + batch_size]
        end = start + len(batch_uids)
        print(f"  downloading batch {start + 1}-{end}/{total}...")
        try:
            batch_downloads = with_retries(
                f"load_objects batch {start + 1}-{end}",
                retries,
                retry_sleep,
                lambda batch_uids=batch_uids: objaverse.load_objects(
                    uids=batch_uids,
                    download_processes=download_processes,
                ),
            )
        except RuntimeError as error:
            if len(batch_uids) == 1:
                print(f"  [warn] Skipping failed UID {batch_uids[0]}: {error}")
                continue

            print(f"  [warn] Batch {start + 1}-{end} failed; retrying one UID at a time.")
            single_downloads = download_objects_in_batches(
                objaverse,
                batch_uids,
                download_processes,
                1,
                retries,
                retry_sleep,
            )
            downloaded_objects.update(single_downloads)
            continue

        downloaded_objects.update(batch_downloads)

    return downloaded_objects


def main() -> None:
    args = parse_args()
    if args.target_per_class <= 0:
        raise ValueError("--target-per-class must be greater than 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")
    if args.download_processes <= 0:
        raise ValueError("--download-processes must be greater than 0")
    if args.download_batch_size <= 0:
        raise ValueError("--download-batch-size must be greater than 0")

    output_dir = args.output_dir.resolve()
    manifest_path = (
        args.manifest.resolve()
        if args.manifest is not None
        else output_dir / "selection_manifest.json"
    )
    review_csv_path = (
        args.review_csv.resolve()
        if args.review_csv is not None
        else output_dir / "candidate_review.csv"
    )
    reject_uids_path = (
        args.reject_uids.resolve()
        if args.reject_uids is not None
        else output_dir / "rejected_uids.txt"
    )

    print("==============================================")
    print("  Objaverse military 3D dataset generator")
    print("==============================================")
    print(f"Output directory: {output_dir}")
    print(f"Target per class: {args.target_per_class}")

    rejected_uids = load_rejected_uids(reject_uids_path)
    category_uids = filter_rejected_uids(select_uids(args, manifest_path, review_csv_path), rejected_uids)
    selected_uids = [uid for uids in category_uids.values() for uid in uids]

    if not selected_uids:
        print("[stop] No matching Objaverse objects were selected.")
        return

    print("[summary] Selected UID counts:")
    for category, uids in category_uids.items():
        status = "ok" if len(uids) >= args.target_per_class else "short"
        print(f"  {category}: {len(uids)} ({status})")

    if args.dry_run:
        print("[dry-run] Selection completed. Skipping downloads.")
        return

    selected_uids = pending_uids_for_resume(category_uids, output_dir, args.target_per_class)
    if not selected_uids:
        print("[resume] All category folders already reached the target count.")
        return

    print(f"[3/4] Downloading {len(selected_uids)} Objaverse objects...")
    objaverse = require_objaverse()
    downloaded_objects = download_objects_in_batches(
        objaverse,
        selected_uids,
        args.download_processes,
        args.download_batch_size,
        args.retries,
        args.retry_sleep,
    )
    copy_downloads(downloaded_objects, category_uids, output_dir, args.target_per_class)

    print(f"[done] Dataset copied to: {output_dir}")


if __name__ == "__main__":
    main()

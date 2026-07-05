"""Build contact sheets from rendered review thumbnails."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DEFAULT_DATASET_DIR = Path(__file__).resolve().parent / "Military_3D_Dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create review contact sheets from thumbnails.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--thumb-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--candidate-review", type=Path, default=None)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--label-height", type=int, default=48)
    return parser.parse_args()


def uid_prefix_from_name(filename: str) -> str:
    return Path(filename).stem.rsplit("_", 1)[-1]


def load_uid_lookup(candidate_review_path: Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    if not candidate_review_path.exists():
        return lookup
    with candidate_review_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            uid = row.get("uid", "")
            if uid:
                lookup[uid[:8].lower()] = uid
    return lookup


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def make_sheet(
    category: str,
    thumbs: list[Path],
    out_path: Path,
    dataset_dir: Path,
    uid_lookup: dict[str, str],
    columns: int,
    tile_size: int,
    label_height: int,
) -> list[dict[str, str]]:
    rows = max(1, math.ceil(len(thumbs) / columns))
    sheet = Image.new("RGB", (columns * tile_size, rows * (tile_size + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    font = load_font(14)
    small_font = load_font(11)
    csv_rows: list[dict[str, str]] = []

    for index, thumb_path in enumerate(thumbs, start=1):
        col = (index - 1) % columns
        row = (index - 1) // columns
        x = col * tile_size
        y = row * (tile_size + label_height)

        image = Image.open(thumb_path).convert("RGB")
        image.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
        px = x + (tile_size - image.width) // 2
        py = y + (tile_size - image.height) // 2
        sheet.paste(image, (px, py))
        draw.rectangle((x, y, x + tile_size - 1, y + tile_size + label_height - 1), outline=(210, 210, 210), width=1)

        uid_prefix = uid_prefix_from_name(thumb_path.name)
        model_path = dataset_dir / category / f"{thumb_path.stem}.glb"
        draw.text((x + 6, y + tile_size + 5), f"{index:03d}  {uid_prefix}", fill=(0, 0, 0), font=font)
        draw.text((x + 6, y + tile_size + 25), thumb_path.stem[:32], fill=(80, 80, 80), font=small_font)
        csv_rows.append(
            {
                "keep": "1",
                "category": category,
                "sheet_index": str(index),
                "uid_prefix": uid_prefix,
                "uid": uid_lookup.get(uid_prefix.lower(), ""),
                "file": str(model_path),
                "thumbnail": str(thumb_path),
                "reason": "",
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return csv_rows


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    thumb_dir = (args.thumb_dir or dataset_dir / "_review_thumbnails").resolve()
    out_dir = (args.out_dir or dataset_dir / "_review_sheets").resolve()
    candidate_review_path = (
        args.candidate_review.resolve()
        if args.candidate_review is not None
        else dataset_dir / "candidate_review.csv"
    )
    uid_lookup = load_uid_lookup(candidate_review_path)
    all_rows: list[dict[str, str]] = []

    for category_dir in sorted(path for path in thumb_dir.iterdir() if path.is_dir()):
        thumbs = sorted(category_dir.glob("*.png"))
        if not thumbs:
            continue
        out_path = out_dir / f"{category_dir.name}.png"
        rows = make_sheet(
            category_dir.name,
            thumbs,
            out_path,
            dataset_dir,
            uid_lookup,
            args.columns,
            args.tile_size,
            args.label_height,
        )
        all_rows.extend(rows)
        print(f"[sheet] {category_dir.name}: {len(thumbs)} thumbnails -> {out_path}")

    csv_path = out_dir / "thumbnail_review.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        fieldnames = ["keep", "category", "sheet_index", "uid_prefix", "uid", "file", "thumbnail", "reason"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[done] Wrote {csv_path}")


if __name__ == "__main__":
    main()

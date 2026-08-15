#!/usr/bin/env python3
"""Create leakage-resistant rooster-detection train/val/test splits.

The input image names encode the camera and the Unix timestamp of the source
recording.  All frames with the same ``camera + timestamp`` are treated as one
source session and are assigned to exactly one split.

The recommended split for the current 909-image dataset is:

* test: 2025-07-16 through 2025-07-21 (independent date holdout);
* val:  2025-11-19 (used only for model selection/early stopping);
* train: all remaining dates.

This produces a 724/185 development/test split (79.65%/20.35%), followed by a
654/70 train/validation split inside the development data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SOURCE_RE = re.compile(
    r"^(?P<camera>(?:helix|nx)\d+-\d+)-(?P<timestamp>\d{10,13})(?P<rest>.*)$",
    flags=re.IGNORECASE,
)
FRAME_SUFFIX_RE = re.compile(r"(?:_frame_\d+|_(?:first|middle|last))$", re.IGNORECASE)
FINAL_INTEGER_RE = re.compile(r"-\d+$")
CHINA_TIME = timezone(timedelta(hours=8), name="Asia/Shanghai")


@dataclass(frozen=True)
class ImageRecord:
    filename: str
    path: str
    split: str
    camera: str
    timestamp_raw: str
    captured_at: str
    capture_date: str
    source_group: str
    clip_id: str
    sha256: str = ""


def parse_date_spec(values: Sequence[str]) -> set[date]:
    """Parse YYYY-MM-DD values and inclusive YYYY-MM-DD:YYYY-MM-DD ranges."""
    result: set[date] = set()
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                result.add(date.fromisoformat(item))
                continue
            start_text, end_text = item.split(":", maxsplit=1)
            start = date.fromisoformat(start_text)
            end = date.fromisoformat(end_text)
            if end < start:
                raise ValueError(f"Invalid descending date range: {item}")
            current = start
            while current <= end:
                result.add(current)
                current += timedelta(days=1)
    return result


def parse_source(stem: str) -> tuple[str, str, datetime, str, str]:
    """Return camera, raw timestamp, datetime, source group, and clip ID."""
    match = SOURCE_RE.match(stem)
    if not match:
        raise ValueError(
            "filename does not begin with a recognized camera and 10/13-digit "
            f"Unix timestamp: {stem}"
        )

    camera = match.group("camera")
    timestamp_raw = match.group("timestamp")
    seconds = int(timestamp_raw)
    if len(timestamp_raw) == 13:
        seconds //= 1000
    captured_at = datetime.fromtimestamp(seconds, tz=timezone.utc).astimezone(CHINA_TIME)

    # Deliberately conservative grouping: suffixes such as "-1", "-2", and
    # "-3", or "-637-7" and "-728-8", may be adjacent/derived clips from the
    # same recording session.  They therefore share one source_group.
    source_group = f"{camera}-{timestamp_raw}"

    clip_id = FRAME_SUFFIX_RE.sub("", stem)
    if camera.lower().startswith("helix"):
        clip_id = FINAL_INTEGER_RE.sub("", clip_id)
    elif len(timestamp_raw) == 13:
        # These names end in a sampled-frame counter, not a distinct clip ID.
        clip_id = source_group

    return camera, timestamp_raw, captured_at, source_group, clip_id


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_images(images_dir: Path, recursive: bool) -> list[Path]:
    iterator: Iterable[Path]
    iterator = images_dir.rglob("*") if recursive else images_dir.iterdir()
    return sorted(
        (
            path.resolve()
            for path in iterator
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: str(path).lower(),
    )


def assign_split(capture_date: date, test_dates: set[date], val_dates: set[date]) -> str:
    if capture_date in test_dates:
        return "test"
    if capture_date in val_dates:
        return "val"
    return "train"


def build_records(
    image_paths: Sequence[Path],
    test_dates: set[date],
    val_dates: set[date],
    hash_images: bool,
) -> tuple[list[ImageRecord], list[str]]:
    records: list[ImageRecord] = []
    parse_errors: list[str] = []

    for image_path in image_paths:
        try:
            camera, timestamp_raw, captured_at, source_group, clip_id = parse_source(
                image_path.stem
            )
        except ValueError as exc:
            parse_errors.append(f"{image_path.name}: {exc}")
            continue

        split = assign_split(captured_at.date(), test_dates, val_dates)
        records.append(
            ImageRecord(
                filename=image_path.name,
                path=str(image_path),
                split=split,
                camera=camera,
                timestamp_raw=timestamp_raw,
                captured_at=captured_at.isoformat(),
                capture_date=captured_at.date().isoformat(),
                source_group=source_group,
                clip_id=clip_id,
                sha256=sha256_file(image_path) if hash_images else "",
            )
        )

    return records, parse_errors


def validate_records(
    records: Sequence[ImageRecord],
    discovered_count: int,
    parse_errors: Sequence[str],
    test_dates: set[date],
    val_dates: set[date],
) -> dict:
    errors: list[str] = []
    warnings: list[str] = []

    if parse_errors:
        errors.append(f"{len(parse_errors)} filenames could not be parsed")
    if len(records) != discovered_count:
        errors.append(
            f"Only {len(records)} of {discovered_count} discovered images were assigned"
        )
    if test_dates & val_dates:
        errors.append(
            "test_dates and val_dates overlap: "
            + ", ".join(sorted(day.isoformat() for day in test_dates & val_dates))
        )

    observed_dates = {date.fromisoformat(record.capture_date) for record in records}
    missing_test_dates = test_dates - observed_dates
    missing_val_dates = val_dates - observed_dates
    if missing_test_dates:
        warnings.append(
            "Requested test dates absent from the dataset: "
            + ", ".join(sorted(day.isoformat() for day in missing_test_dates))
        )
    if missing_val_dates:
        warnings.append(
            "Requested validation dates absent from the dataset: "
            + ", ".join(sorted(day.isoformat() for day in missing_val_dates))
        )

    by_group: dict[str, set[str]] = defaultdict(set)
    by_clip: dict[str, set[str]] = defaultdict(set)
    by_hash: dict[str, set[str]] = defaultdict(set)
    by_date: dict[str, set[str]] = defaultdict(set)
    for record in records:
        by_group[record.source_group].add(record.split)
        by_clip[record.clip_id].add(record.split)
        by_date[record.capture_date].add(record.split)
        if record.sha256:
            by_hash[record.sha256].add(record.split)

    leaked_groups = {key: value for key, value in by_group.items() if len(value) > 1}
    leaked_clips = {key: value for key, value in by_clip.items() if len(value) > 1}
    leaked_dates = {key: value for key, value in by_date.items() if len(value) > 1}
    leaked_hashes = {key: value for key, value in by_hash.items() if len(value) > 1}

    if leaked_groups:
        errors.append(f"{len(leaked_groups)} source groups cross split boundaries")
    if leaked_clips:
        errors.append(f"{len(leaked_clips)} clip IDs cross split boundaries")
    if leaked_dates:
        errors.append(f"{len(leaked_dates)} capture dates cross split boundaries")
    if leaked_hashes:
        errors.append(f"{len(leaked_hashes)} exact image hashes cross split boundaries")

    split_counts = Counter(record.split for record in records)
    for split in ("train", "val", "test"):
        if split == "val" and not val_dates:
            continue
        if split_counts[split] == 0:
            errors.append(f"The {split} split is empty")

    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "unparsed_files": list(parse_errors),
        "source_group_overlap_count": len(leaked_groups),
        "clip_overlap_count": len(leaked_clips),
        "date_overlap_count": len(leaked_dates),
        "exact_hash_overlap_count": len(leaked_hashes),
    }


def summarize(records: Sequence[ImageRecord], audit: dict) -> dict:
    total = len(records)
    split_names = ("train", "val", "test")
    result: dict = {
        "total_images": total,
        "total_source_groups": len({record.source_group for record in records}),
        "total_clips": len({record.clip_id for record in records}),
        "total_dates": len({record.capture_date for record in records}),
        "splits": {},
        "audit": audit,
    }

    for split in split_names:
        selected = [record for record in records if record.split == split]
        result["splits"][split] = {
            "images": len(selected),
            "fraction": (len(selected) / total) if total else 0.0,
            "source_groups": len({record.source_group for record in selected}),
            "clips": len({record.clip_id for record in selected}),
            "dates": sorted({record.capture_date for record in selected}),
            "cameras": sorted({record.camera for record in selected}),
        }

    development_images = (
        result["splits"]["train"]["images"] + result["splits"]["val"]["images"]
    )
    result["development_vs_test"] = {
        "development_images": development_images,
        "test_images": result["splits"]["test"]["images"],
        "development_fraction": development_images / total if total else 0.0,
        "test_fraction": result["splits"]["test"]["images"] / total if total else 0.0,
    }
    return result


def write_manifest(
    records: Sequence[ImageRecord], output_dir: Path, images_dir: Path
) -> None:
    fieldnames = list(asdict(records[0]).keys())
    with (output_dir / "split_manifest.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["path"] = Path(record.path).relative_to(images_dir).as_posix()
            writer.writerow(row)

    for split in ("train", "val", "test"):
        with (output_dir / f"{split}.txt").open("w", encoding="utf-8") as handle:
            for record in records:
                if record.split == split:
                    relative_path = Path(record.path).relative_to(images_dir)
                    handle.write(relative_path.as_posix() + "\n")

    grouped: dict[tuple[str, str], list[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.split, record.source_group)].append(record)
    with (output_dir / "source_groups.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        fieldnames = [
            "split",
            "source_group",
            "capture_date",
            "camera",
            "image_count",
            "clip_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for (split, source_group), group_records in sorted(grouped.items()):
            writer.writerow(
                {
                    "split": split,
                    "source_group": source_group,
                    "capture_date": group_records[0].capture_date,
                    "camera": group_records[0].camera,
                    "image_count": len(group_records),
                    "clip_count": len({record.clip_id for record in group_records}),
                }
            )


def link_or_copy(source: Path, destination: Path, method: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {destination}")
    if method == "copy":
        shutil.copy2(source, destination)
    elif method == "hardlink":
        os.link(source, destination)
    elif method == "symlink":
        destination.symlink_to(source)
    else:
        raise ValueError(f"Unsupported materialization method: {method}")


def materialize_dataset(
    records: Sequence[ImageRecord],
    output_dir: Path,
    labels_dir: Path,
    label_suffix: str,
    method: str,
) -> None:
    missing_labels: list[Path] = []
    pairs: list[tuple[ImageRecord, Path]] = []
    for record in records:
        label_path = labels_dir / (Path(record.filename).stem + label_suffix)
        if not label_path.is_file():
            missing_labels.append(label_path)
        pairs.append((record, label_path))

    if missing_labels:
        preview = "\n".join(f"  {path}" for path in missing_labels[:10])
        raise FileNotFoundError(
            f"{len(missing_labels)} YOLO label files are missing. First entries:\n{preview}"
        )

    dataset_root = output_dir / "dataset"
    for record, label_path in pairs:
        image_source = Path(record.path)
        image_destination = dataset_root / "images" / record.split / record.filename
        label_destination = (
            dataset_root
            / "labels"
            / record.split
            / (image_source.stem + label_suffix)
        )
        link_or_copy(image_source, image_destination, method)
        link_or_copy(label_path.resolve(), label_destination, method)

    yaml_text = (
        f"path: {dataset_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        "  0: rooster\n"
    )
    (output_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Split rooster-detection images by capture date while keeping every "
            "camera+timestamp source session wholly within one split."
        )
    )
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--test-dates",
        required=True,
        nargs="+",
        help=(
            "Independent test dates. Accepts YYYY-MM-DD, comma-separated values, "
            "or inclusive YYYY-MM-DD:YYYY-MM-DD ranges."
        ),
    )
    parser.add_argument(
        "--val-dates",
        nargs="*",
        default=[],
        help="Validation dates used for model selection; same syntax as --test-dates.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search images-dir instead of reading only its top level.",
    )
    parser.add_argument(
        "--hash-images",
        action="store_true",
        help="Compute SHA-256 hashes and verify that exact duplicates do not cross splits.",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        help="Directory containing YOLO .txt labels; required with --materialize.",
    )
    parser.add_argument(
        "--label-suffix",
        default=".txt",
        help="Label extension used with --materialize (default: .txt).",
    )
    parser.add_argument(
        "--materialize",
        choices=("none", "copy", "hardlink", "symlink"),
        default="none",
        help=(
            "Optionally create dataset/images/{train,val,test} and matching labels. "
            "The default writes audit manifests only."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow writing into an existing output directory (files are still not overwritten).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    images_dir = args.images_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not images_dir.is_dir():
        parser.error(f"images-dir does not exist or is not a directory: {images_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        parser.error(
            f"output-dir is not empty: {output_dir}. Use a new directory or --force."
        )
    if args.materialize != "none" and args.labels_dir is None:
        parser.error("--labels-dir is required when --materialize is not 'none'")

    try:
        test_dates = parse_date_spec(args.test_dates)
        val_dates = parse_date_spec(args.val_dates)
    except ValueError as exc:
        parser.error(str(exc))

    image_paths = discover_images(images_dir, args.recursive)
    if not image_paths:
        parser.error(f"No supported image files found in {images_dir}")

    records, parse_errors = build_records(
        image_paths=image_paths,
        test_dates=test_dates,
        val_dates=val_dates,
        hash_images=args.hash_images,
    )
    audit = validate_records(
        records=records,
        discovered_count=len(image_paths),
        parse_errors=parse_errors,
        test_dates=test_dates,
        val_dates=val_dates,
    )
    summary = summarize(records, audit)

    output_dir.mkdir(parents=True, exist_ok=True)
    if records:
        write_manifest(records, output_dir, images_dir)
    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if args.materialize != "none":
        materialize_dataset(
            records=records,
            output_dir=output_dir,
            labels_dir=args.labels_dir.resolve(),
            label_suffix=args.label_suffix,
            method=args.materialize,
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not audit["passed"]:
        print(
            "Split audit FAILED. See split_summary.json for details.",
            file=sys.stderr,
        )
        return 2
    print(f"Split audit passed. Outputs written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

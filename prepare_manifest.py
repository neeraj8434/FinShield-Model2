#!/usr/bin/env python3
"""Validate and normalize a licensed dataset's labeled video manifest.

The input CSV must provide video_path, label, split, source_id, group_id, and
manipulation_type. Paths must be relative to --data-root. Split/group metadata
is required so paired originals and derived fakes cannot silently leak.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


REQUIRED = {"video_path", "label", "split", "source_id", "group_id", "manipulation_type"}
OPTIONAL = ("dataset", "identity_id", "license", "source_url")
LABELS = {"real": "REAL", "0": "REAL", "fake": "FAKE", "1": "FAKE"}
SPLITS = {"train", "val", "test"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV containing labels and official provenance/split metadata")
    parser.add_argument("--data-root", required=True, help="Root directory for the licensed dataset")
    parser.add_argument("--output", required=True, help="Normalized manifest CSV destination")
    parser.add_argument("--dataset-name", default=None, help="Override the dataset column for every row")
    parser.add_argument("--audit-output", default=None, help="Audit JSON path (default: <manifest stem>_audit.json)")
    args = parser.parse_args()

    source = Path(args.input).resolve()
    root = Path(args.data_root).resolve()
    output = Path(args.output).resolve()
    audit_output = Path(args.audit_output).resolve() if args.audit_output else output.with_name(output.stem + "_audit.json")
    if not source.is_file():
        raise FileNotFoundError(source)
    if not root.is_dir():
        raise NotADirectoryError(root)

    with source.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")
        input_rows = list(reader)
    if not input_rows:
        raise ValueError("Input CSV contains no rows")

    rows = []
    seen_paths = set()
    group_splits: dict[str, set[str]] = defaultdict(set)
    counts: dict[str, Counter] = defaultdict(Counter)
    method_counts: dict[str, Counter] = defaultdict(Counter)
    hashes: dict[str, list[str]] = defaultdict(list)
    for line_number, raw in enumerate(input_rows, start=2):
        path_value = (raw.get("video_path") or "").strip().replace("\\", "/")
        relative = PurePosixPath(path_value)
        if not path_value or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Line {line_number}: video_path must be a safe relative path, got {path_value!r}")
        video = (root / Path(*relative.parts)).resolve()
        if not video.is_relative_to(root):
            raise ValueError(f"Line {line_number}: path escapes the dataset root")
        if video.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError(f"Line {line_number}: unsupported video extension {video.suffix!r}")
        if not video.is_file():
            raise FileNotFoundError(f"Line {line_number}: video not found: {video}")
        relative_key = video.relative_to(root).as_posix()
        if relative_key in seen_paths:
            raise ValueError(f"Line {line_number}: duplicate video_path {relative_key!r}")
        seen_paths.add(relative_key)

        label = LABELS.get((raw.get("label") or "").strip().lower())
        split = (raw.get("split") or "").strip().lower()
        source_id = (raw.get("source_id") or "").strip()
        group_id = (raw.get("group_id") or "").strip()
        method = (raw.get("manipulation_type") or "").strip()
        if label is None or split not in SPLITS:
            raise ValueError(f"Line {line_number}: label must be REAL/FAKE and split train/val/test")
        if not source_id or not group_id or not method:
            raise ValueError(f"Line {line_number}: source_id, group_id and manipulation_type cannot be empty")
        dataset_name = args.dataset_name or (raw.get("dataset") or "").strip()
        if not dataset_name:
            raise ValueError(f"Line {line_number}: provide --dataset-name or a non-empty dataset column")
        group_splits[group_id].add(split)
        counts[split][label] += 1
        method_counts[split][f"{label}:{method}"] += 1
        sha = digest(video)
        hashes[sha].append(relative_key)
        rows.append({
            "video_path": relative_key,
            "label": label,
            "dataset": dataset_name,
            "split": split,
            "source_id": source_id,
            "group_id": group_id,
            "manipulation_type": method,
            "identity_id": (raw.get("identity_id") or "").strip(),
            "license": (raw.get("license") or "").strip(),
            "source_url": (raw.get("source_url") or "").strip(),
        })
    leaked_groups = {group: sorted(splits) for group, splits in group_splits.items() if len(splits) > 1}
    if leaked_groups:
        sample = dict(list(leaked_groups.items())[:10])
        raise ValueError(f"Source groups cross splits ({len(leaked_groups)} groups); refusing to write manifest. Examples: {sample}")

    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["video_path", "label", "dataset", "split", "source_id", "group_id",
              "manipulation_type", "identity_id", "license", "source_url"]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["split"], row["label"], row["video_path"])))
    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    audit = {
        "dataset_root": str(root),
        "source_metadata_csv": str(source),
        "manifest": str(output),
        "video_count": len(rows),
        "exact_sha256_files_checked": len(rows),
        "exact_duplicate_hash_groups": duplicate_groups,
        "group_leakage_count": 0,
        "group_count": len(group_splits),
        "counts_by_split_and_label": {split: dict(counts[split]) for split in ("train", "val", "test")},
        "counts_by_split_label_and_manipulation": {split: dict(method_counts[split]) for split in ("train", "val", "test")},
        "license": "Values copied from input provenance; verify dataset terms and access approval before training.",
        "identity_grouping": "Input group_id/source_id values are preserved; importer does not infer identity links.",
    }
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

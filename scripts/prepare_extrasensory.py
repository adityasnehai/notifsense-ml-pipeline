"""Prepare ExtraSensory per-UUID files into metadata and split artifacts.

This script is intentionally dependency-free (stdlib only) so it can run before
installing the ML stack. It does not build tensors yet; it creates:
- data/processed/extrasensory/manifest.csv
- data/processed/extrasensory/metadata.json
- data/splits/uuid_splits.json
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple


def list_uuid_files(raw_dir: Path) -> List[Path]:
    files = sorted(raw_dir.glob("*.features_labels.csv.gz"))
    if not files:
        raise FileNotFoundError(f"No *.features_labels.csv.gz files found in {raw_dir}")
    return files


def read_header_columns(gz_csv_path: Path) -> List[str]:
    with gzip.open(gz_csv_path, mode="rt", newline="") as f:
        reader = csv.reader(f)
        return next(reader)


def split_columns(columns: List[str]) -> Tuple[List[str], List[str]]:
    label_cols = [c for c in columns if c.startswith("label:")]
    feature_cols = [c for c in columns if c != "timestamp" and not c.startswith("label:")]
    if not label_cols:
        raise ValueError("No label columns found (expected columns prefixed with 'label:').")
    if not feature_cols:
        raise ValueError("No feature columns found.")
    return feature_cols, label_cols


def is_nan_like(value: str) -> bool:
    v = value.strip().lower()
    return v == "" or v == "nan"


def is_positive_label(value: str) -> bool:
    if is_nan_like(value):
        return False
    try:
        return float(value) >= 0.5
    except ValueError:
        return False


def build_manifest_and_stats(
    files: List[Path],
    label_cols: List[str],
) -> Tuple[List[Dict[str, object]], Dict[str, int], int]:
    manifest: List[Dict[str, object]] = []
    global_label_positive_counts = {label: 0 for label in label_cols}
    total_rows_all = 0

    for file_path in files:
        uuid = file_path.name.split(".")[0]
        total_rows = 0
        rows_with_any_label = 0

        with gzip.open(file_path, mode="rt", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_rows += 1
                any_pos = False
                for label in label_cols:
                    value = row.get(label, "")
                    if is_positive_label(value):
                        any_pos = True
                        global_label_positive_counts[label] += 1
                if any_pos:
                    rows_with_any_label += 1

        total_rows_all += total_rows
        manifest.append(
            {
                "uuid": uuid,
                "file": str(file_path),
                "total_rows": total_rows,
                "rows_with_any_positive_label": rows_with_any_label,
            }
        )

    return manifest, global_label_positive_counts, total_rows_all


def split_uuids(uuids: List[str], seed: int, train_ratio: float, val_ratio: float) -> Dict[str, List[str]]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be in (0, 1)")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")

    rng = random.Random(seed)
    shuffled = list(uuids)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(math.floor(n * train_ratio))
    n_val = int(math.floor(n * val_ratio))
    n_test = n - n_train - n_val

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]

    if len(test) != n_test:
        raise RuntimeError("Split sizing mismatch")

    return {"train": train, "val": val, "test": test}


def write_manifest_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        fieldnames = ["uuid", "file", "total_rows", "rows_with_any_positive_label"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ExtraSensory metadata and UUID splits")
    parser.add_argument("--raw-dir", default="data/raw/extrasensory_per_uuid", help="Directory containing *.features_labels.csv.gz")
    parser.add_argument("--processed-dir", default="data/processed/extrasensory", help="Output directory for manifest/metadata")
    parser.add_argument("--splits-dir", default="data/splits", help="Output directory for split files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for UUID split")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    splits_dir = Path(args.splits_dir)

    files = list_uuid_files(raw_dir)
    columns = read_header_columns(files[0])
    feature_cols, label_cols = split_columns(columns)

    manifest, label_positive_counts, total_rows = build_manifest_and_stats(files, label_cols)
    uuids = [row["uuid"] for row in manifest]
    splits = split_uuids(uuids, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio)

    manifest_path = processed_dir / "manifest.csv"
    metadata_path = processed_dir / "metadata.json"
    splits_path = splits_dir / "uuid_splits.json"

    write_manifest_csv(manifest_path, manifest)
    write_json(
        metadata_path,
        {
            "source_raw_dir": str(raw_dir),
            "num_uuid_files": len(files),
            "num_rows_total": total_rows,
            "num_feature_columns": len(feature_cols),
            "num_label_columns": len(label_cols),
            "feature_columns": feature_cols,
            "label_columns": label_cols,
            "label_positive_counts": label_positive_counts,
        },
    )
    write_json(
        splits_path,
        {
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": 1.0 - args.train_ratio - args.val_ratio,
            "counts": {k: len(v) for k, v in splits.items()},
            "splits": splits,
        },
    )

    print(f"Prepared {len(files)} UUID files")
    print(f"Feature columns: {len(feature_cols)} | Label columns: {len(label_cols)}")
    print(f"Total rows: {total_rows}")
    print(f"Wrote: {manifest_path}")
    print(f"Wrote: {metadata_path}")
    print(f"Wrote: {splits_path}")


if __name__ == "__main__":
    main()

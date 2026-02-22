"""Prepare ExtraSensory data with explicit NotifSense feature/label taxonomy.

This script:
1. Ensures ExtraSensory per-UUID files are extracted under data/raw.
2. Loads and concatenates all per-UUID CSV.gz files.
3. Prints dataset overview and missingness stats.
4. Uses an explicit feature whitelist provided by product constraints.
5. Builds a fixed 24-label taxonomy with merged/derived labels.
6. Cleans missing values and performs UUID-aware train/val/test split.
7. Saves train/val/test CSVs and metadata JSON.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

# Exactly the requested feature list.
FEATURE_WHITELIST: List[str] = [
    "raw_acc:magnitude_stats:mean",
    "raw_acc:magnitude_stats:std",
    "raw_acc:magnitude_stats:moment3",
    "raw_acc:magnitude_stats:moment4",
    "raw_acc:magnitude_stats:percentile25",
    "raw_acc:magnitude_stats:percentile50",
    "raw_acc:magnitude_stats:percentile75",
    "raw_acc:magnitude_stats:value_entropy",
    "raw_acc:magnitude_stats:time_entropy",
    "raw_acc:magnitude_spectrum:log_energy_band0",
    "raw_acc:magnitude_spectrum:log_energy_band1",
    "raw_acc:magnitude_spectrum:log_energy_band2",
    "raw_acc:magnitude_spectrum:log_energy_band3",
    "raw_acc:magnitude_spectrum:log_energy_band4",
    "raw_acc:magnitude_spectrum:spectral_entropy",
    "raw_acc:magnitude_autocorrelation:period",
    "raw_acc:magnitude_autocorrelation:normalized_ac",
    "raw_acc:3d:mean_x",
    "raw_acc:3d:mean_y",
    "raw_acc:3d:mean_z",
    "raw_acc:3d:std_x",
    "raw_acc:3d:std_y",
    "raw_acc:3d:std_z",
    "raw_acc:3d:ro_xy",
    "raw_acc:3d:ro_xz",
    "raw_acc:3d:ro_yz",
    "proc_gyro:magnitude_stats:mean",
    "proc_gyro:magnitude_stats:std",
    "proc_gyro:magnitude_stats:moment3",
    "proc_gyro:magnitude_stats:moment4",
    "proc_gyro:magnitude_stats:percentile25",
    "proc_gyro:magnitude_stats:percentile50",
    "proc_gyro:magnitude_stats:percentile75",
    "proc_gyro:magnitude_stats:value_entropy",
    "proc_gyro:magnitude_stats:time_entropy",
    "proc_gyro:magnitude_spectrum:log_energy_band0",
    "proc_gyro:magnitude_spectrum:log_energy_band1",
    "proc_gyro:magnitude_spectrum:log_energy_band2",
    "proc_gyro:magnitude_spectrum:log_energy_band3",
    "proc_gyro:magnitude_spectrum:log_energy_band4",
    "proc_gyro:magnitude_spectrum:spectral_entropy",
    "proc_gyro:magnitude_autocorrelation:period",
    "proc_gyro:magnitude_autocorrelation:normalized_ac",
    "proc_gyro:3d:mean_x",
    "proc_gyro:3d:mean_y",
    "proc_gyro:3d:mean_z",
    "proc_gyro:3d:std_x",
    "proc_gyro:3d:std_y",
    "proc_gyro:3d:std_z",
    "proc_gyro:3d:ro_xy",
    "proc_gyro:3d:ro_xz",
    "proc_gyro:3d:ro_yz",
    "lf_measurements:light",
    "lf_measurements:screen_brightness",
    "lf_measurements:proximity",
    "lf_measurements:proximity_cm",
    "lf_measurements:battery_level",
    "discrete:battery_plugged:is_ac",
    "discrete:battery_plugged:is_usb",
    "discrete:battery_plugged:is_wireless",
    "discrete:battery_state:is_unplugged",
    "discrete:battery_state:is_not_charging",
    "discrete:battery_state:is_discharging",
    "discrete:battery_state:is_charging",
    "discrete:battery_state:is_full",
    "discrete:app_state:is_active",
    "discrete:app_state:is_inactive",
    "discrete:app_state:is_background",
    "discrete:ringer_mode:is_normal",
    "discrete:ringer_mode:is_silent_no_vibrate",
    "discrete:ringer_mode:is_silent_with_vibrate",
    "discrete:time_of_day:between0and6",
    "discrete:time_of_day:between3and9",
    "discrete:time_of_day:between6and12",
    "discrete:time_of_day:between9and15",
    "discrete:time_of_day:between12and18",
    "discrete:time_of_day:between15and21",
    "discrete:time_of_day:between18and24",
    "discrete:time_of_day:between21and3",
]

# Atomic label mapping (raw ExtraSensory label columns).
ATOMIC_LABEL_MAP: Dict[str, List[str]] = {
    "LyingDown": ["label:LYING_DOWN"],
    "Sleeping": ["label:SLEEPING"],
    "Sitting": ["label:SITTING"],
    "Standing": ["label:OR_standing"],
    "Walking": ["label:FIX_walking"],
    "Strolling": ["label:STROLLING"],
    "Running": ["label:FIX_running"],
    "Bicycling": ["label:BICYCLING"],
    "Exercising": ["label:OR_exercise"],
    "Stairs": ["label:STAIRS_-_GOING_UP", "label:STAIRS_-_GOING_DOWN"],
    "Elevator": ["label:ELEVATOR"],
    "PhoneInPocket": ["label:PHONE_IN_POCKET"],
    "PhoneInHand": ["label:PHONE_IN_HAND"],
    "PhoneInBag": ["label:PHONE_IN_BAG"],
    "PhoneOnTable": ["label:PHONE_ON_TABLE"],
    "Indoors": ["label:OR_indoors"],
    "Outdoors": ["label:OR_outside"],
    "InCar": ["label:IN_A_CAR"],
    "OnBus": ["label:ON_A_BUS"],
    "AtGym": ["label:AT_THE_GYM"],
    "ComputerWork": ["label:COMPUTER_WORK"],
    "InMeeting": ["label:IN_A_MEETING"],
}

# Final output order requested by user (includes derived labels).
FINAL_LABEL_ORDER: List[str] = [
    "LyingDown",
    "Sleeping",
    "Sitting",
    "Standing",
    "Walking",
    "Strolling",
    "Running",
    "Bicycling",
    "Exercising",
    "Stairs",
    "Elevator",
    "Stationary",
    "Moving",
    "PhoneInPocket",
    "PhoneInHand",
    "PhoneInBag",
    "PhoneOnTable",
    "Indoors",
    "Outdoors",
    "InCar",
    "OnBus",
    "AtGym",
    "ComputerWork",
    "InMeeting",
]


def find_zip_path(project_root: Path) -> Path:
    candidates = [
        project_root / "dataset" / "ExtraSensory.per_uuid_features_labels.zip",
        project_root / "data" / "raw" / "ExtraSensory.per_uuid_features_labels.zip",
        project_root / "ExtraSensory.per_uuid_features_labels.zip",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find ExtraSensory.per_uuid_features_labels.zip.")


def ensure_extracted(project_root: Path) -> Path:
    raw_dir = project_root / "data" / "raw" / "extrasensory_per_uuid"
    raw_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(raw_dir.glob("*.features_labels.csv.gz"))
    if existing:
        print(f"Extraction already available: {len(existing)} files in {raw_dir}")
        return raw_dir

    zip_path = find_zip_path(project_root)
    print(f"Extracting {zip_path} -> {raw_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)

    extracted = sorted(raw_dir.glob("*.features_labels.csv.gz"))
    if not extracted:
        raise RuntimeError("Extraction completed but no per-UUID csv.gz files found.")

    return raw_dir


def load_all_uuid_files(raw_dir: Path) -> pd.DataFrame:
    files = sorted(raw_dir.glob("*.features_labels.csv.gz"))
    if not files:
        raise FileNotFoundError(f"No input files in {raw_dir}")

    frames: List[pd.DataFrame] = []
    for i, file_path in enumerate(files, start=1):
        df_part = pd.read_csv(file_path, compression="gzip")
        # Insert identifiers in one concat to avoid fragmentation warnings.
        id_df = pd.DataFrame(
            {
                "uuid": [file_path.name.split(".")[0]] * len(df_part),
                "source_file": [file_path.name] * len(df_part),
            }
        )
        frames.append(pd.concat([df_part, id_df], axis=1))

        if i % 10 == 0 or i == len(files):
            print(f"Loaded {i}/{len(files)} files")

    df = pd.concat(frames, ignore_index=True, sort=False)
    print(f"Concatenated DataFrame shape: {df.shape}")
    return df


def split_feature_and_label_columns(columns: Sequence[str]) -> Tuple[List[str], List[str]]:
    label_cols = [c for c in columns if c.startswith("label:")]
    non_feature_cols = {"timestamp", "uuid", "source_file"}
    feature_cols = [c for c in columns if c not in non_feature_cols and c not in label_cols]
    return feature_cols, label_cols


def print_dataset_overview(df: pd.DataFrame, feature_cols: Sequence[str], label_cols: Sequence[str]) -> None:
    print("\n=== Dataset Overview ===")
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {df.shape[1]:,}")
    print(f"Feature columns: {len(feature_cols):,}")
    print(f"Label columns: {len(label_cols):,}")

    missing_pct = (df.isna().mean() * 100.0).sort_values(ascending=False)
    print("\nTop 30 most-missing columns (%):")
    print(missing_pct.head(30).to_string(float_format=lambda v: f"{v:6.2f}"))


def select_feature_columns(feature_cols: Sequence[str]) -> Tuple[List[str], List[str]]:
    available = set(feature_cols)
    selected = [c for c in FEATURE_WHITELIST if c in available]
    missing = [c for c in FEATURE_WHITELIST if c not in available]

    print("\n=== Feature Selection ===")
    print(f"Requested feature columns: {len(FEATURE_WHITELIST)}")
    print(f"Selected feature columns: {len(selected)}")
    if missing:
        print(f"Missing requested features: {len(missing)}")
        for col in missing:
            print(f"  - {col}")

    if not selected:
        raise RuntimeError("No requested features were found in the dataset.")

    return selected, missing


def _binary_or(df: pd.DataFrame, source_cols: List[str]) -> pd.Series:
    numeric = df.loc[:, source_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return (numeric.to_numpy() > 0.0).any(axis=1).astype(np.int8)


def build_output_labels(df: pd.DataFrame, available_label_cols: Sequence[str]) -> Dict[str, List[str]]:
    available = set(available_label_cols)
    label_sources_used: Dict[str, List[str]] = {}

    # Build all atomic labels first.
    for out_label, candidates in ATOMIC_LABEL_MAP.items():
        present = [c for c in candidates if c in available]
        if not present:
            print(f"Warning: no source columns found for label '{out_label}' -> filling zeros")
            df[out_label] = 0
            label_sources_used[out_label] = []
            continue

        df[out_label] = _binary_or(df, present)
        label_sources_used[out_label] = present

    # Derived labels.
    stationary_sources = ["Sitting", "Standing", "LyingDown", "Sleeping"]
    moving_sources = ["Walking", "Strolling", "Running", "Bicycling", "Exercising", "Stairs"]

    df["Stationary"] = (df[stationary_sources].to_numpy() > 0).any(axis=1).astype(np.int8)
    df["Moving"] = (df[moving_sources].to_numpy() > 0).any(axis=1).astype(np.int8)

    label_sources_used["Stationary"] = stationary_sources
    label_sources_used["Moving"] = moving_sources

    return label_sources_used


def compute_label_positive_rates(df: pd.DataFrame, label_cols: Sequence[str]) -> Dict[str, float]:
    return {col: float(df[col].mean() * 100.0) for col in label_cols}


def build_stratify_key(df: pd.DataFrame, label_cols: Sequence[str], label_sum_cap: int = 3) -> pd.Series:
    label_matrix = df.loc[:, label_cols].to_numpy(dtype=np.int16)
    label_counts = label_matrix.sum(axis=1)
    capped_counts = np.minimum(label_counts, label_sum_cap)

    has_positive = label_counts > 0
    top_indices = label_matrix.argmax(axis=1)
    label_names = np.array(label_cols, dtype=object)
    top_label_names = np.where(has_positive, label_names[top_indices], "none")

    return pd.Series(capped_counts.astype(str), index=df.index) + "_" + pd.Series(top_label_names, index=df.index)


def make_safe_stratify_values(values: Sequence[str]) -> np.ndarray | None:
    series = pd.Series(values)
    counts = series.value_counts()
    merged = series.where(series.map(counts) >= 2, "__rare__")
    merged_counts = merged.value_counts()

    if merged.nunique() < 2:
        return None
    if merged_counts.min() < 2:
        return None
    return merged.to_numpy()


def split_with_fallback(
    items: np.ndarray,
    test_size: float,
    random_state: int,
    stratify_values: Sequence[str] | None,
) -> Tuple[np.ndarray, np.ndarray]:
    stratify = None if stratify_values is None else make_safe_stratify_values(stratify_values)
    try:
        a, b = train_test_split(
            items,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=stratify,
        )
        return np.array(a), np.array(b)
    except ValueError as exc:
        print(f"Warning: stratified split failed ({exc}); falling back to random split.")
        a, b = train_test_split(
            items,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=None,
        )
        return np.array(a), np.array(b)


def split_dataframe(df: pd.DataFrame, label_cols: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["stratify_key"] = build_stratify_key(df, label_cols)

    if "uuid" in df.columns:
        print("\nSplitting by UUID to avoid subject leakage.")
        uuid_key = (
            df.groupby("uuid", sort=False)["stratify_key"]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
        )

        uuids = uuid_key.index.to_numpy()
        keys = uuid_key.to_numpy()

        train_uuids, temp_uuids = split_with_fallback(
            uuids,
            test_size=0.30,
            random_state=RANDOM_SEED,
            stratify_values=keys,
        )

        temp_keys = uuid_key.loc[temp_uuids].to_numpy()
        val_uuids, test_uuids = split_with_fallback(
            temp_uuids,
            test_size=0.50,
            random_state=RANDOM_SEED,
            stratify_values=temp_keys,
        )

        train_df = df[df["uuid"].isin(set(train_uuids))].copy()
        val_df = df[df["uuid"].isin(set(val_uuids))].copy()
        test_df = df[df["uuid"].isin(set(test_uuids))].copy()
    else:
        print("\nUUID column not found; splitting by rows.")
        row_idx = df.index.to_numpy()
        row_keys = df["stratify_key"].to_numpy()

        train_idx, temp_idx = split_with_fallback(
            row_idx,
            test_size=0.30,
            random_state=RANDOM_SEED,
            stratify_values=row_keys,
        )

        temp_keys = df.loc[temp_idx, "stratify_key"].to_numpy()
        val_idx, test_idx = split_with_fallback(
            temp_idx,
            test_size=0.50,
            random_state=RANDOM_SEED,
            stratify_values=temp_keys,
        )

        train_df = df.loc[train_idx].copy()
        val_df = df.loc[val_idx].copy()
        test_df = df.loc[test_idx].copy()

    for split_df in (train_df, val_df, test_df):
        split_df.drop(columns=["stratify_key"], inplace=True, errors="ignore")
        split_df.reset_index(drop=True, inplace=True)

    return train_df, val_df, test_df


def save_outputs(
    project_root: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    selected_feature_cols: Sequence[str],
    final_label_cols: Sequence[str],
    label_sources_used: Dict[str, List[str]],
    missing_features: Sequence[str],
) -> None:
    out_dir = project_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    id_cols = [c for c in ["uuid", "source_file", "timestamp"] if c in train_df.columns]
    ordered_cols = id_cols + list(selected_feature_cols) + list(final_label_cols)

    train_out = out_dir / "train.csv"
    val_out = out_dir / "val.csv"
    test_out = out_dir / "test.csv"

    train_df.loc[:, ordered_cols].to_csv(train_out, index=False)
    val_df.loc[:, ordered_cols].to_csv(val_out, index=False)
    test_df.loc[:, ordered_cols].to_csv(test_out, index=False)

    full_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
    label_freq = {
        label: {
            "positive_count": int(full_df[label].sum()),
            "positive_rate_pct": float(full_df[label].mean() * 100.0),
        }
        for label in final_label_cols
    }

    metadata = {
        "selected_feature_columns": list(selected_feature_cols),
        "missing_requested_features": list(missing_features),
        "final_label_columns": list(final_label_cols),
        "label_source_mapping": label_sources_used,
        "sampling_window_assumptions": {
            "sampling_rate_hz": 20,
            "window_size_seconds": 5,
            "stride_seconds": 2,
            "note": "ExtraSensory file contains precomputed window-level features.",
        },
        "label_frequencies": label_freq,
        "splits": {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
        },
        "random_seed": RANDOM_SEED,
    }

    metadata_out = out_dir / "metadata.json"
    metadata_out.write_text(json.dumps(metadata, indent=2))

    print("\nSaved files:")
    print(f"  - {train_out}")
    print(f"  - {val_out}")
    print(f"  - {test_out}")
    print(f"  - {metadata_out}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    raw_dir = ensure_extracted(project_root)
    df = load_all_uuid_files(raw_dir)

    feature_cols, raw_label_cols = split_feature_and_label_columns(df.columns.tolist())
    print_dataset_overview(df, feature_cols, raw_label_cols)

    selected_feature_cols, missing_features = select_feature_columns(feature_cols)

    label_sources_used = build_output_labels(df, raw_label_cols)
    final_label_cols = list(FINAL_LABEL_ORDER)

    # Drop rows where all selected features are missing, then fill remaining NaNs.
    before_drop = len(df)
    all_missing = df.loc[:, selected_feature_cols].isna().all(axis=1)
    df = df.loc[~all_missing].copy()
    dropped_rows = before_drop - len(df)
    print(f"\nDropped rows with all selected features missing: {dropped_rows}")

    df.loc[:, selected_feature_cols] = df.loc[:, selected_feature_cols].fillna(0.0)
    for lbl in final_label_cols:
        df[lbl] = pd.to_numeric(df[lbl], errors="coerce").fillna(0).astype(np.int8)

    id_cols = [c for c in ["uuid", "source_file", "timestamp"] if c in df.columns]
    keep_cols = id_cols + selected_feature_cols + final_label_cols
    final_df = df.loc[:, keep_cols].copy()

    train_df, val_df, test_df = split_dataframe(final_df, final_label_cols)

    save_outputs(
        project_root=project_root,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        selected_feature_cols=selected_feature_cols,
        final_label_cols=final_label_cols,
        label_sources_used=label_sources_used,
        missing_features=missing_features,
    )

    print("\n=== Final Summary ===")
    print(f"Train shape: {train_df.shape}")
    print(f"Val shape:   {val_df.shape}")
    print(f"Test shape:  {test_df.shape}")
    print(f"Final number of features: {len(selected_feature_cols)}")
    print(f"Final number of labels: {len(final_label_cols)}")

    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    rates = compute_label_positive_rates(full_df, final_label_cols)
    print("Final label positive rates (%):")
    for col in final_label_cols:
        print(f"  - {col}: {rates[col]:.3f}%")


if __name__ == "__main__":
    main()

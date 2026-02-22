"""Dataset helpers for NotifSense.

Includes:
- ProcessedCSVDataset: loads prepared CSV splits from scripts/prepare_data.py
- DummyMultiLabelDataset: synthetic fallback for simple smoke checks
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ProcessedCSVDataset(Dataset):
    """Dataset backed by prepared CSV files.

    Expected CSV schema:
    - optional identifiers: uuid, source_file, timestamp
    - selected feature columns from metadata.json
    - final label columns from metadata.json

    Model input shape returned per sample:
    - features: (1, num_features)
      We treat the engineered feature vector as a 1-channel sequence so the
      existing Conv1d+LSTM pipeline can be exercised without changing model code.
    - labels: (num_labels,)
    """

    def __init__(
        self,
        csv_path: str,
        feature_columns: List[str],
        label_columns: List[str],
    ) -> None:
        super().__init__()
        self.csv_path = Path(csv_path)
        self.feature_columns = list(feature_columns)
        self.label_columns = list(label_columns)

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        required_cols = self.feature_columns + self.label_columns
        df = pd.read_csv(self.csv_path, usecols=required_cols)

        missing_in_file = [c for c in required_cols if c not in df.columns]
        if missing_in_file:
            raise ValueError(
                f"Required columns missing in {self.csv_path}: {missing_in_file[:10]}"
            )

        features_np = df[self.feature_columns].to_numpy(dtype=np.float32, copy=True)
        labels_np = df[self.label_columns].to_numpy(dtype=np.float32, copy=True)

        # Enforce binary targets from any residual non-binary values.
        labels_np = (labels_np > 0.0).astype(np.float32)

        # Conv1d expects (channels, timesteps) per sample.
        # Use one channel and feature dimension as pseudo-timesteps.
        self.features = torch.from_numpy(features_np).unsqueeze(1)
        self.labels = torch.from_numpy(labels_np)

    def __len__(self) -> int:
        return self.features.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class DummyMultiLabelDataset(Dataset):
    """Synthetic dataset for validating the training pipeline end-to-end."""

    def __init__(
        self,
        num_samples: int,
        input_channels: int,
        timesteps: int,
        num_labels: int,
    ) -> None:
        super().__init__()
        self.features = torch.randn(num_samples, input_channels, timesteps)
        self.labels = torch.randint(0, 2, (num_samples, num_labels)).float()

    def __len__(self) -> int:
        return self.features.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def load_processed_metadata(metadata_path: str) -> Dict[str, object]:
    """Load metadata emitted by scripts/prepare_data.py."""
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata not found: {path}")

    return json.loads(path.read_text())

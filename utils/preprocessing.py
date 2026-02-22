"""Preprocessing utilities for sensor time-series."""

from typing import Optional, Tuple

import numpy as np


def zscore_normalize(
    data: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize data using z-score normalization.

    Args:
        data: Array with shape (..., channels).
        mean: Optional precomputed mean for each channel.
        std: Optional precomputed std for each channel.
        eps: Small constant to avoid divide-by-zero.

    Returns:
        normalized_data, mean, std
    """
    if mean is None:
        mean = data.mean(axis=0)
    if std is None:
        std = data.std(axis=0)

    normalized = (data - mean) / (std + eps)
    return normalized, mean, std

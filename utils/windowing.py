"""Sliding-window utilities for time-series data."""

from typing import Tuple

import numpy as np
import torch


def create_sliding_windows(
    data,
    labels,
    sampling_rate: int,
    window_sec: int,
    stride_sec: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert continuous time-series data into overlapping windows.

    Args:
        data: Time-series array/tensor with shape (timesteps, channels).
        labels: Label array/tensor. Supported shapes:
            - (timesteps, num_labels): per-timestep multi-label annotations.
            - (timesteps,): per-timestep binary/ordinal labels.
            - (num_labels,): one multi-label target used for all windows.
        sampling_rate: Samples per second.
        window_sec: Window length in seconds.
        stride_sec: Window stride in seconds.

    Returns:
        A tuple (windowed_data, windowed_labels):
        - windowed_data: shape (num_windows, channels, window_length)
        - windowed_labels: shape (num_windows, num_labels) or (num_windows,)
    """
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be > 0")
    if window_sec <= 0 or stride_sec <= 0:
        raise ValueError("window_sec and stride_sec must be > 0")

    window_len = int(sampling_rate * window_sec)
    stride_len = int(sampling_rate * stride_sec)

    if window_len <= 0 or stride_len <= 0:
        raise ValueError("window length and stride length must be positive")

    data_np = data.detach().cpu().numpy() if torch.is_tensor(data) else np.asarray(data)
    labels_np = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)

    if data_np.ndim != 2:
        raise ValueError("data must have shape (timesteps, channels)")

    total_timesteps, _ = data_np.shape
    if total_timesteps < window_len:
        raise ValueError("data is shorter than one full window")

    windows = []
    window_labels = []

    for start in range(0, total_timesteps - window_len + 1, stride_len):
        end = start + window_len

        # Keep model-ready shape: (channels, timesteps).
        windows.append(data_np[start:end].T)

        if labels_np.ndim == 2 and labels_np.shape[0] == total_timesteps:
            # Aggregate per-timestep multi-label annotations to one target/window.
            label_window = labels_np[start:end].max(axis=0)
        elif labels_np.ndim == 1 and labels_np.shape[0] == total_timesteps:
            # Aggregate scalar per-timestep labels.
            label_window = labels_np[start:end].max()
        else:
            # Assume already window-level static label.
            label_window = labels_np

        window_labels.append(label_window)

    windowed_data = torch.tensor(np.stack(windows), dtype=torch.float32)
    windowed_labels = torch.tensor(np.stack(window_labels), dtype=torch.float32)

    return windowed_data, windowed_labels

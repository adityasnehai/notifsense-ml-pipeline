"""MLP model for NotifSense multi-label context recognition.

An MLP is a strong baseline for precomputed tabular sensor features because it:
- Learns non-linear interactions across engineered features.
- Trains faster than sequence models on fixed-length feature vectors.
- Is straightforward to deploy and optimize later.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class NotifSenseMLP(nn.Module):
    """Multi-label MLP that returns raw logits (no sigmoid in forward)."""

    def __init__(self, input_dim: int, num_labels: int) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits for BCEWithLogitsLoss."""
        return self.net(x)

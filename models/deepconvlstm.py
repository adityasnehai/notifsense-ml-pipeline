"""DeepConvLSTM model definition for multi-label activity recognition."""

import torch
import torch.nn as nn


class DeepConvLSTM(nn.Module):
    """
    A compact DeepConvLSTM architecture for sequence classification.

    Input shape:
        (batch_size, input_channels, timesteps)

    Architecture:
    1. Temporal feature extractor with Conv1d layers.
       - Conv1d(input_channels -> 64, kernel_size=5)
       - ReLU
       - Conv1d(64 -> 128, kernel_size=5)
       - ReLU
    2. Sequence modeling with LSTM.
       - LSTM(input_size=128, hidden_size=128, batch_first=True)
    3. Classification head.
       - Linear(128 -> num_labels)

    Notes:
    - The model returns raw logits for each label.
    - Sigmoid must be applied outside the model for probabilities.
    - BCEWithLogitsLoss should be used during training.
    """

    def __init__(self, input_channels: int, num_labels: int) -> None:
        super().__init__()

        # Convolutional backbone for local temporal pattern extraction.
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5)
        self.relu2 = nn.ReLU()

        # LSTM to model longer temporal dependencies after convolutional encoding.
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)

        # Final classifier that maps sequence representation to multi-label logits.
        self.classifier = nn.Linear(in_features=128, out_features=num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_channels, timesteps).

        Returns:
            Logits tensor of shape (batch_size, num_labels).
        """
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))

        # LSTM expects (batch_size, sequence_length, feature_dim).
        x = x.transpose(1, 2)

        # Sequence output shape is (batch_size, sequence_length, hidden_size).
        sequence_output, _ = self.lstm(x)

        # Use the last timestep as the sequence summary for classification.
        final_timestep = sequence_output[:, -1, :]
        logits = self.classifier(final_timestep)
        return logits

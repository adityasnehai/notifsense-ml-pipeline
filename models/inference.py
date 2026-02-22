"""Inference utilities for DeepConvLSTM."""

from pathlib import Path
from typing import Optional

import torch
import yaml

from models.deepconvlstm import DeepConvLSTM


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class NotifSenseInference:
    """Thin inference wrapper around DeepConvLSTM."""

    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        device: str = "auto",
        num_labels: int = 12,
        input_channels: int = 6,
    ) -> None:
        if config_path:
            cfg = yaml.safe_load(Path(config_path).read_text())
            num_labels = cfg.get("num_labels", num_labels)
            input_channels = cfg.get("input_channels", input_channels)

        self.device = _resolve_device(device)
        self.model = DeepConvLSTM(input_channels=input_channels, num_labels=num_labels).to(self.device)

        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, batch: torch.Tensor) -> torch.Tensor:
        """Return per-label probabilities for a batch of inputs."""
        batch = batch.to(self.device)
        logits = self.model(batch)
        probs = torch.sigmoid(logits)
        return probs

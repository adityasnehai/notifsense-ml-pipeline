"""Inference utilities for DeepConvLSTM.

This module supports both raw state_dict checkpoints and wrapped checkpoints
that store metadata under keys like `model_state_dict`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.deepconvlstm import DeepConvLSTM


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _extract_state_dict(checkpoint_obj: Any) -> Dict[str, torch.Tensor]:
    """Extract a PyTorch state_dict from common checkpoint formats."""
    if isinstance(checkpoint_obj, dict):
        if "model_state_dict" in checkpoint_obj:
            return checkpoint_obj["model_state_dict"]
        if "state_dict" in checkpoint_obj:
            return checkpoint_obj["state_dict"]
        if all(isinstance(k, str) for k in checkpoint_obj.keys()) and any(
            k.endswith(".weight") or k.endswith(".bias") for k in checkpoint_obj.keys()
        ):
            return checkpoint_obj
    raise ValueError("Unsupported checkpoint format for DeepConvLSTM inference.")


def _looks_like_mlp_state_dict(state_dict: Dict[str, torch.Tensor]) -> bool:
    return any(k.startswith("net.") for k in state_dict.keys())


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
            num_labels = int(cfg.get("num_labels", num_labels))
            input_channels = int(cfg.get("input_channels", input_channels))

        self.device = _resolve_device(device)

        checkpoint_obj = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint_obj, dict):
            # Checkpoint metadata is the safest source of architecture dims.
            num_labels = int(checkpoint_obj.get("num_labels", num_labels))
            input_channels = int(checkpoint_obj.get("input_channels", input_channels))

        state_dict = _extract_state_dict(checkpoint_obj)
        if _looks_like_mlp_state_dict(state_dict):
            raise ValueError(
                "Checkpoint appears to be an MLP checkpoint (keys like 'net.*'). "
                "Use a DeepConvLSTM checkpoint with this inference wrapper."
            )

        self.model = DeepConvLSTM(input_channels=input_channels, num_labels=num_labels).to(self.device)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, batch: torch.Tensor) -> torch.Tensor:
        """Return per-label probabilities for a batch of inputs."""
        batch = batch.to(self.device)
        logits = self.model(batch)
        probs = torch.sigmoid(logits)
        return probs

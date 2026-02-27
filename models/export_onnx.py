"""Export DeepConvLSTM PyTorch weights to ONNX.

Supports both raw state_dict checkpoints and wrapped checkpoints with
`model_state_dict` / `state_dict`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.deepconvlstm import DeepConvLSTM


def _extract_state_dict(checkpoint_obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint_obj, dict):
        if "model_state_dict" in checkpoint_obj:
            return checkpoint_obj["model_state_dict"]
        if "state_dict" in checkpoint_obj:
            return checkpoint_obj["state_dict"]
        if all(isinstance(k, str) for k in checkpoint_obj.keys()) and any(
            k.endswith(".weight") or k.endswith(".bias") for k in checkpoint_obj.keys()
        ):
            return checkpoint_obj
    raise ValueError("Unsupported checkpoint format for DeepConvLSTM ONNX export.")


def _looks_like_mlp_state_dict(state_dict: Dict[str, torch.Tensor]) -> bool:
    return any(k.startswith("net.") for k in state_dict.keys())


def export_to_onnx(config_path: str, checkpoint_path: str, output_path: str, opset: int = 13) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())

    checkpoint_obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(checkpoint_obj)

    if _looks_like_mlp_state_dict(state_dict):
        raise ValueError(
            "Checkpoint appears to be an MLP checkpoint (keys like 'net.*'). "
            "Use the MLP export flow for this checkpoint, not DeepConvLSTM ONNX export."
        )

    input_channels = int(cfg["input_channels"])
    num_labels = int(cfg["num_labels"])
    if isinstance(checkpoint_obj, dict):
        num_labels = int(checkpoint_obj.get("num_labels", num_labels))
        input_channels = int(checkpoint_obj.get("input_channels", input_channels))

    model = DeepConvLSTM(input_channels=input_channels, num_labels=num_labels)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    timesteps = int(cfg["sampling_rate"]) * int(cfg["window_size_seconds"])
    dummy_input = torch.randn(1, input_channels, timesteps)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "timesteps"},
            "logits": {0: "batch_size"},
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export DeepConvLSTM checkpoint to ONNX")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt model checkpoint")
    parser.add_argument("--output", default="models/deepconvlstm.onnx", help="Output ONNX file path")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version")
    args = parser.parse_args()

    export_to_onnx(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        opset=args.opset,
    )
    print(f"ONNX export completed: {args.output}")

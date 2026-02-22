"""Export DeepConvLSTM PyTorch weights to ONNX."""

import argparse
from pathlib import Path

import torch
import yaml

from models.deepconvlstm import DeepConvLSTM


def export_to_onnx(config_path: str, checkpoint_path: str, output_path: str, opset: int = 13) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())

    model = DeepConvLSTM(
        input_channels=cfg["input_channels"],
        num_labels=cfg["num_labels"],
    )
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    timesteps = cfg["sampling_rate"] * cfg["window_size_seconds"]
    dummy_input = torch.randn(1, cfg["input_channels"], timesteps)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
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

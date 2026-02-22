"""PyTorch-only mobile export pipeline for NotifSense MLP.

This script intentionally avoids TensorFlow/TFLite and focuses on a
PyTorch Android deployment flow:
1) Load trained PyTorch checkpoint.
2) Apply dynamic INT8 quantization to Linear layers.
3) Evaluate original and quantized models on a CSV split.
4) Benchmark inference latency on CPU.
5) Export quantized model to TorchScript and Lite Interpreter (.ptl).
6) Optionally delete legacy ONNX/TFLite artifacts.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.mobile_optimizer import optimize_for_mobile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.mlp_model import NotifSenseMLP


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024.0 * 1024.0)


def load_checkpoint(checkpoint_path: Path) -> Dict[str, object]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def build_model_from_checkpoint(ckpt: Dict[str, object]) -> NotifSenseMLP:
    input_dim = int(ckpt["input_dim"])
    num_labels = int(ckpt["num_labels"])

    model = NotifSenseMLP(input_dim=input_dim, num_labels=num_labels)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def apply_feature_standardization(x: np.ndarray, ckpt: Dict[str, object]) -> np.ndarray:
    """Apply training-time feature normalization if present in checkpoint."""
    mean = ckpt.get("feature_mean")
    std = ckpt.get("feature_std")
    if mean is None or std is None:
        return x

    mean_np = np.asarray(mean, dtype=np.float32)
    std_np = np.asarray(std, dtype=np.float32)
    std_np = np.where(std_np < 1e-6, 1.0, std_np)
    return ((x - mean_np) / std_np).astype(np.float32)


def load_eval_arrays(
    csv_path: Path,
    feature_columns: Sequence[str],
    label_columns: Sequence[str],
    ckpt: Dict[str, object],
) -> Tuple[np.ndarray, np.ndarray]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Evaluation CSV not found: {csv_path}")

    usecols = list(feature_columns) + list(label_columns)
    df = pd.read_csv(csv_path, usecols=usecols)

    x = df[list(feature_columns)].to_numpy(dtype=np.float32, copy=True)
    x = apply_feature_standardization(x, ckpt)

    y = df[list(label_columns)].to_numpy(dtype=np.float32, copy=True)
    y = (y > 0.0).astype(np.int32)
    return x, y


def get_threshold_vector(label_columns: Sequence[str], ckpt: Dict[str, object]) -> np.ndarray:
    thresholds = ckpt.get("thresholds")
    if isinstance(thresholds, dict):
        return np.array([float(thresholds.get(lbl, 0.5)) for lbl in label_columns], dtype=np.float32)
    return np.full(len(label_columns), 0.5, dtype=np.float32)


def evaluate_model(
    model: nn.Module,
    x: np.ndarray,
    y_true: np.ndarray,
    thresholds: np.ndarray,
    batch_size: int = 1024,
) -> Tuple[float, float]:
    model.eval()
    probs_chunks: List[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = start + batch_size
            xb = torch.from_numpy(x[start:end])
            logits = model(xb)
            probs_chunks.append(torch.sigmoid(logits).cpu().numpy())

    y_prob = np.concatenate(probs_chunks, axis=0)
    y_pred = (y_prob >= thresholds.reshape(1, -1)).astype(np.int32)

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    return macro_f1, micro_f1


def benchmark_model(model: nn.Module, input_dim: int, runs: int = 1000, warmup: int = 100) -> float:
    model.eval()
    dummy = torch.randn(1, input_dim)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)

        start = time.perf_counter()
        for _ in range(runs):
            _ = model(dummy)
        elapsed = time.perf_counter() - start

    return (elapsed / runs) * 1000.0


def save_quantized_checkpoint(
    quantized_model: nn.Module,
    ckpt: Dict[str, object],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "quantized_model": quantized_model,
            "input_dim": ckpt["input_dim"],
            "num_labels": ckpt["num_labels"],
            "label_columns": ckpt.get("label_columns", []),
            "feature_columns": ckpt.get("feature_columns", []),
            "thresholds": ckpt.get("thresholds", {}),
            "feature_mean": ckpt.get("feature_mean"),
            "feature_std": ckpt.get("feature_std"),
        },
        out_path,
    )


def export_torchscript_mobile(
    quantized_model: nn.Module,
    input_dim: int,
    scripted_out: Path,
    lite_out: Path,
) -> nn.Module:
    """Export quantized model to TorchScript and Lite Interpreter format."""
    scripted_out.parent.mkdir(parents=True, exist_ok=True)
    lite_out.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, input_dim)
    quantized_model.eval()

    with torch.no_grad():
        traced = torch.jit.trace(quantized_model, dummy)
        traced = torch.jit.freeze(traced)

    traced.save(str(scripted_out))

    mobile_optimized = optimize_for_mobile(traced)
    mobile_optimized._save_for_lite_interpreter(str(lite_out))

    return mobile_optimized


def remove_legacy_artifacts(paths: Sequence[Path]) -> List[Path]:
    removed: List[Path] = []
    for path in paths:
        if path.exists() and path.is_file():
            path.unlink()
            removed.append(path)
    return removed


def main(args: argparse.Namespace) -> None:
    checkpoint_path = Path(args.checkpoint)
    eval_csv = Path(args.eval_csv)
    quantized_out = Path(args.quantized_out)
    scripted_out = Path(args.scripted_out)
    lite_out = Path(args.lite_out)

    ckpt = load_checkpoint(checkpoint_path)
    feature_columns = list(ckpt.get("feature_columns", []))
    label_columns = list(ckpt.get("label_columns", []))

    if not feature_columns or not label_columns:
        raise RuntimeError("Checkpoint must contain feature_columns and label_columns metadata.")

    input_dim = int(ckpt["input_dim"])

    if args.remove_legacy_artifacts:
        legacy = [Path("models/notifsense_quantized.onnx"), Path("models/notifsense_mobile.onnx")]
        removed = remove_legacy_artifacts(legacy)
        if removed:
            print("Removed legacy artifacts:")
            for p in removed:
                print(f"  - {p}")

    model = build_model_from_checkpoint(ckpt)

    original_size = file_size_mb(checkpoint_path)
    print(f"Original checkpoint: {checkpoint_path}")
    print(f"Original size: {original_size:.4f} MB")

    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )
    quantized_model.eval()

    save_quantized_checkpoint(quantized_model, ckpt, quantized_out)
    quantized_size = file_size_mb(quantized_out)
    print(f"Quantized checkpoint: {quantized_out}")
    print(f"Quantized size: {quantized_size:.4f} MB")

    mobile_model = export_torchscript_mobile(
        quantized_model=quantized_model,
        input_dim=input_dim,
        scripted_out=scripted_out,
        lite_out=lite_out,
    )
    scripted_size = file_size_mb(scripted_out)
    lite_size = file_size_mb(lite_out)
    print(f"TorchScript model: {scripted_out} ({scripted_size:.4f} MB)")
    print(f"Lite Interpreter model: {lite_out} ({lite_size:.4f} MB)")

    x_eval, y_eval = load_eval_arrays(eval_csv, feature_columns, label_columns, ckpt)
    thresholds = get_threshold_vector(label_columns, ckpt)

    orig_macro, orig_micro = evaluate_model(model, x_eval, y_eval, thresholds, batch_size=args.eval_batch_size)
    quant_macro, quant_micro = evaluate_model(
        quantized_model,
        x_eval,
        y_eval,
        thresholds,
        batch_size=args.eval_batch_size,
    )
    mobile_macro, mobile_micro = evaluate_model(
        mobile_model,
        x_eval,
        y_eval,
        thresholds,
        batch_size=args.eval_batch_size,
    )

    orig_ms = benchmark_model(model, input_dim=input_dim, runs=args.runs)
    quant_ms = benchmark_model(quantized_model, input_dim=input_dim, runs=args.runs)
    mobile_ms = benchmark_model(mobile_model, input_dim=input_dim, runs=args.runs)

    print("\n=== Evaluation (PyTorch only) ===")
    print(f"Labels: {len(label_columns)}")
    print(f"Input features: {input_dim}")
    print(f"Original - Macro F1: {orig_macro:.4f}, Micro F1: {orig_micro:.4f}")
    print(f"Quantized - Macro F1: {quant_macro:.4f}, Micro F1: {quant_micro:.4f}")
    print(f"Mobile TS - Macro F1: {mobile_macro:.4f}, Micro F1: {mobile_micro:.4f}")

    print("\n=== Latency (CPU, ms/sample) ===")
    print(f"Original:  {orig_ms:.6f}")
    print(f"Quantized: {quant_ms:.6f}")
    print(f"Mobile TS: {mobile_ms:.6f}")

    if orig_ms > 0:
        print(f"Quantized speedup vs original: {(orig_ms / quant_ms):.3f}x")
        print(f"Mobile TS speedup vs original: {(orig_ms / mobile_ms):.3f}x")

    print("\n=== Summary ===")
    print(f"Original size (MB): {original_size:.4f}")
    print(f"Quantized size (MB): {quantized_size:.4f}")
    print(f"TorchScript size (MB): {scripted_size:.4f}")
    print(f"Lite model size (MB): {lite_size:.4f}")
    print(f"Macro F1 drop (quantized): {(orig_macro - quant_macro):.4f}")
    print(f"Micro F1 drop (quantized): {(orig_micro - quant_micro):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch-only Android export for NotifSense")
    parser.add_argument("--checkpoint", default="models/notifsense_final.pt", help="Path to trained checkpoint")
    parser.add_argument("--eval-csv", default="data/processed/test.csv", help="CSV split used for evaluation")
    parser.add_argument("--quantized-out", default="models/notifsense_quantized.pt", help="Path to save quantized checkpoint")
    parser.add_argument("--scripted-out", default="models/notifsense_quantized_scripted.pt", help="Path to save TorchScript model")
    parser.add_argument("--lite-out", default="models/notifsense_android.ptl", help="Path to save Lite Interpreter model")
    parser.add_argument("--runs", type=int, default=1000, help="Number of latency benchmark iterations")
    parser.add_argument("--eval-batch-size", type=int, default=512, help="Batch size used during metric evaluation")
    parser.add_argument(
        "--remove-legacy-artifacts",
        action="store_true",
        help="Delete legacy ONNX/TFLite outputs from models/ if present",
    )
    main(parser.parse_args())

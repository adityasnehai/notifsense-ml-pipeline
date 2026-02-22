"""Evaluate a DeepConvLSTM checkpoint on processed multi-label CSV data.

This script is intentionally strict about checkpoint compatibility so we do not
accidentally evaluate an MLP checkpoint with the DeepConvLSTM architecture.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.deepconvlstm import DeepConvLSTM
from utils.dataset import ProcessedCSVDataset, load_processed_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DeepConvLSTM on test split")
    parser.add_argument("--config", default="configs/config.yaml", help="YAML config path")
    parser.add_argument(
        "--checkpoint",
        default="models/deepconvlstm_best.pt",
        help="Path to DeepConvLSTM checkpoint",
    )
    parser.add_argument(
        "--metadata",
        default="data/processed/metadata.json",
        help="Path to processed metadata JSON",
    )
    parser.add_argument(
        "--split-csv",
        default="data/processed/test.csv",
        help="CSV split to evaluate (test/val/train)",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--output",
        default="data/processed/deepconvlstm_eval.json",
        help="Path to save structured evaluation results",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return yaml.safe_load(path.read_text())


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def extract_state_dict(ckpt_obj: object) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict):
        if "model_state_dict" in ckpt_obj:
            return ckpt_obj["model_state_dict"]
        if "state_dict" in ckpt_obj:
            return ckpt_obj["state_dict"]
        # Raw state_dict case.
        if all(isinstance(k, str) for k in ckpt_obj.keys()) and any(
            k.endswith(".weight") or k.endswith(".bias") for k in ckpt_obj.keys()
        ):
            return ckpt_obj
    raise ValueError("Checkpoint does not contain a valid PyTorch state_dict")


def checkpoint_looks_like_mlp(state_dict: Dict[str, torch.Tensor]) -> bool:
    # Current MLP checkpoints in this project use keys like net.0.weight.
    return any(k.startswith("net.") for k in state_dict.keys())


def load_thresholds(
    checkpoint_obj: object,
    label_columns: List[str],
    default_threshold: float,
) -> np.ndarray:
    if isinstance(checkpoint_obj, dict) and "thresholds" in checkpoint_obj:
        raw = checkpoint_obj["thresholds"]
        if isinstance(raw, dict):
            vals = [float(raw.get(lbl, default_threshold)) for lbl in label_columns]
            return np.array(vals, dtype=np.float32)
    return np.full((len(label_columns),), float(default_threshold), dtype=np.float32)


def collect_predictions(
    model: DeepConvLSTM,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true_all: List[np.ndarray] = []
    y_prob_all: List[np.ndarray] = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.sigmoid(logits)

            y_true_all.append(batch_y.numpy().astype(np.int32))
            y_prob_all.append(probs.cpu().numpy())

    y_true = np.concatenate(y_true_all, axis=0)
    y_prob = np.concatenate(y_prob_all, axis=0)
    return y_true, y_prob


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_columns: List[str],
) -> Dict[str, object]:
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    subset_accuracy = float(accuracy_score(y_true, y_pred))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )

    per_label = []
    for i, label in enumerate(label_columns):
        per_label.append(
            {
                "label": label,
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
        )
    per_label.sort(key=lambda row: row["f1"], reverse=True)

    mcm = multilabel_confusion_matrix(y_true, y_pred)
    confusion = []
    for i, label in enumerate(label_columns):
        tn, fp, fn, tp = mcm[i].ravel()
        confusion.append(
            {
                "label": label,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=label_columns,
        zero_division=0,
    )

    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "subset_accuracy": subset_accuracy,
        "per_label": per_label,
        "confusion_matrix_per_label": confusion,
        "classification_report": report_text,
    }


def main() -> None:
    args = parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve()
    checkpoint_path = (PROJECT_ROOT / args.checkpoint).resolve()
    metadata_path = (PROJECT_ROOT / args.metadata).resolve()
    split_csv_path = (PROJECT_ROOT / args.split_csv).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()

    cfg = load_config(config_path)
    device = resolve_device(str(cfg.get("device", "auto")))

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Tip: train and save a DeepConvLSTM checkpoint first (e.g., models/deepconvlstm_best.pt)."
        )

    metadata = load_processed_metadata(str(metadata_path))
    feature_columns = list(metadata["selected_feature_columns"])
    label_columns = list(metadata["final_label_columns"])

    dataset = ProcessedCSVDataset(
        csv_path=str(split_csv_path),
        feature_columns=feature_columns,
        label_columns=label_columns,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    checkpoint_obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = extract_state_dict(checkpoint_obj)

    if checkpoint_looks_like_mlp(state_dict):
        raise ValueError(
            "Checkpoint appears to be an MLP checkpoint (keys start with 'net.'). "
            "Please provide a DeepConvLSTM checkpoint for this evaluator."
        )

    input_channels = int(cfg.get("input_channels", 1))
    num_labels_cfg = int(cfg.get("num_labels", len(label_columns)))
    num_labels = len(label_columns)
    if num_labels_cfg != num_labels:
        print(
            f"[WARN] config num_labels={num_labels_cfg} but metadata has {num_labels}. "
            "Using metadata label count."
        )

    model = DeepConvLSTM(input_channels=input_channels, num_labels=num_labels).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print("[WARN] Non-strict checkpoint load detected")
        if missing:
            print(f"  Missing keys: {missing[:10]}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected[:10]}")

    thresholds = load_thresholds(checkpoint_obj, label_columns, args.threshold)

    y_true, y_prob = collect_predictions(model, loader, device)
    y_pred = (y_prob >= thresholds.reshape(1, -1)).astype(np.int32)

    metrics = compute_metrics(y_true, y_pred, label_columns)
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["num_samples"] = int(y_true.shape[0])
    metrics["num_features"] = int(len(feature_columns))
    metrics["num_labels"] = int(len(label_columns))
    metrics["thresholds"] = {label_columns[i]: float(thresholds[i]) for i in range(len(label_columns))}

    print("\n=== DeepConvLSTM Evaluation ===")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Samples: {metrics['num_samples']}")
    print(f"Features: {metrics['num_features']}")
    print(f"Labels: {metrics['num_labels']}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Micro F1: {metrics['micro_f1']:.4f}")
    print(f"Subset Accuracy: {metrics['subset_accuracy']:.4f}")

    print("\nPer-label metrics (sorted by F1):")
    for row in metrics["per_label"]:
        print(
            f"  - {row['label']}: "
            f"P={row['precision']:.4f}, "
            f"R={row['recall']:.4f}, "
            f"F1={row['f1']:.4f}, "
            f"Support={row['support']}"
        )

    print("\nClassification report:")
    print(metrics["classification_report"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved evaluation JSON: {output_path}")


if __name__ == "__main__":
    main()

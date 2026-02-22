"""NotifSense optimization-phase training.

This script performs two-stage training for reliability:
1) Train on full label set and evaluate per-label quality.
2) Prune weak labels by F1/support criteria.
3) Retrain on pruned labels.
4) Tune per-label thresholds and save final model.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

from models.mlp_model import NotifSenseMLP


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> Dict[str, object]:
    return yaml.safe_load(Path(config_path).read_text())


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_metadata(metadata_path: str) -> Tuple[List[str], List[str]]:
    metadata = json.loads(Path(metadata_path).read_text())
    return list(metadata["selected_feature_columns"]), list(metadata["final_label_columns"])


def load_split_arrays(
    csv_path: str,
    feature_columns: List[str],
    label_columns: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    usecols = feature_columns + label_columns
    df = pd.read_csv(csv_path, usecols=usecols)

    x = df[feature_columns].to_numpy(dtype=np.float32, copy=True)
    y = df[label_columns].to_numpy(dtype=np.float32, copy=True)
    y = (y > 0.0).astype(np.float32)
    return x, y


def compute_standardization_stats(x_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def to_tensors(x: np.ndarray, y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.from_numpy(x), torch.from_numpy(y)


def build_loader(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )


def compute_pos_weight(y_train: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    n = y_train.size(0)
    pos = y_train.sum(dim=0)
    neg = n - pos
    w = neg / torch.clamp(pos, min=eps)
    w[pos <= 0] = 1.0
    return w.float()


class FocalLoss(nn.Module):
    """Multi-label focal loss on logits with optional alpha weighting."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | None = None,
        pos_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.base_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.base_bce(logits, targets)
        probs = torch.sigmoid(logits)

        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        modulating = (1.0 - p_t).pow(self.gamma)

        if self.alpha is not None:
            alpha_factor = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_factor * modulating * bce
        else:
            loss = modulating * bce

        return loss.mean()


def get_criterion(config: Dict[str, object], pos_weight: torch.Tensor) -> nn.Module:
    use_focal = bool(config.get("use_focal_loss", False))
    if use_focal:
        gamma = float(config.get("focal_gamma", 2.0))
        alpha = config.get("focal_alpha", None)
        alpha = None if alpha is None else float(alpha)
        return FocalLoss(gamma=gamma, alpha=alpha, pos_weight=pos_weight)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_rows = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        rows = batch_x.size(0)
        total_loss += float(loss.item()) * rows
        total_rows += rows

    return total_loss / max(total_rows, 1)


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_rows = 0
    probs_all: List[np.ndarray] = []
    true_all: List[np.ndarray] = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            probs = torch.sigmoid(logits)

            rows = batch_x.size(0)
            total_loss += float(loss.item()) * rows
            total_rows += rows

            probs_all.append(probs.cpu().numpy())
            true_all.append(batch_y.cpu().numpy())

    y_prob = np.concatenate(probs_all, axis=0)
    y_true = np.concatenate(true_all, axis=0).astype(np.int32)
    return total_loss / max(total_rows, 1), y_true, y_prob


def tune_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: Sequence[str],
    t_min: float = 0.10,
    t_max: float = 0.90,
    t_step: float = 0.05,
) -> Dict[str, float]:
    """Tune threshold per label to maximize validation F1."""
    grid = np.arange(t_min, t_max + 1e-9, t_step, dtype=np.float32)
    tuned: Dict[str, float] = {}

    for idx, label in enumerate(label_names):
        yt = y_true[:, idx]
        yp = y_prob[:, idx]

        best_f1 = -1.0
        best_t = 0.5
        for t in grid:
            pred = (yp >= t).astype(np.int32)
            f1 = f1_score(yt, pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)

        tuned[label] = round(best_t, 4)

    return tuned


def predict_with_threshold_dict(
    y_prob: np.ndarray,
    label_names: Sequence[str],
    threshold_dict: Dict[str, float],
) -> np.ndarray:
    thresholds = np.array([threshold_dict[name] for name in label_names], dtype=np.float32)
    return (y_prob >= thresholds.reshape(1, -1)).astype(np.int32)


def compute_per_label_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Sequence[str],
) -> List[Dict[str, float]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )

    rows: List[Dict[str, float]] = []
    for i, name in enumerate(label_names):
        rows.append(
            {
                "label": name,
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
        )

    rows.sort(key=lambda x: x["f1"], reverse=True)
    return rows


def print_per_label_table(rows: List[Dict[str, float]]) -> None:
    print("\nPer-label metrics (sorted by F1):")
    for row in rows:
        print(
            f"  - {row['label']}: "
            f"P={row['precision']:.4f}, "
            f"R={row['recall']:.4f}, "
            f"F1={row['f1']:.4f}, "
            f"Support={row['support']}"
        )


def evaluate_summary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    micro = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    per_label = f1_score(y_true, y_pred, average=None, zero_division=0)
    avg_per_label = float(np.mean(per_label))
    return {
        "macro_f1": macro,
        "micro_f1": micro,
        "avg_per_label_f1": avg_per_label,
    }


def train_with_early_stopping(
    config: Dict[str, object],
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    lr = float(config.get("learning_rate", 1e-3))
    weight_decay = float(config.get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(config.get("lr_reduce_factor", 0.5)),
        patience=int(config.get("lr_reduce_patience", 2)),
    )

    epochs = int(config.get("epochs", 30))
    patience = int(config.get("early_stopping_patience", 5))

    best_state = copy.deepcopy(model.state_dict())
    best_macro = -1.0
    stale = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, y_val_true, y_val_prob = collect_predictions(model, val_loader, criterion, device)
        y_val_pred = (y_val_prob >= 0.5).astype(np.int32)
        val_macro = float(f1_score(y_val_true, y_val_pred, average="macro", zero_division=0))

        scheduler.step(val_macro)
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{epochs} | lr={lr_now:.6g} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | val_macro_f1={val_macro:.4f}"
        )

        if val_macro > best_macro:
            best_macro = val_macro
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    return best_state


def select_labels_to_keep(
    per_label_rows_sorted: List[Dict[str, float]],
    train_support_counts: Dict[str, int],
    dataset_size: int,
    min_f1: float,
    min_support_pct: float,
) -> Tuple[List[str], List[Dict[str, object]]]:
    to_keep: List[str] = []
    to_remove: List[Dict[str, object]] = []

    min_support_count = int(np.ceil((min_support_pct / 100.0) * dataset_size))

    rows_by_label = {row["label"]: row for row in per_label_rows_sorted}
    for label, row in rows_by_label.items():
        support = int(train_support_counts.get(label, 0))
        support_pct = (100.0 * support) / dataset_size

        reasons: List[str] = []
        if row["f1"] < min_f1:
            reasons.append(f"F1 {row['f1']:.4f} < {min_f1}")
        if support < min_support_count:
            reasons.append(f"Support {support} ({support_pct:.3f}%) < {min_support_pct:.1f}%")

        if reasons:
            to_remove.append(
                {
                    "label": label,
                    "f1": float(row["f1"]),
                    "support": support,
                    "support_pct": support_pct,
                    "reason": " | ".join(reasons),
                }
            )
        else:
            to_keep.append(label)

    return to_keep, to_remove


def save_thresholds(path: Path, thresholds: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(thresholds, indent=2, sort_keys=True))


def save_final_model(
    out_path: Path,
    model_state: Dict[str, torch.Tensor],
    input_dim: int,
    label_columns: List[str],
    feature_columns: List[str],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    thresholds: Dict[str, float],
    removed_labels: List[Dict[str, object]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model_state,
            "input_dim": input_dim,
            "num_labels": len(label_columns),
            "label_columns": label_columns,
            "feature_columns": feature_columns,
            "feature_mean": feature_mean.astype(np.float32),
            "feature_std": feature_std.astype(np.float32),
            "thresholds": thresholds,
            "removed_labels": removed_labels,
        },
        out_path,
    )


def main(config_path: str) -> None:
    config = load_config(config_path)
    set_seed(int(config.get("seed", 42)))

    device = resolve_device(str(config["device"]))

    metadata_path = str(config.get("metadata_path", "data/processed/metadata.json"))
    train_csv = str(config.get("train_csv", "data/processed/train.csv"))
    val_csv = str(config.get("val_csv", "data/processed/val.csv"))
    test_csv = str(config.get("test_csv", "data/processed/test.csv"))

    feature_columns, label_columns_all = load_metadata(metadata_path)

    x_train_np, y_train_np = load_split_arrays(train_csv, feature_columns, label_columns_all)
    x_val_np, y_val_np = load_split_arrays(val_csv, feature_columns, label_columns_all)
    x_test_np, y_test_np = load_split_arrays(test_csv, feature_columns, label_columns_all)

    mean, std = compute_standardization_stats(x_train_np)
    x_train_np = standardize(x_train_np, mean, std)
    x_val_np = standardize(x_val_np, mean, std)
    x_test_np = standardize(x_test_np, mean, std)

    x_train, y_train_all = to_tensors(x_train_np, y_train_np)
    x_val, y_val_all = to_tensors(x_val_np, y_val_np)
    x_test, y_test_all = to_tensors(x_test_np, y_test_np)

    batch_size = int(config.get("batch_size", 64))
    num_workers = int(config.get("num_workers", 0))

    print("=== Stage 1: Full-Label Training ===")
    print(f"Features: {x_train.shape[1]} | Labels: {y_train_all.shape[1]}")

    train_loader_full = build_loader(x_train, y_train_all, batch_size, shuffle=True, num_workers=num_workers)
    val_loader_full = build_loader(x_val, y_val_all, batch_size, shuffle=False, num_workers=num_workers)

    model_full = NotifSenseMLP(input_dim=x_train.shape[1], num_labels=y_train_all.shape[1]).to(device)
    print(f"Model params: {count_parameters(model_full):,}")

    pos_weight_full = compute_pos_weight(y_train_all).to(device)
    criterion_full = get_criterion(config, pos_weight_full)

    best_state_full = train_with_early_stopping(
        config=config,
        model=model_full,
        train_loader=train_loader_full,
        val_loader=val_loader_full,
        criterion=criterion_full,
        device=device,
    )
    model_full.load_state_dict(best_state_full)

    val_loss_full, y_val_true_full, y_val_prob_full = collect_predictions(
        model_full, val_loader_full, criterion_full, device
    )
    thresholds_full = tune_thresholds(y_val_true_full, y_val_prob_full, label_columns_all)
    y_val_pred_full = predict_with_threshold_dict(y_val_prob_full, label_columns_all, thresholds_full)

    per_label_full = compute_per_label_metrics(y_val_true_full, y_val_pred_full, label_columns_all)
    print(f"Validation loss (stage-1): {val_loss_full:.6f}")
    print_per_label_table(per_label_full)

    # Label pruning criteria.
    f1_threshold = float(config.get("prune_f1_threshold", 0.35))
    support_pct_threshold = float(config.get("prune_support_pct_threshold", 2.0))

    train_support_counts = {
        label: int(y_train_all[:, idx].sum().item())
        for idx, label in enumerate(label_columns_all)
    }

    labels_keep, labels_remove = select_labels_to_keep(
        per_label_rows_sorted=per_label_full,
        train_support_counts=train_support_counts,
        dataset_size=len(x_train),
        min_f1=f1_threshold,
        min_support_pct=support_pct_threshold,
    )

    print("\n=== Label Selection ===")
    print(f"Labels to keep ({len(labels_keep)}):")
    for name in labels_keep:
        print(f"  - {name}")

    print(f"\nLabels to remove ({len(labels_remove)}):")
    for row in labels_remove:
        print(
            f"  - {row['label']}: F1={row['f1']:.4f}, "
            f"Support={row['support']} ({row['support_pct']:.3f}%), Reason={row['reason']}"
        )

    if not labels_keep:
        raise RuntimeError("No labels left after pruning. Relax criteria.")

    keep_indices = [label_columns_all.index(name) for name in labels_keep]

    y_train = y_train_all[:, keep_indices]
    y_val = y_val_all[:, keep_indices]
    y_test = y_test_all[:, keep_indices]

    print("\n=== Stage 2: Retraining on Pruned Labels ===")
    print(f"Final label count for retraining: {y_train.shape[1]}")

    train_loader = build_loader(x_train, y_train, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = build_loader(x_val, y_val, batch_size, shuffle=False, num_workers=num_workers)
    test_loader = build_loader(x_test, y_test, batch_size, shuffle=False, num_workers=num_workers)

    model = NotifSenseMLP(input_dim=x_train.shape[1], num_labels=y_train.shape[1]).to(device)
    pos_weight = compute_pos_weight(y_train).to(device)
    criterion = get_criterion(config, pos_weight)

    best_state = train_with_early_stopping(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
    )
    model.load_state_dict(best_state)

    # Per-label threshold tuning for kept labels.
    val_loss, y_val_true, y_val_prob = collect_predictions(model, val_loader, criterion, device)
    tuned_thresholds = tune_thresholds(y_val_true, y_val_prob, labels_keep)

    thresholds_path = Path("data/processed/label_thresholds.json")
    save_thresholds(thresholds_path, tuned_thresholds)
    print(f"\nSaved tuned thresholds: {thresholds_path}")

    # Final test evaluation using tuned thresholds.
    test_loss, y_test_true, y_test_prob = collect_predictions(model, test_loader, criterion, device)
    y_test_pred = predict_with_threshold_dict(y_test_prob, labels_keep, tuned_thresholds)

    per_label_final = compute_per_label_metrics(y_test_true, y_test_pred, labels_keep)
    summary = evaluate_summary(y_test_true, y_test_pred)

    print("\n=== Final Test Metrics ===")
    print(f"Test loss: {test_loss:.6f}")
    print(f"Macro F1: {summary['macro_f1']:.4f}")
    print(f"Micro F1: {summary['micro_f1']:.4f}")
    print(f"Average per-label F1: {summary['avg_per_label_f1']:.4f}")
    print(f"Final label count: {len(labels_keep)}")

    print_per_label_table(per_label_final)

    final_model_path = Path("models/notifsense_final.pt")
    save_final_model(
        out_path=final_model_path,
        model_state=best_state,
        input_dim=x_train.shape[1],
        label_columns=labels_keep,
        feature_columns=feature_columns,
        feature_mean=mean,
        feature_std=std,
        thresholds=tuned_thresholds,
        removed_labels=labels_remove,
    )
    print(f"\nSaved final model: {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NotifSense optimization-phase training")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    main(args.config)

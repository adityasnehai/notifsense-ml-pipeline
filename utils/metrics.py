"""Evaluation metrics for multi-label classification."""

from typing import Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_multilabel_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute core multi-label metrics from model logits.

    Args:
        logits: Tensor of shape (batch_size, num_labels).
        labels: Tensor of shape (batch_size, num_labels).
        threshold: Sigmoid threshold to binarize predictions.

    Returns:
        Dictionary containing F1, precision, recall, and subset accuracy.
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).int().cpu().numpy()
    targets = labels.int().cpu().numpy()

    metrics = {
        "f1_micro": f1_score(targets, preds, average="micro", zero_division=0),
        "precision_micro": precision_score(targets, preds, average="micro", zero_division=0),
        "recall_micro": recall_score(targets, preds, average="micro", zero_division=0),
        "subset_accuracy": accuracy_score(targets, preds),
    }

    return {k: float(v) for k, v in metrics.items()}

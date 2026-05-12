"""Evaluation metrics: F1, AUROC, AUPRC, pass@k, API validity, test pass rate."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    precision_score, recall_score, accuracy_score,
)


def compute_all_metrics(
    y_true: np.ndarray, y_pred: np.ndarray,
    y_proba: np.ndarray = None,
) -> Dict[str, float]:
    """Compute all standard metrics for hallucination detection."""
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
    }
    if y_proba is not None and len(set(y_true)) > 1:
        metrics["auroc"] = float(roc_auc_score(y_true, y_proba))
        metrics["auprc"] = float(average_precision_score(y_true, y_proba))
    return metrics


def pass_at_k(n: int, c: int, k: int) -> float:
    """Compute pass@k metric: probability of at least 1 correct in k samples.

    Args:
        n: Total number of generated samples.
        c: Number of correct samples.
        k: Number of samples to consider.
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def compute_pass_at_k(
    results: List[Dict[str, bool]], k_values: List[int] = [1, 3, 5],
) -> Dict[str, float]:
    """Compute pass@k for a list of task results.

    Each result dict has: {'n': total_samples, 'c': correct_samples}.
    """
    metrics = {}
    for k in k_values:
        scores = [pass_at_k(r["n"], r["c"], k) for r in results if r["n"] >= k]
        metrics[f"pass@{k}"] = float(np.mean(scores)) if scores else 0.0
    return metrics

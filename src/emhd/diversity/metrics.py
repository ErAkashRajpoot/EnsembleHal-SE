"""Ensemble diversity metrics: Yule's Q-statistic and Fleiss' kappa."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def pairwise_q_statistic(
    predictions_a: np.ndarray, predictions_b: np.ndarray,
) -> float:
    """Compute Yule's Q-statistic between two binary classifiers.

    Q ∈ [-1, 1]: Q=0 means independent, Q=1 means identical.
    Values < 0.7 indicate sufficient diversity.

    Args:
        predictions_a: Binary predictions from model A.
        predictions_b: Binary predictions from model B.
    """
    n11 = np.sum((predictions_a == 1) & (predictions_b == 1))
    n00 = np.sum((predictions_a == 0) & (predictions_b == 0))
    n10 = np.sum((predictions_a == 1) & (predictions_b == 0))
    n01 = np.sum((predictions_a == 0) & (predictions_b == 1))

    numerator = n11 * n00 - n10 * n01
    denominator = n11 * n00 + n10 * n01

    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def q_statistic_matrix(
    all_predictions: List[np.ndarray],
) -> np.ndarray:
    """Compute pairwise Q-statistic matrix for all model pairs."""
    n_models = len(all_predictions)
    matrix = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(i + 1, n_models):
            q = pairwise_q_statistic(all_predictions[i], all_predictions[j])
            matrix[i, j] = q
            matrix[j, i] = q
    np.fill_diagonal(matrix, 1.0)
    return matrix


def fleiss_kappa(
    ratings: np.ndarray,
) -> float:
    """Compute Fleiss' kappa for inter-rater agreement.

    Args:
        ratings: (n_subjects, n_categories) matrix where ratings[i][j]
                 is the number of raters who assigned category j to subject i.
    """
    n_subjects, n_categories = ratings.shape
    n_raters = int(ratings[0].sum())

    if n_raters <= 1:
        return 0.0

    # Proportion of ratings per category
    p_j = ratings.sum(axis=0) / (n_subjects * n_raters)

    # Per-subject agreement
    P_i = (np.sum(ratings ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = np.mean(P_i)

    # Expected agreement by chance
    P_e = np.sum(p_j ** 2)

    if abs(1 - P_e) < 1e-10:
        return 1.0 if abs(P_bar - 1.0) < 1e-10 else 0.0

    return float((P_bar - P_e) / (1 - P_e))


def bootstrap_ci(
    predictions: List[np.ndarray],
    metric_fn,
    n_iterations: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Bootstrap 95% CI for a diversity metric."""
    rng = np.random.RandomState(seed)
    n = len(predictions[0])
    scores = []
    for _ in range(n_iterations):
        idx = rng.randint(0, n, size=n)
        sampled = [p[idx] for p in predictions]
        scores.append(metric_fn(sampled))
    scores = np.array(scores)
    return {
        "mean": float(np.mean(scores)),
        "ci_lower": float(np.percentile(scores, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(scores, 100 * (1 - alpha / 2))),
    }

"""Statistical tests: bootstrap CIs, Wilcoxon, McNemar, DeLong."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score


class StatisticalTests:
    """Statistical significance tests for model comparison."""

    def __init__(self, seed: int = 42, n_bootstrap: int = 10000):
        self.seed = seed
        self.n_bootstrap = n_bootstrap

    def bootstrap_ci(
        self, y_true: np.ndarray, y_pred: np.ndarray,
        metric_fn=f1_score, alpha: float = 0.05,
    ) -> Dict[str, float]:
        """Bootstrap confidence intervals (percentile method)."""
        rng = np.random.RandomState(self.seed)
        scores = []
        n = len(y_true)
        for _ in range(self.n_bootstrap):
            idx = rng.randint(0, n, size=n)
            try:
                s = metric_fn(y_true[idx], y_pred[idx])
                scores.append(s)
            except Exception:
                continue
        scores = np.array(scores)
        lo = np.percentile(scores, 100 * alpha / 2)
        hi = np.percentile(scores, 100 * (1 - alpha / 2))
        return {"mean": float(np.mean(scores)), "ci_lower": float(lo), "ci_upper": float(hi)}

    def wilcoxon_signed_rank(
        self, fold_scores_a: List[float], fold_scores_b: List[float],
    ) -> Dict[str, float]:
        """Wilcoxon signed-rank test across CV folds."""
        a, b = np.array(fold_scores_a), np.array(fold_scores_b)
        if len(a) < 5:
            return {"statistic": 0.0, "p_value": 1.0}
        stat, p = stats.wilcoxon(a, b, alternative="two-sided")
        return {"statistic": float(stat), "p_value": float(p)}

    def mcnemar_test(
        self, y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
    ) -> Dict[str, float]:
        """McNemar's test for paired nominal data."""
        correct_a = (pred_a == y_true)
        correct_b = (pred_b == y_true)
        b_count = int(np.sum(correct_a & ~correct_b))  # A right, B wrong
        c_count = int(np.sum(~correct_a & correct_b))  # A wrong, B right
        if b_count + c_count == 0:
            return {"statistic": 0.0, "p_value": 1.0}
        stat = (abs(b_count - c_count) - 1) ** 2 / (b_count + c_count)
        p = float(stats.chi2.sf(stat, df=1))
        return {"statistic": float(stat), "p_value": p}

    def holm_correction(self, p_values: List[float]) -> List[float]:
        """Holm-Bonferroni correction for multiple comparisons."""
        n = len(p_values)
        indexed = sorted(enumerate(p_values), key=lambda x: x[1])
        corrected = [0.0] * n
        for rank, (orig_idx, p) in enumerate(indexed):
            corrected[orig_idx] = min(p * (n - rank), 1.0)
        # Enforce monotonicity
        for i in range(1, n):
            idx = indexed[i][0]
            prev_idx = indexed[i - 1][0]
            corrected[idx] = max(corrected[idx], corrected[prev_idx])
        return corrected

    def delong_test(
        self, y_true: np.ndarray, proba_a: np.ndarray, proba_b: np.ndarray,
    ) -> Dict[str, float]:
        """DeLong test for comparing two AUROC values."""
        auc_a = roc_auc_score(y_true, proba_a)
        auc_b = roc_auc_score(y_true, proba_b)
        # Simplified: use bootstrap for AUROC difference significance
        rng = np.random.RandomState(self.seed)
        diffs = []
        n = len(y_true)
        for _ in range(self.n_bootstrap):
            idx = rng.randint(0, n, size=n)
            try:
                a = roc_auc_score(y_true[idx], proba_a[idx])
                b = roc_auc_score(y_true[idx], proba_b[idx])
                diffs.append(a - b)
            except Exception:
                continue
        diffs = np.array(diffs)
        z = np.mean(diffs) / (np.std(diffs) + 1e-10)
        p = float(2 * stats.norm.sf(abs(z)))
        return {"auc_a": auc_a, "auc_b": auc_b, "z_stat": float(z), "p_value": p}

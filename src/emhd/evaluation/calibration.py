"""Calibration pipeline: isotonic regression, ECE, and Brier score."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE)."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        avg_confidence = y_prob[mask].mean()
        avg_accuracy = y_true[mask].mean()
        ece += mask.sum() * abs(avg_accuracy - avg_confidence)
    return float(ece / len(y_true))


class CalibrationPipeline:
    """Isotonic regression calibration with ECE and Brier metrics."""

    def __init__(self):
        self._calibrator = IsotonicRegression(out_of_bounds="clip")

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        """Fit isotonic regression on calibration split."""
        self._calibrator.fit(y_prob, y_true)

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply calibration to probabilities."""
        return self._calibrator.predict(y_prob)

    def evaluate(
        self, y_true: np.ndarray, y_prob_raw: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate pre- and post-calibration metrics."""
        y_prob_cal = self.calibrate(y_prob_raw)
        return {
            "ece_pre": expected_calibration_error(y_true, y_prob_raw),
            "ece_post": expected_calibration_error(y_true, y_prob_cal),
            "brier_pre": float(brier_score_loss(y_true, y_prob_raw)),
            "brier_post": float(brier_score_loss(y_true, y_prob_cal)),
        }

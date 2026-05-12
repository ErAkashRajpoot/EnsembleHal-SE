"""Simple statistical baselines: majority voting, RF, LR, best single model."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class MajorityVotingBaseline:
    """Majority voting across model outputs (binary: hallucinated or not)."""

    def predict_single(self, model_predictions: List[int]) -> int:
        if not model_predictions:
            return 0
        return 1 if sum(model_predictions) > len(model_predictions) / 2 else 0

    def predict_batch(self, predictions_list: List[List[int]]) -> List[int]:
        return [self.predict_single(p) for p in predictions_list]


class BestSingleModelBaseline:
    """Use the best-performing single model's confidence as the baseline."""

    def __init__(self):
        self.best_model_idx: int = 0

    def fit(self, features: np.ndarray, labels: np.ndarray, model_indices: List[int]) -> None:
        from sklearn.metrics import f1_score
        best_f1, best_idx = 0.0, 0
        for idx in set(model_indices):
            mask = [i == idx for i in model_indices]
            f1 = f1_score(labels[mask], (features[mask, 0] > 0.5).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_idx = f1, idx
        self.best_model_idx = best_idx

    def predict(self, features: np.ndarray) -> np.ndarray:
        return (features[:, 0] > 0.5).astype(int)


class RandomForestBaseline:
    """Random Forest meta-learner baseline."""

    def __init__(self, n_estimators: int = 100, seed: int = 42):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, class_weight="balanced")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class LogisticRegressionBaseline:
    """Logistic Regression meta-learner (sanity check baseline)."""

    def __init__(self, seed: int = 42):
        self.model = LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

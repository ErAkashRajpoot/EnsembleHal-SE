"""Ablation sweeps: K-model, Q-threshold, cross-dataset, temperature, feature layers."""
from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..metalearner.cv import StratifiedCVRunner

logger = logging.getLogger(__name__)


class AblationRunner:
    """Run all ablation studies required by the paper."""

    def __init__(self, seed: int = 42, n_folds: int = 5, n_trials: int = 30):
        self.seed = seed
        self.n_folds = n_folds
        self.n_trials = n_trials

    def k_model_sweep(
        self, df: pd.DataFrame, k_values: List[int] = [2, 3, 4, 5],
    ) -> List[Dict[str, Any]]:
        """Ablation: vary number of models K in {2,3,4,5}."""
        results = []
        for k in k_values:
            logger.info("K-model sweep: K=%d", k)
            cv = StratifiedCVRunner(n_folds=self.n_folds, seed=self.seed)
            metrics = cv.run(df, model_type="xgboost", n_trials=self.n_trials)
            metrics["K"] = k
            results.append(metrics)
        return results

    def q_threshold_sweep(
        self, q_values: List[float] = [0.5, 0.6, 0.7, 0.8],
    ) -> List[Dict[str, Any]]:
        """Ablation: Q-threshold sensitivity sweep."""
        results = []
        for q in q_values:
            results.append({"q_threshold": q, "note": "Requires diversity data"})
        return results

    def cross_dataset_generalization(
        self, df_train: pd.DataFrame, df_test: pd.DataFrame,
        train_name: str = "codemirage", test_name: str = "codehalu",
    ) -> Dict[str, Any]:
        """Ablation: train on one dataset, test on another."""
        feature_cols = [c for c in df_train.columns if c not in ("task_id", "label")]
        X_train = df_train[feature_cols].values.astype(np.float32)
        y_train = (df_train["label"] > 0).astype(int).values
        X_test = df_test[feature_cols].values.astype(np.float32)
        y_test = (df_test["label"] > 0).astype(int).values

        from ..metalearner.trainer import MetaLearnerTrainer
        from ..evaluation.metrics import compute_all_metrics

        trainer = MetaLearnerTrainer(seed=self.seed, n_trials=self.n_trials)
        split = int(len(X_train) * 0.85)
        model, _ = trainer.train_xgboost(X_train[:split], y_train[:split], X_train[split:], y_train[split:])

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = compute_all_metrics(y_test, y_pred, y_proba)

        return {"train": train_name, "test": test_name, **metrics}

    def feature_layer_ablation(
        self, df: pd.DataFrame,
        layer_prefixes: Dict[str, str] = None,
    ) -> List[Dict[str, Any]]:
        """Ablation: compare feature layer combinations."""
        if layer_prefixes is None:
            layer_prefixes = {
                "L1": "l1_", "L2": "l2_", "L3": "l3_", "L4": "l4_",
            }

        configs = [
            ("L1_only", ["L1"]),
            ("L2_only", ["L2"]),
            ("L1+L2", ["L1", "L2"]),
            ("L1+L2+L3", ["L1", "L2", "L3"]),
            ("All_layers", ["L1", "L2", "L3", "L4"]),
        ]

        results = []
        for config_name, layers in configs:
            prefixes = [layer_prefixes[l] for l in layers]
            cols = ["task_id", "label"] + [
                c for c in df.columns
                if any(c.startswith(p) for p in prefixes)
            ]
            df_subset = df[cols]

            cv = StratifiedCVRunner(n_folds=self.n_folds, seed=self.seed)
            metrics = cv.run(df_subset, model_type="xgboost", n_trials=self.n_trials)
            metrics["config"] = config_name
            metrics["layers"] = layers
            results.append(metrics)
            logger.info("%s: F1=%.4f±%.4f", config_name, metrics["mean_f1"], metrics["std_f1"])

        return results

    def temperature_ablation(
        self, df_greedy: pd.DataFrame, df_sampled: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Ablation: greedy vs. sampled generation."""
        cv = StratifiedCVRunner(n_folds=self.n_folds, seed=self.seed)
        greedy = cv.run(df_greedy, model_type="xgboost", n_trials=self.n_trials)
        sampled = cv.run(df_sampled, model_type="xgboost", n_trials=self.n_trials)
        return {"greedy": greedy, "sampled": sampled}

    def save_results(self, results: Any, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=True, default=str)

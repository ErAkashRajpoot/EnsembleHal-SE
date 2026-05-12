"""Meta-learner trainer with Optuna hyperparameter optimization."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import optuna
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import f1_score, roc_auc_score

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


class MetaLearnerTrainer:
    """Trains XGBoost and LightGBM meta-learners with Optuna TPE search."""

    def __init__(self, seed: int = 42, n_trials: int = 50, timeout: int = 600):
        self.seed = seed
        self.n_trials = n_trials
        self.timeout = timeout

    # ── XGBoost ───────────────────────────────────────────────────────────

    def _xgb_objective(
        self, trial: optuna.Trial,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
    ) -> float:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "seed": self.seed,
            "verbosity": 0,
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 10.0),
        }

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        y_pred = model.predict(X_val)
        return f1_score(y_val, y_pred, average="binary")

    def train_xgboost(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
    ) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
        """Train XGBoost with Optuna HPO."""
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
        )
        study.optimize(
            lambda trial: self._xgb_objective(trial, X_train, y_train, X_val, y_val),
            n_trials=self.n_trials, timeout=self.timeout,
        )

        best_params = {**study.best_params, "objective": "binary:logistic",
                       "eval_metric": "logloss", "tree_method": "hist",
                       "seed": self.seed, "verbosity": 0}

        model = xgb.XGBClassifier(**best_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        logger.info("XGBoost best F1=%.4f, params=%s", study.best_value, study.best_params)
        return model, {"best_f1": study.best_value, "best_params": study.best_params}

    # ── LightGBM ──────────────────────────────────────────────────────────

    def _lgb_objective(
        self, trial: optuna.Trial,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
    ) -> float:
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "seed": self.seed,
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "is_unbalance": True,
        }

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.log_evaluation(period=0)])

        y_pred = model.predict(X_val)
        return f1_score(y_val, y_pred, average="binary")

    def train_lightgbm(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
    ) -> Tuple[lgb.LGBMClassifier, Dict[str, Any]]:
        """Train LightGBM with Optuna HPO."""
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
        )
        study.optimize(
            lambda trial: self._lgb_objective(trial, X_train, y_train, X_val, y_val),
            n_trials=self.n_trials, timeout=self.timeout,
        )

        best_params = {**study.best_params, "objective": "binary",
                       "metric": "binary_logloss", "verbosity": -1,
                       "seed": self.seed, "is_unbalance": True}

        model = lgb.LGBMClassifier(**best_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.log_evaluation(period=0)])

        logger.info("LightGBM best F1=%.4f, params=%s", study.best_value, study.best_params)
        return model, {"best_f1": study.best_value, "best_params": study.best_params}

    def save_model(self, model: Any, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)

    def load_model(self, path: Path) -> Any:
        return joblib.load(path)

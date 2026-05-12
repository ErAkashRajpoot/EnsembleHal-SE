"""CLI: Train meta-learner (XGBoost/LightGBM) with Optuna HPO."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from emhd.metalearner import MetaLearnerTrainer, StratifiedCVRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train meta-learner.")
    parser.add_argument("--features", default="data/features/features.csv")
    parser.add_argument("--output-dir", default="data/models")
    parser.add_argument("--model-type", choices=["xgboost", "lightgbm", "both"], default="both")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    df = pd.read_csv(args.features)
    logger.info("Loaded %d samples with %d features", len(df), len(df.columns) - 2)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cv_runner = StratifiedCVRunner(n_folds=args.n_folds, seed=args.seed)

    if args.model_type in ("xgboost", "both"):
        logger.info("Running XGBoost CV...")
        xgb_results = cv_runner.run(df, model_type="xgboost", n_trials=args.n_trials)
        logger.info("XGBoost: F1=%.4f±%.4f AUROC=%.4f±%.4f",
                     xgb_results["mean_f1"], xgb_results["std_f1"],
                     xgb_results["mean_auroc"], xgb_results["std_auroc"])
        with (output_dir / "xgboost_cv_results.json").open("w") as f:
            json.dump(xgb_results, f, indent=2, default=str)

    if args.model_type in ("lightgbm", "both"):
        logger.info("Running LightGBM CV...")
        lgb_results = cv_runner.run(df, model_type="lightgbm", n_trials=args.n_trials)
        logger.info("LightGBM: F1=%.4f±%.4f AUROC=%.4f±%.4f",
                     lgb_results["mean_f1"], lgb_results["std_f1"],
                     lgb_results["mean_auroc"], lgb_results["std_auroc"])
        with (output_dir / "lightgbm_cv_results.json").open("w") as f:
            json.dump(lgb_results, f, indent=2, default=str)


if __name__ == "__main__":
    main()

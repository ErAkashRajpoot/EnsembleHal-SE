"""CLI: Run all ablation studies."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from emhd.ablation import AblationRunner, ErrorAnalyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation studies.")
    parser.add_argument("--features", default="data/features/features.csv")
    parser.add_argument("--output-dir", default="data/results/ablations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features)
    logger.info("Loaded %d samples for ablation", len(df))

    runner = AblationRunner(seed=args.seed, n_trials=args.n_trials)

    # Feature layer ablation
    logger.info("Running feature layer ablation...")
    layer_results = runner.feature_layer_ablation(df)
    runner.save_results(layer_results, output_dir / "feature_layer_ablation.json")

    logger.info("Ablation studies complete. Results saved to %s", output_dir)


if __name__ == "__main__":
    main()

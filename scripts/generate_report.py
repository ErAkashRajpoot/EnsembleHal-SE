"""CLI: Generate LaTeX tables and Matplotlib figures from evaluation results."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from emhd.ablation.reporting import generate_full_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper tables and figures.")
    parser.add_argument("--results-dir", default="data/results", help="Directory with result JSONs")
    parser.add_argument("--output-dir", default="data/results/report", help="Output for LaTeX + figures")
    args = parser.parse_args()

    logger.info("Generating report from %s", args.results_dir)
    generate_full_report(Path(args.results_dir), Path(args.output_dir))
    logger.info("Report generation complete.")


if __name__ == "__main__":
    main()

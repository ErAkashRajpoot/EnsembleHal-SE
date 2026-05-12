"""CLI: Run ensemble diversity validation pilot.

Computes pairwise Q-statistic and Fleiss' kappa across the 5 offline
models to verify they produce sufficiently diverse outputs (Q < 0.7).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer

from emhd.diversity.metrics import (
    pairwise_q_statistic,
    q_statistic_matrix,
    fleiss_kappa,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run diversity validation pilot.")
    parser.add_argument("--registry", default="data/generations/registry.jsonl", help="Generation registry")
    parser.add_argument("--output-dir", default="data/results/diversity", help="Output directory")
    parser.add_argument("--similarity-threshold", type=float, default=0.5,
                        help="ROUGE-L threshold: above = 'agree', below = 'disagree'")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_registry(path: str) -> pd.DataFrame:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return pd.DataFrame(records)


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load registry
    logger.info("Loading registry from %s", args.registry)
    registry = _load_registry(args.registry)
    logger.info("Loaded %d generation records", len(registry))

    # Group by task_id and model_id
    model_ids = sorted(registry["model_id"].unique())
    task_ids = sorted(registry["task_id"].unique())
    logger.info("Models: %s", model_ids)
    logger.info("Tasks: %d", len(task_ids))

    # Build per-model output vectors
    # For each task, pick the first (greedy) generation per model
    model_outputs: Dict[str, Dict[str, str]] = {m: {} for m in model_ids}
    for _, row in registry.iterrows():
        mid = row["model_id"]
        tid = row["task_id"]
        if tid not in model_outputs[mid]:
            model_outputs[mid][tid] = str(row.get("output", row.get("generated_text", "")))

    # Get tasks present in ALL models
    common_tasks = set(task_ids)
    for mid in model_ids:
        common_tasks &= set(model_outputs[mid].keys())
    common_tasks = sorted(common_tasks)
    logger.info("Tasks present in all %d models: %d", len(model_ids), len(common_tasks))

    if len(common_tasks) < 10:
        logger.error("Too few common tasks for diversity analysis. Need at least 10.")
        return

    # ── Pairwise ROUGE-L similarity ──────────────────────────────────
    logger.info("Computing pairwise ROUGE-L similarity...")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    # Build binary prediction matrices (agree/disagree with reference model)
    # Reference = first model's output; binary = "similar to reference"
    n_tasks = len(common_tasks)
    binary_preds: List[np.ndarray] = []

    # Use first model as reference baseline
    ref_model = model_ids[0]
    for mid in model_ids:
        preds = np.zeros(n_tasks, dtype=int)
        for i, tid in enumerate(common_tasks):
            ref_out = model_outputs[ref_model][tid]
            model_out = model_outputs[mid][tid]
            rouge_result = scorer.score(ref_out, model_out)
            similarity = rouge_result["rougeL"].fmeasure
            preds[i] = 1 if similarity > args.similarity_threshold else 0
        binary_preds.append(preds)

    # ── Q-statistic matrix ───────────────────────────────────────────
    logger.info("Computing Q-statistic matrix...")
    q_matrix = q_statistic_matrix(binary_preds)

    logger.info("Q-statistic matrix:")
    for i, mi in enumerate(model_ids):
        row_str = "  ".join(f"{q_matrix[i, j]:+.3f}" for j in range(len(model_ids)))
        logger.info("  %s: %s", mi[:15].ljust(15), row_str)

    # Average off-diagonal Q
    mask = ~np.eye(len(model_ids), dtype=bool)
    avg_q = float(np.mean(q_matrix[mask]))
    logger.info("Average pairwise Q-statistic: %.4f (target < 0.7 for sufficient diversity)", avg_q)

    # ── Fleiss' kappa ────────────────────────────────────────────────
    logger.info("Computing Fleiss' kappa...")

    # Build ratings matrix: (n_tasks, 2) where col 0 = "disagree", col 1 = "agree"
    ratings = np.zeros((n_tasks, 2))
    for preds in binary_preds:
        for i in range(n_tasks):
            ratings[i, preds[i]] += 1

    kappa = fleiss_kappa(ratings)
    logger.info("Fleiss' kappa: %.4f", kappa)
    logger.info("  Interpretation: < 0.2 = poor agreement (high diversity, good!)")
    logger.info("                  0.2-0.4 = fair | 0.4-0.6 = moderate | > 0.6 = substantial")

    # ── Pairwise similarity matrix (ROUGE-L) ─────────────────────────
    logger.info("Computing pairwise model similarity (ROUGE-L)...")
    sim_matrix = np.zeros((len(model_ids), len(model_ids)))

    for i, mi in enumerate(model_ids):
        for j, mj in enumerate(model_ids):
            if i == j:
                sim_matrix[i, j] = 1.0
                continue
            sims = []
            for tid in common_tasks:
                out_i = model_outputs[mi][tid]
                out_j = model_outputs[mj][tid]
                result = scorer.score(out_i, out_j)
                sims.append(result["rougeL"].fmeasure)
            sim_matrix[i, j] = float(np.mean(sims))

    logger.info("Pairwise ROUGE-L similarity matrix:")
    for i, mi in enumerate(model_ids):
        row_str = "  ".join(f"{sim_matrix[i, j]:.3f}" for j in range(len(model_ids)))
        logger.info("  %s: %s", mi[:15].ljust(15), row_str)

    avg_sim = float(np.mean(sim_matrix[mask]))
    logger.info("Average pairwise ROUGE-L similarity: %.4f", avg_sim)

    # ── Save results ─────────────────────────────────────────────────
    results = {
        "model_ids": model_ids,
        "n_common_tasks": len(common_tasks),
        "similarity_threshold": args.similarity_threshold,
        "q_statistic_matrix": q_matrix.tolist(),
        "average_q_statistic": avg_q,
        "q_diversity_sufficient": avg_q < 0.7,
        "fleiss_kappa": kappa,
        "rouge_similarity_matrix": sim_matrix.tolist(),
        "average_rouge_similarity": avg_sim,
    }

    results_path = output_dir / "diversity_pilot.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", results_path)

    # ── Summary ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("DIVERSITY VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info("  Average Q-statistic : %.4f  %s", avg_q,
                "✓ Sufficient diversity" if avg_q < 0.7 else "✗ Models too similar")
    logger.info("  Fleiss' kappa       : %.4f  %s", kappa,
                "✓ Low agreement (good)" if kappa < 0.4 else "✗ High agreement")
    logger.info("  Average ROUGE-L sim : %.4f", avg_sim)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

"""Layer 1: Cross-model disagreement features (≥16 features).

This is the core novelty of the paper — measuring how much different LLMs
disagree on the same task to detect hallucinations.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
from rouge_score import rouge_scorer


def _safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_var(values: List[float]) -> float:
    return float(np.var(values)) if len(values) > 1 else 0.0


def _safe_std(values: List[float]) -> float:
    return float(np.std(values)) if len(values) > 1 else 0.0


# ── Token-level metrics ──────────────────────────────────────────────────────

def _token_jaccard(a: str, b: str) -> float:
    """Jaccard similarity between token sets."""
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance between two strings (word-level)."""
    words_a = a.split()
    words_b = b.split()
    m, n = len(words_a), len(words_b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if words_a[i - 1] == words_b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def _normalized_edit_distance(a: str, b: str) -> float:
    """Edit distance normalized by max length."""
    max_len = max(len(a.split()), len(b.split()))
    if max_len == 0:
        return 0.0
    return _edit_distance(a, b) / max_len


# ── ROUGE metrics ────────────────────────────────────────────────────────────

def _compute_rouge(a: str, b: str) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(a, b)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


# ── Pairwise computations ────────────────────────────────────────────────────

def _all_pairs(outputs: List[str]):
    """Generate all unique pairs of outputs."""
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            yield outputs[i], outputs[j]


def extract_layer1_features(
    outputs: List[str],
    embeddings: Optional[List[np.ndarray]] = None,
    bertscore_values: Optional[Dict[str, List[float]]] = None,
    ast_edit_distances: Optional[List[float]] = None,
    identifier_jaccards: Optional[List[float]] = None,
    api_call_jaccards: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Extract Layer 1 disagreement features from multiple model outputs.

    Args:
        outputs: List of text outputs from different models for the same task.
        embeddings: Optional pre-computed embeddings for cosine distance.
        bertscore_values: Optional pre-computed BERTScore P/R/F1 lists.
        ast_edit_distances: Optional pairwise AST edit distances.
        identifier_jaccards: Optional pairwise identifier Jaccard values.
        api_call_jaccards: Optional pairwise API call graph Jaccard values.

    Returns:
        Dictionary of feature names → values (≥16 features).
    """
    features: Dict[str, float] = {}

    if len(outputs) < 2:
        # Not enough outputs for disagreement; return zeros
        feature_names = [
            "jaccard_mean", "jaccard_var",
            "edit_dist_mean", "edit_dist_var",
            "rouge1_mean", "rouge1_var",
            "rouge2_mean", "rouge2_var",
            "rougeL_mean", "rougeL_var",
            "cosine_dist_mean", "cosine_dist_var",
            "bertscore_p_mean", "bertscore_p_std",
            "bertscore_r_mean", "bertscore_r_std",
            "bertscore_f1_mean", "bertscore_f1_std",
            "ast_edit_dist_var",
            "identifier_jaccard_var",
            "api_callgraph_jaccard",
        ]
        return {name: 0.0 for name in feature_names}

    # ── Token Jaccard ──
    jaccards = [_token_jaccard(a, b) for a, b in _all_pairs(outputs)]
    features["jaccard_mean"] = _safe_mean(jaccards)
    features["jaccard_var"] = _safe_var(jaccards)

    # ── Edit distance ──
    edit_dists = [_normalized_edit_distance(a, b) for a, b in _all_pairs(outputs)]
    features["edit_dist_mean"] = _safe_mean(edit_dists)
    features["edit_dist_var"] = _safe_var(edit_dists)

    # ── ROUGE ──
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    for a, b in _all_pairs(outputs):
        rouge = _compute_rouge(a, b)
        rouge1_scores.append(rouge["rouge1"])
        rouge2_scores.append(rouge["rouge2"])
        rougeL_scores.append(rouge["rougeL"])

    features["rouge1_mean"] = _safe_mean(rouge1_scores)
    features["rouge1_var"] = _safe_var(rouge1_scores)
    features["rouge2_mean"] = _safe_mean(rouge2_scores)
    features["rouge2_var"] = _safe_var(rouge2_scores)
    features["rougeL_mean"] = _safe_mean(rougeL_scores)
    features["rougeL_var"] = _safe_var(rougeL_scores)

    # ── Cosine distance (from pre-computed embeddings) ──
    if embeddings and len(embeddings) >= 2:
        cosine_dists = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                e_i = embeddings[i] / (np.linalg.norm(embeddings[i]) + 1e-9)
                e_j = embeddings[j] / (np.linalg.norm(embeddings[j]) + 1e-9)
                cos_sim = float(np.dot(e_i, e_j))
                cosine_dists.append(1.0 - cos_sim)
        features["cosine_dist_mean"] = _safe_mean(cosine_dists)
        features["cosine_dist_var"] = _safe_var(cosine_dists)
    else:
        features["cosine_dist_mean"] = 0.0
        features["cosine_dist_var"] = 0.0

    # ── BERTScore (from pre-computed values) ──
    if bertscore_values:
        features["bertscore_p_mean"] = _safe_mean(bertscore_values.get("precision", []))
        features["bertscore_p_std"] = _safe_std(bertscore_values.get("precision", []))
        features["bertscore_r_mean"] = _safe_mean(bertscore_values.get("recall", []))
        features["bertscore_r_std"] = _safe_std(bertscore_values.get("recall", []))
        features["bertscore_f1_mean"] = _safe_mean(bertscore_values.get("f1", []))
        features["bertscore_f1_std"] = _safe_std(bertscore_values.get("f1", []))
    else:
        for suffix in ["p_mean", "p_std", "r_mean", "r_std", "f1_mean", "f1_std"]:
            features[f"bertscore_{suffix}"] = 0.0

    # ── Code-specific: AST edit distance variance ──
    features["ast_edit_dist_var"] = _safe_var(ast_edit_distances or [])

    # ── Code-specific: Identifier Jaccard variance ──
    features["identifier_jaccard_var"] = _safe_var(identifier_jaccards or [])

    # ── Code-specific: API call graph Jaccard ──
    features["api_callgraph_jaccard"] = _safe_mean(api_call_jaccards or [])

    return features

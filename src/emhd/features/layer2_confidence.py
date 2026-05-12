"""Layer 2: Confidence features (≥8 features).

Captures model uncertainty from log-probabilities and verbalized confidence.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional

import numpy as np


def _safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_var(values: List[float]) -> float:
    return float(np.var(values)) if len(values) > 1 else 0.0


# ── Log-probability features ─────────────────────────────────────────────────

def _extract_token_logprobs(logprobs_data: Optional[Dict[str, Any]]) -> List[float]:
    """Extract per-token log-probabilities from various API formats."""
    if not logprobs_data:
        return []

    # OpenAI format: {"content": [{"logprob": -0.5, ...}, ...]}
    if "content" in logprobs_data and isinstance(logprobs_data["content"], list):
        return [
            float(item.get("logprob", 0.0))
            for item in logprobs_data["content"]
            if isinstance(item, dict) and "logprob" in item
        ]

    # Together/Groq format: {"tokens": [...], "token_logprobs": [...]}
    if "token_logprobs" in logprobs_data:
        return [float(lp) for lp in logprobs_data["token_logprobs"] if lp is not None]

    return []


def _logprob_entropy(token_logprobs: List[float]) -> float:
    """Compute entropy from token log-probabilities.

    H = -sum(p * log(p)) where p = exp(logprob)
    """
    if not token_logprobs:
        return 0.0

    probs = [math.exp(lp) for lp in token_logprobs]
    probs = [max(p, 1e-10) for p in probs]  # Avoid log(0)
    entropy = -sum(p * math.log(p) for p in probs)
    return entropy / max(len(probs), 1)  # Normalize by token count


def _logprob_mean(token_logprobs: List[float]) -> float:
    """Mean log-probability (higher = more confident)."""
    return _safe_mean(token_logprobs) if token_logprobs else 0.0


# ── Verbalized uncertainty ───────────────────────────────────────────────────

UNCERTAINTY_PATTERNS = [
    (r"\bi(?:'m| am) (?:not )?(?:sure|certain|confident)", 0.7),
    (r"\bprobably\b", 0.5),
    (r"\bperhaps\b", 0.6),
    (r"\bmaybe\b", 0.6),
    (r"\bmight\b", 0.5),
    (r"\bcould be\b", 0.5),
    (r"\bi think\b", 0.4),
    (r"\bi believe\b", 0.3),
    (r"\bnot sure\b", 0.8),
    (r"\buncertain\b", 0.8),
    (r"\bapproximately\b", 0.3),
    (r"\broughly\b", 0.3),
    (r"\bguess\b", 0.7),
    (r"\bassume\b", 0.5),
    (r"\bnote that\b", 0.2),
    (r"\bhowever\b", 0.2),
]


def _verbalized_uncertainty(text: str) -> float:
    """Detect uncertainty language in model output (0 = certain, 1 = very uncertain)."""
    text_lower = text.lower()
    matched_scores = []
    for pattern, score in UNCERTAINTY_PATTERNS:
        if re.search(pattern, text_lower):
            matched_scores.append(score)

    if not matched_scores:
        return 0.0

    # Combine: take the max uncertainty and weight by frequency
    max_uncertainty = max(matched_scores)
    frequency_weight = min(len(matched_scores) / 5.0, 1.0)
    return max_uncertainty * (0.5 + 0.5 * frequency_weight)


# ── Disagreement entropy ─────────────────────────────────────────────────────

def _disagreement_entropy(outputs: List[str]) -> float:
    """Entropy of the output distribution (higher = more disagreement).

    Uses exact-match frequency as the distribution.
    """
    if len(outputs) < 2:
        return 0.0

    # Normalize: strip and lowercase for comparison
    normalized = [o.strip().lower() for o in outputs]
    freq: Dict[str, int] = {}
    for o in normalized:
        freq[o] = freq.get(o, 0) + 1

    total = len(normalized)
    probs = [count / total for count in freq.values()]
    entropy = -sum(p * math.log(p + 1e-10) for p in probs)
    max_entropy = math.log(total + 1e-10)
    return entropy / max_entropy if max_entropy > 0 else 0.0


# ── Main feature extractor ───────────────────────────────────────────────────

def extract_layer2_features(
    outputs: List[str],
    logprobs_list: Optional[List[Optional[Dict[str, Any]]]] = None,
    has_logprobs_flags: Optional[List[bool]] = None,
) -> Dict[str, float]:
    """Extract Layer 2 confidence features.

    Args:
        outputs: Text outputs from different models.
        logprobs_list: Per-model logprobs data (None for models without support).
        has_logprobs_flags: Per-model flag for logprobs availability.

    Returns:
        Dictionary of feature names → values (≥8 features).
    """
    features: Dict[str, float] = {}

    # ── Logprob-based features ──
    if logprobs_list:
        entropies = []
        mean_logprobs = []
        for lp_data in logprobs_list:
            token_lps = _extract_token_logprobs(lp_data)
            if token_lps:
                entropies.append(_logprob_entropy(token_lps))
                mean_logprobs.append(_logprob_mean(token_lps))

        features["logprob_entropy_mean"] = _safe_mean(entropies)
        features["logprob_entropy_var"] = _safe_var(entropies)
        features["logprob_mean_mean"] = _safe_mean(mean_logprobs)
        features["logprob_mean_var"] = _safe_var(mean_logprobs)
    else:
        features["logprob_entropy_mean"] = 0.0
        features["logprob_entropy_var"] = 0.0
        features["logprob_mean_mean"] = 0.0
        features["logprob_mean_var"] = 0.0

    # ── Verbalized uncertainty ──
    uncertainties = [_verbalized_uncertainty(o) for o in outputs]
    features["verbalized_uncertainty_mean"] = _safe_mean(uncertainties)
    features["verbalized_uncertainty_max"] = max(uncertainties) if uncertainties else 0.0

    # ── Disagreement entropy ──
    features["disagreement_entropy"] = _disagreement_entropy(outputs)

    # ── has_logprobs flag (fraction of models with logprobs) ──
    if has_logprobs_flags:
        features["has_logprobs_ratio"] = sum(has_logprobs_flags) / len(has_logprobs_flags)
    else:
        features["has_logprobs_ratio"] = 0.0

    return features

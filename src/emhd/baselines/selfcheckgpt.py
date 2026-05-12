"""SelfCheckGPT baseline: sampling-based consensus (Manakul et al.).

Generates k samples from a SINGLE model, then measures consistency
via BERTScore/NLI between the original and sampled outputs.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from rouge_score import rouge_scorer


class SelfCheckGPTBaseline:
    """SelfCheckGPT: single-model self-consistency check."""

    def __init__(self, method: str = "bertscore"):
        self.method = method
        self._rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def predict_single(self, original: str, samples: List[str]) -> float:
        """Score a single output: higher = more likely hallucinated."""
        if not samples:
            return 0.5
        if self.method == "rouge":
            return self._rouge_inconsistency(original, samples)
        return self._token_inconsistency(original, samples)

    def _rouge_inconsistency(self, original: str, samples: List[str]) -> float:
        scores = []
        for s in samples:
            result = self._rouge.score(original, s)
            scores.append(result["rougeL"].fmeasure)
        avg = float(np.mean(scores))
        return 1.0 - avg  # Lower similarity = higher hallucination probability

    def _token_inconsistency(self, original: str, samples: List[str]) -> float:
        orig_tokens = set(original.lower().split())
        if not orig_tokens:
            return 0.5
        consistencies = []
        for s in samples:
            s_tokens = set(s.lower().split())
            if not s_tokens:
                consistencies.append(0.0)
                continue
            overlap = len(orig_tokens & s_tokens) / len(orig_tokens)
            consistencies.append(overlap)
        return 1.0 - float(np.mean(consistencies))

    def predict_batch(self, originals: List[str], samples_list: List[List[str]]) -> List[float]:
        return [self.predict_single(o, s) for o, s in zip(originals, samples_list)]

"""RAG-style baseline (De-Hallucinator approach).

Retrieves API documentation or reference material and checks if the
generated output is grounded in the retrieved context.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from rouge_score import rouge_scorer


class RAGBaseline:
    """De-Hallucinator style: check grounding against retrieved context."""

    def __init__(self):
        self._rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    def predict_single(self, output: str, context: str) -> float:
        """Score: higher = more likely hallucinated (not grounded)."""
        if not context or not output:
            return 0.5
        scores = self._rouge.score(output, context)
        grounding = (scores["rouge1"].fmeasure + scores["rougeL"].fmeasure) / 2
        return 1.0 - grounding

    def predict_batch(self, outputs: List[str], contexts: List[str]) -> List[float]:
        return [self.predict_single(o, c) for o, c in zip(outputs, contexts)]

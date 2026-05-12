"""MetaQA baseline: prompt mutation metamorphic relations.

Applies metamorphic testing by mutating the prompt and checking
if the model's outputs remain consistent under transformations.
"""
from __future__ import annotations

import random
import re
from typing import Dict, List

import numpy as np
from rouge_score import rouge_scorer


class MetaQABaseline:
    """MetaQA: metamorphic relation-based hallucination detection."""

    MUTATIONS = ["rephrase", "negate", "simplify"]

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self._rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def mutate_prompt(self, prompt: str, mutation: str = "rephrase") -> str:
        """Apply a metamorphic mutation to the prompt."""
        if mutation == "rephrase":
            # Simple rephrasing: reorder sentences
            sentences = [s.strip() for s in prompt.split(".") if s.strip()]
            if len(sentences) > 1:
                self._rng.shuffle(sentences)
            return ". ".join(sentences) + "."
        elif mutation == "negate":
            return prompt.replace("should", "should not").replace("must", "must not")
        elif mutation == "simplify":
            words = prompt.split()
            if len(words) > 20:
                return " ".join(words[:int(len(words) * 0.7)])
            return prompt
        return prompt

    def predict_single(
        self, original_output: str, mutated_outputs: List[str],
    ) -> float:
        """Score: inconsistency between original and mutated-prompt outputs."""
        if not mutated_outputs:
            return 0.5
        scores = []
        for mo in mutated_outputs:
            result = self._rouge.score(original_output, mo)
            scores.append(result["rougeL"].fmeasure)
        avg = float(np.mean(scores))
        return 1.0 - avg

    def predict_batch(
        self, originals: List[str], mutated_list: List[List[str]],
    ) -> List[float]:
        return [self.predict_single(o, m) for o, m in zip(originals, mutated_list)]

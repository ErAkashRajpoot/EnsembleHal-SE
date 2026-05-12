"""Error analysis: systematic inspection of FP/FN with taxonomy labels."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


class ErrorAnalyzer:
    """Analyze false positives and false negatives with taxonomy labels."""

    def analyze(
        self, y_true: np.ndarray, y_pred: np.ndarray,
        task_ids: List[str], labels: np.ndarray,
        n_fp: int = 50, n_fn: int = 50, seed: int = 42,
    ) -> Dict[str, Any]:
        """Sample FP and FN cases for qualitative error analysis."""
        rng = np.random.RandomState(seed)

        fp_idx = np.where((y_pred == 1) & (y_true == 0))[0]
        fn_idx = np.where((y_pred == 0) & (y_true == 1))[0]

        fp_sample = rng.choice(fp_idx, size=min(n_fp, len(fp_idx)), replace=False) if len(fp_idx) > 0 else np.array([])
        fn_sample = rng.choice(fn_idx, size=min(n_fn, len(fn_idx)), replace=False) if len(fn_idx) > 0 else np.array([])

        fp_cases = [{"task_id": task_ids[i], "true_label": int(labels[i]), "type": "FP"} for i in fp_sample.astype(int)]
        fn_cases = [{"task_id": task_ids[i], "true_label": int(labels[i]), "type": "FN"} for i in fn_sample.astype(int)]

        return {
            "total_fp": int(len(fp_idx)),
            "total_fn": int(len(fn_idx)),
            "sampled_fp": len(fp_cases),
            "sampled_fn": len(fn_cases),
            "fp_cases": fp_cases,
            "fn_cases": fn_cases,
        }

    def save(self, analysis: Dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=True)

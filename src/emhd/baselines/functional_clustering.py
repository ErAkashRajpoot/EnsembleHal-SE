"""Functional Clustering baseline: execution-based code grouping.

Groups generated code by execution behavior (pass/fail on test cases)
and flags outputs that cluster away from the majority.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np


class FunctionalClusteringBaseline:
    """Cluster code outputs by functional behavior."""

    def cluster_by_output(
        self, execution_results: List[Optional[str]],
    ) -> Dict[str, List[int]]:
        """Cluster outputs by their execution result hash."""
        clusters: Dict[str, List[int]] = {}
        for idx, result in enumerate(execution_results):
            key = hashlib.md5((result or "ERROR").encode()).hexdigest()[:8]
            clusters.setdefault(key, []).append(idx)
        return clusters

    def predict_single(
        self, execution_results: List[Optional[str]], target_idx: int = 0,
    ) -> float:
        """Score: 1.0 if the target output is in a minority cluster."""
        if not execution_results:
            return 0.5
        clusters = self.cluster_by_output(execution_results)
        # Find which cluster the target belongs to
        target_result = execution_results[target_idx] or "ERROR"
        target_key = hashlib.md5(target_result.encode()).hexdigest()[:8]
        target_cluster_size = len(clusters.get(target_key, []))
        max_cluster_size = max(len(v) for v in clusters.values())
        if max_cluster_size == 0:
            return 0.5
        return 1.0 - (target_cluster_size / len(execution_results))

    def predict_batch(
        self, execution_results_list: List[List[Optional[str]]],
    ) -> List[float]:
        return [self.predict_single(er) for er in execution_results_list]

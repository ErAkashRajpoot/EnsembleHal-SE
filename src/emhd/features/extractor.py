"""Feature extractor: orchestrates all 4 layers into a unified feature matrix."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .layer1_disagreement import extract_layer1_features
from .layer2_confidence import extract_layer2_features
from .layer3_verification import extract_layer3_features
from .layer4_context import extract_layer4_features


class FeatureExtractor:
    """Combine all feature layers for each task."""

    def extract_for_task(
        self,
        task_id: str,
        outputs: List[str],
        reference: str,
        artefact_type: str,
        prompt: str,
        dataset_source: str,
        hallucination_label: int,
        embeddings: Optional[List[np.ndarray]] = None,
        bertscore_values: Optional[Dict[str, List[float]]] = None,
        logprobs_list: Optional[List[Optional[Dict[str, Any]]]] = None,
        has_logprobs_flags: Optional[List[bool]] = None,
        nli_scores: Optional[Dict[str, float]] = None,
        ast_edit_distances: Optional[List[float]] = None,
        identifier_jaccards: Optional[List[float]] = None,
        api_call_jaccards: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Extract all features for a single task."""
        features: Dict[str, Any] = {"task_id": task_id, "label": hallucination_label}

        # Layer 1: Disagreement
        l1 = extract_layer1_features(
            outputs, embeddings, bertscore_values,
            ast_edit_distances, identifier_jaccards, api_call_jaccards,
        )
        features.update({f"l1_{k}": v for k, v in l1.items()})

        # Layer 2: Confidence
        l2 = extract_layer2_features(outputs, logprobs_list, has_logprobs_flags)
        features.update({f"l2_{k}": v for k, v in l2.items()})

        # Layer 3: Verification
        l3 = extract_layer3_features(outputs, reference, nli_scores)
        features.update({f"l3_{k}": v for k, v in l3.items()})

        # Layer 4: Context
        l4 = extract_layer4_features(artefact_type, prompt, dataset_source)
        features.update({f"l4_{k}": v for k, v in l4.items()})

        features["num_outputs"] = len(outputs)
        return features

    def extract_batch(self, tasks: List[Dict[str, Any]]) -> pd.DataFrame:
        """Extract features for a batch of tasks and return as DataFrame."""
        rows = []
        for task in tasks:
            row = self.extract_for_task(**task)
            rows.append(row)
        return pd.DataFrame(rows)

    def save_features(self, df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    def load_features(self, path: Path) -> pd.DataFrame:
        return pd.read_csv(path)

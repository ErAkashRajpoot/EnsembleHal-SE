"""Layer 3: NLI and AST verification features (≥4 features)."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

import numpy as np


def extract_nli_features(nli_scores: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    if nli_scores:
        return {
            "nli_entailment": float(nli_scores.get("entailment", 0.0)),
            "nli_contradiction": float(nli_scores.get("contradiction", 0.0)),
            "nli_neutral": float(nli_scores.get("neutral", 0.0)),
        }
    return {"nli_entailment": 0.0, "nli_contradiction": 0.0, "nli_neutral": 0.0}


_ID_PATTERN = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b")
_PY_KW = {"False","None","True","and","as","assert","async","await","break","class","continue","def","del","elif","else","except","finally","for","from","global","if","import","in","is","lambda","nonlocal","not","or","pass","raise","return","try","while","with","yield"}
_JAVA_KW = {"abstract","assert","boolean","break","byte","case","catch","char","class","const","continue","default","do","double","else","enum","extends","final","finally","float","for","goto","if","implements","import","instanceof","int","interface","long","native","new","package","private","protected","public","return","short","static","strictfp","super","switch","synchronized","this","throw","throws","transient","try","void","volatile","while","true","false","null"}


def _extract_identifiers(code: str, language: str = "python") -> Set[str]:
    keywords = _PY_KW if language == "python" else _JAVA_KW
    return {m for m in _ID_PATTERN.findall(code) if m not in keywords and len(m) > 1}


def compute_ast_identifier_coverage(output: str, reference: str, language: str = "python") -> float:
    """ρ(o,s) = |ID(o) ∩ AST(s)| / |ID(o)|"""
    output_ids = _extract_identifiers(output, language)
    if not output_ids:
        return 1.0
    return len(output_ids & _extract_identifiers(reference, language)) / len(output_ids)


def extract_layer3_features(
    outputs: List[str], reference: str,
    nli_scores: Optional[Dict[str, float]] = None, language: str = "python",
) -> Dict[str, float]:
    features = extract_nli_features(nli_scores)
    if outputs and reference:
        coverages = [compute_ast_identifier_coverage(o, reference, language) for o in outputs]
        features["ast_id_coverage_mean"] = float(np.mean(coverages))
        features["ast_id_coverage_min"] = float(np.min(coverages))
        features["ast_id_coverage_var"] = float(np.var(coverages)) if len(coverages) > 1 else 0.0
    else:
        features.update({"ast_id_coverage_mean": 0.0, "ast_id_coverage_min": 0.0, "ast_id_coverage_var": 0.0})
    return features

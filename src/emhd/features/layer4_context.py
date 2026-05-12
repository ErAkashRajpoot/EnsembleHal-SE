"""Layer 4: Task context features (≥6 features)."""
from __future__ import annotations
from typing import Any, Dict, Optional

_ARTEFACT_TYPE_MAP = {"code": 0, "api": 1, "test": 2, "doc": 3, "review": 4, "explanation": 5}
_SOURCE_MAP = {"codemirage": 0, "codehalu": 1, "cloudapibench": 2, "defects4j": 3, "humaneval": 4, "mbpp": 5, "codereviewer": 6, "tl-codesum": 7}

def extract_layer4_features(
    artefact_type: str, prompt: str, dataset_source: str,
    repo_context_depth: int = 0, api_rarity: float = 0.0, test_difficulty: float = 0.0,
) -> Dict[str, float]:
    return {
        "artefact_type_enc": float(_ARTEFACT_TYPE_MAP.get(artefact_type, -1)),
        "prompt_length": float(len(prompt.split())),
        "prompt_char_length": float(len(prompt)),
        "repo_context_depth": float(repo_context_depth),
        "api_rarity": float(api_rarity),
        "test_difficulty": float(test_difficulty),
        "dataset_source_enc": float(_SOURCE_MAP.get(dataset_source, -1)),
    }

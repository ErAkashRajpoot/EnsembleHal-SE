from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .io import load_yaml


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    providers: List[str]
    has_logprobs: Optional[bool]
    temperature: float
    samples_per_task: int
    include_greedy_sample: bool
    greedy_temperature: float
    max_tokens: int
    notes: str


def load_models_config(path: str | Path) -> List[ModelConfig]:
    raw = load_yaml(path)
    defaults = raw.get("defaults", {})
    models = []
    for item in raw.get("models", []):
        model = ModelConfig(
            model_id=str(item["model_id"]),
            providers=list(item.get("providers", [])),
            has_logprobs=item.get("has_logprobs"),
            temperature=float(item.get("temperature", defaults.get("temperature", 0.7))),
            samples_per_task=int(item.get("samples_per_task", defaults.get("samples_per_task", 5))),
            include_greedy_sample=bool(
                item.get("include_greedy_sample", defaults.get("include_greedy_sample", True))
            ),
            greedy_temperature=float(
                item.get("greedy_temperature", defaults.get("greedy_temperature", 0.0))
            ),
            max_tokens=int(item.get("max_tokens", defaults.get("max_tokens", 512))),
            notes=str(item.get("notes", "")),
        )
        if not model.providers:
            raise ValueError(f"Model {model.model_id} has no providers configured.")
        models.append(model)
    if not models:
        raise ValueError("No models configured in models.yaml")
    return models

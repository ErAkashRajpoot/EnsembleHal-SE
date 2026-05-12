"""Cost tracking for API usage across models and providers."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


# Pricing per 1M tokens (input / output) — free tier or near-zero for these
DEFAULT_PRICING: Dict[str, Dict[str, float]] = {
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gemini-2.0-flash": {"input": 0.0, "output": 0.0},
    "gemini-2.0-flash-lite": {"input": 0.0, "output": 0.0},
    "llama-3.3-70b-versatile": {"input": 0.0, "output": 0.0},
    "llama-3.1-8b-instant": {"input": 0.0, "output": 0.0},
    "qwen/qwen3-32b": {"input": 0.0, "output": 0.0},
    # Offline synthetic models — zero cost
    "offline/model-alpha": {"input": 0.0, "output": 0.0},
    "offline/model-beta": {"input": 0.0, "output": 0.0},
    "offline/model-gamma": {"input": 0.0, "output": 0.0},
    "offline/model-delta": {"input": 0.0, "output": 0.0},
    "offline/model-epsilon": {"input": 0.0, "output": 0.0},
}


@dataclass
class ModelUsage:
    """Token usage tracker for a single model."""
    model_id: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    request_count: int = 0
    error_count: int = 0

    @property
    def estimated_cost(self) -> float:
        pricing = DEFAULT_PRICING.get(self.model_id, {"input": 0.0, "output": 0.0})
        input_cost = (self.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost


@dataclass
class CostTracker:
    """Aggregate cost tracker across all models."""
    _usage: Dict[str, ModelUsage] = field(default_factory=dict)

    def record(
        self,
        model_id: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        is_error: bool = False,
    ) -> None:
        key = f"{model_id}:{provider}"
        if key not in self._usage:
            self._usage[key] = ModelUsage(model_id=model_id, provider=provider)
        usage = self._usage[key]
        usage.input_tokens += input_tokens
        usage.output_tokens += output_tokens
        usage.request_count += 1
        if is_error:
            usage.error_count += 1

    @property
    def total_cost(self) -> float:
        return sum(u.estimated_cost for u in self._usage.values())

    def summary(self) -> List[Dict[str, object]]:
        rows = []
        for usage in sorted(self._usage.values(), key=lambda u: u.model_id):
            rows.append({
                "model_id": usage.model_id,
                "provider": usage.provider,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "requests": usage.request_count,
                "errors": usage.error_count,
                "estimated_cost_usd": round(usage.estimated_cost, 4),
            })
        rows.append({
            "model_id": "TOTAL",
            "provider": "",
            "input_tokens": sum(u.input_tokens for u in self._usage.values()),
            "output_tokens": sum(u.output_tokens for u in self._usage.values()),
            "requests": sum(u.request_count for u in self._usage.values()),
            "errors": sum(u.error_count for u in self._usage.values()),
            "estimated_cost_usd": round(self.total_cost, 4),
        })
        return rows

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.summary(), f, indent=2, ensure_ascii=True)

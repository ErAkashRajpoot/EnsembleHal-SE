"""Abstract base class for LLM API clients."""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GenerationResult:
    """Single generation output from an LLM."""
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    logprobs: Optional[Dict[str, Any]] = None
    finish_reason: str = ""
    model_id: str = ""
    provider: str = ""
    temperature: float = 0.0
    raw_response: Optional[Dict[str, Any]] = field(default=None, repr=False)


class BaseLLMClient(abc.ABC):
    """Unified interface for LLM API providers."""

    def __init__(self, model_id: str, provider: str, api_key: str, **kwargs: Any):
        self.model_id = model_id
        self.provider = provider
        self.api_key = api_key
        self.extra_config = kwargs

    @abc.abstractmethod
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        logprobs: bool = False,
        n: int = 1,
        **kwargs: Any,
    ) -> List[GenerationResult]:
        """Generate one or more completions for a prompt.

        Args:
            prompt: The input prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            logprobs: Whether to request log-probabilities.
            n: Number of completions to generate.

        Returns:
            List of GenerationResult objects.
        """

    @abc.abstractmethod
    async def close(self) -> None:
        """Release any held resources (HTTP sessions, etc.)."""

    async def __aenter__(self) -> "BaseLLMClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

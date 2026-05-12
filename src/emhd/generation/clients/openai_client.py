"""OpenAI-compatible API client (GPT-4o-mini, OpenRouter, etc.)."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

import aiohttp

from . import BaseLLMClient, GenerationResult

logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI Chat Completions API (and OpenAI-compatible endpoints)."""

    BASE_URL = "https://api.openai.com/v1"
    MAX_PROMPT_CHARS = 15000

    def __init__(
        self,
        model_id: str,
        api_key: str,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(model_id=model_id, provider="openai", api_key=api_key, **kwargs)
        self.base_url = (base_url or self.BASE_URL).rstrip("/")
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=120),
            )
        return self._session

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        logprobs: bool = False,
        n: int = 1,
        **kwargs: Any,
    ) -> List[GenerationResult]:
        session = await self._get_session()

        # Truncate very long prompts to avoid context-length errors
        if len(prompt) > self.MAX_PROMPT_CHARS:
            prompt = prompt[:self.MAX_PROMPT_CHARS] + "\n\n[...truncated]"

        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "n": n,
        }
        if logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = 5

        url = f"{self.base_url}/chat/completions"
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                error_body = await resp.text()
                logger.error(
                    "OpenAI API error %d from %s: %s",
                    resp.status, self.base_url, error_body[:500],
                )
                resp.raise_for_status()
            data = await resp.json()

        results: List[GenerationResult] = []
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)

        for choice in data.get("choices", []):
            message = choice.get("message", {})
            lp = choice.get("logprobs") if logprobs else None
            results.append(
                GenerationResult(
                    text=message.get("content", ""),
                    input_tokens=input_tokens,
                    output_tokens=usage.get("completion_tokens", 0) // max(n, 1),
                    logprobs=lp,
                    finish_reason=choice.get("finish_reason", ""),
                    model_id=self.model_id,
                    provider=self.provider,
                    temperature=temperature,
                    raw_response=choice,
                )
            )

        # Rate limit delay for free tiers
        await asyncio.sleep(1.5)

        return results

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

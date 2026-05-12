"""Together AI / Groq API client (OpenAI-compatible endpoints)."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

import aiohttp

from . import BaseLLMClient, GenerationResult

logger = logging.getLogger(__name__)


class TogetherClient(BaseLLMClient):
    """Client for Together AI / Groq (OpenAI-compatible chat completions)."""

    PROVIDER_URLS = {
        "together": "https://api.together.xyz/v1",
        "groq": "https://api.groq.com/openai/v1",
    }

    # Max input characters to avoid context-length 400 errors on free tiers
    MAX_PROMPT_CHARS = 12000

    def __init__(
        self,
        model_id: str,
        provider: str,
        api_key: str,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(model_id=model_id, provider=provider, api_key=api_key, **kwargs)
        self.base_url = (base_url or self.PROVIDER_URLS.get(provider, "")).rstrip("/")
        if not self.base_url:
            raise ValueError(f"Unknown provider: {provider}. Set base_url explicitly.")
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
        results: List[GenerationResult] = []

        # Truncate very long prompts to avoid 400 errors on free tiers
        if len(prompt) > self.MAX_PROMPT_CHARS:
            prompt = prompt[:self.MAX_PROMPT_CHARS] + "\n\n[...truncated]"

        for _ in range(n):
            payload: Dict[str, Any] = {
                "model": self.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if logprobs:
                payload["logprobs"] = True
                payload["top_logprobs"] = 5

            url = f"{self.base_url}/chat/completions"
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    error_body = await resp.text()
                    logger.error(
                        "API error %d from %s/%s: %s",
                        resp.status, self.provider, self.model_id, error_body[:500],
                    )
                    resp.raise_for_status()
                data = await resp.json()

            usage = data.get("usage", {})
            for choice in data.get("choices", []):
                message = choice.get("message", {})
                lp = choice.get("logprobs") if logprobs else None
                results.append(
                    GenerationResult(
                        text=message.get("content", ""),
                        input_tokens=usage.get("prompt_tokens", 0),
                        output_tokens=usage.get("completion_tokens", 0),
                        logprobs=lp,
                        finish_reason=choice.get("finish_reason", ""),
                        model_id=self.model_id,
                        provider=self.provider,
                        temperature=temperature,
                        raw_response=choice,
                    )
                )

            # Rate limit: small delay between requests (Groq free = 30 RPM)
            await asyncio.sleep(2.5)

        return results

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

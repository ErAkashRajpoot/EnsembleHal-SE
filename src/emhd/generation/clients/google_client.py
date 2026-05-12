"""Google Gemini API client."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

import aiohttp

from . import BaseLLMClient, GenerationResult

logger = logging.getLogger(__name__)


class GoogleClient(BaseLLMClient):
    """Client for Google Gemini generativeai REST API."""

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    MAX_PROMPT_CHARS = 15000

    def __init__(self, model_id: str, api_key: str, **kwargs: Any):
        super().__init__(model_id=model_id, provider="google", api_key=api_key, **kwargs)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Content-Type": "application/json"},
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

        # Truncate very long prompts
        if len(prompt) > self.MAX_PROMPT_CHARS:
            prompt = prompt[:self.MAX_PROMPT_CHARS] + "\n\n[...truncated]"

        # Gemini doesn't support n>1 natively; loop for multiple samples
        for _ in range(n):
            payload: Dict[str, Any] = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                },
            }

            url = (
                f"{self.BASE_URL}/models/{self.model_id}:generateContent"
                f"?key={self.api_key}"
            )
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    error_body = await resp.text()
                    logger.error(
                        "Gemini API error %d: %s", resp.status, error_body[:500],
                    )
                    resp.raise_for_status()
                data = await resp.json()

            candidates = data.get("candidates", [])
            usage = data.get("usageMetadata", {})
            text = ""
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                text = parts[0].get("text", "") if parts else ""

            results.append(
                GenerationResult(
                    text=text,
                    input_tokens=usage.get("promptTokenCount", 0),
                    output_tokens=usage.get("candidatesTokenCount", 0),
                    logprobs=None,
                    finish_reason=candidates[0].get("finishReason", "") if candidates else "",
                    model_id=self.model_id,
                    provider=self.provider,
                    temperature=temperature,
                    raw_response=data,
                )
            )

            # Rate limit: Gemini free tier = 15 RPM → 4 second delay
            await asyncio.sleep(4.5)

        return results

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

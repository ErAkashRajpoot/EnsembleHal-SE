"""Generation orchestrator: runs multi-LLM generation with retry and checkpointing."""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..config.models import ModelConfig
from .clients import BaseLLMClient, GenerationResult
from .clients.openai_client import OpenAIClient
from .clients.google_client import GoogleClient
from .clients.together_client import TogetherClient
from .clients.offline_client import OfflineClient
from .cost_tracker import CostTracker
from .prompts import get_prompt_template
from .registry import ExperimentRegistryWriter

logger = logging.getLogger(__name__)


def create_client(
    model: ModelConfig,
    provider: str,
    api_keys: Dict[str, str],
) -> BaseLLMClient:
    """Factory: create the right client for a model+provider pair."""
    key = api_keys.get(provider, "")
    if not key and provider != "offline":
        raise ValueError(f"No API key found for provider: {provider}")

    if provider == "openai":
        return OpenAIClient(model_id=model.model_id, api_key=key)
    elif provider == "openrouter":
        return OpenAIClient(
            model_id=model.model_id,
            api_key=key,
            base_url="https://openrouter.ai/api/v1",
        )
    elif provider == "google":
        return GoogleClient(model_id=model.model_id, api_key=key)
    elif provider in ("together", "groq"):
        return TogetherClient(model_id=model.model_id, provider=provider, api_key=key)
    elif provider == "offline":
        return OfflineClient(model_id=model.model_id)
    else:
        raise ValueError(f"Unknown provider: {provider}")


class GenerationOrchestrator:
    """Orchestrates multi-LLM generation across tasks.

    For each task and each model:
      - Generates k sampled outputs (temperature > 0)
      - Generates 1 greedy output (temperature = 0)
      - Records all outputs to the experiment registry
      - Tracks costs and checkpoints progress
    """

    def __init__(
        self,
        models: List[ModelConfig],
        api_keys: Dict[str, str],
        registry: ExperimentRegistryWriter,
        cost_tracker: CostTracker,
        checkpoint_dir: Path,
        max_concurrent: int = 5,
    ):
        self.models = models
        self.api_keys = api_keys
        self.registry = registry
        self.cost_tracker = cost_tracker
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_concurrent = max_concurrent
        self._completed: set = self._load_checkpoint()

    def _checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "generation_checkpoint.json"

    def _load_checkpoint(self) -> set:
        path = self._checkpoint_path()
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return set(data.get("completed", []))
        return set()

    def _save_checkpoint(self) -> None:
        path = self._checkpoint_path()
        with path.open("w", encoding="utf-8") as f:
            json.dump({"completed": sorted(self._completed)}, f, ensure_ascii=True)

    def _task_key(self, task_id: str, model_id: str, provider: str) -> str:
        return f"{task_id}::{model_id}::{provider}"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    async def _generate_with_retry(
        self,
        client: BaseLLMClient,
        prompt: str,
        temperature: float,
        max_tokens: int,
        logprobs: bool,
    ) -> List[GenerationResult]:
        return await client.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=logprobs,
        )

    async def _process_task_model(
        self,
        task: Dict[str, Any],
        model: ModelConfig,
        provider: str,
        semaphore: asyncio.Semaphore,
    ) -> int:
        """Generate all samples for one task+model pair."""
        task_id = task["task_id"]
        key = self._task_key(task_id, model.model_id, provider)

        if key in self._completed:
            return 0

        async with semaphore:
            template = get_prompt_template(task.get("artefact_type", "code"))
            rendered_prompt = template.render(prompt=task.get("prompt", ""))
            has_logprobs = model.has_logprobs if model.has_logprobs is not None else False

            client = create_client(model, provider, self.api_keys)
            count = 0

            try:
                # ── Sampled outputs (k samples) ──
                for sample_idx in range(model.samples_per_task):
                    try:
                        results = await self._generate_with_retry(
                            client,
                            rendered_prompt,
                            model.temperature,
                            model.max_tokens,
                            has_logprobs,
                        )
                        for result in results:
                            self.registry.write_entry(
                                result=result,
                                task_id=task_id,
                                dataset_source=task.get("dataset_source", ""),
                                artefact_type=task.get("artefact_type", "code"),
                                prompt_version=template.version,
                                hallucination_label=task.get("hallucination_label", 0),
                                contamination_flag=task.get("contamination_flag", False),
                                split=task.get("split", "train"),
                                generation_mode="sampled",
                                sample_index=sample_idx,
                                has_logprobs=has_logprobs,
                            )
                            self.cost_tracker.record(
                                model.model_id, provider,
                                result.input_tokens, result.output_tokens,
                            )
                            count += 1
                    except Exception as exc:
                        logger.error(
                            "Failed sample %d for task=%s model=%s: %s",
                            sample_idx, task_id, model.model_id, exc,
                        )
                        self.cost_tracker.record(
                            model.model_id, provider, 0, 0, is_error=True,
                        )

                # ── Greedy output (temperature = 0) ──
                if model.include_greedy_sample:
                    try:
                        results = await self._generate_with_retry(
                            client,
                            rendered_prompt,
                            model.greedy_temperature,
                            model.max_tokens,
                            has_logprobs,
                        )
                        for result in results:
                            self.registry.write_entry(
                                result=result,
                                task_id=task_id,
                                dataset_source=task.get("dataset_source", ""),
                                artefact_type=task.get("artefact_type", "code"),
                                prompt_version=template.version,
                                hallucination_label=task.get("hallucination_label", 0),
                                contamination_flag=task.get("contamination_flag", False),
                                split=task.get("split", "train"),
                                generation_mode="greedy",
                                sample_index=0,
                                has_logprobs=has_logprobs,
                            )
                            self.cost_tracker.record(
                                model.model_id, provider,
                                result.input_tokens, result.output_tokens,
                            )
                            count += 1
                    except Exception as exc:
                        logger.error(
                            "Failed greedy for task=%s model=%s: %s",
                            task_id, model.model_id, exc,
                        )

                self._completed.add(key)
                self._save_checkpoint()

            finally:
                await client.close()

            return count

    async def run(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run generation for all tasks across all models."""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        total_generated = 0
        total_tasks = len(tasks) * sum(len(m.providers) for m in self.models)

        logger.info("Starting generation: %d tasks x %d model-provider pairs", len(tasks), total_tasks // len(tasks))

        coroutines = []
        for task in tasks:
            for model in self.models:
                for provider in model.providers:
                    coroutines.append(
                        self._process_task_model(task, model, provider, semaphore)
                    )

        results = await asyncio.gather(*coroutines, return_exceptions=True)
        for r in results:
            if isinstance(r, int):
                total_generated += r
            elif isinstance(r, Exception):
                logger.error("Task failed: %s", r)

        return {
            "total_generated": total_generated,
            "total_cost_usd": round(self.cost_tracker.total_cost, 4),
            "completed_keys": len(self._completed),
        }

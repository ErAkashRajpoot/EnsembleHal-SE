"""CLI: Run multi-LLM generation across all tasks."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from dotenv import load_dotenv
import os

from emhd.config import load_datasets_config, load_models_config, load_project_config
from emhd.generation import GenerationOrchestrator, ExperimentRegistryWriter, CostTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-LLM generation.")
    parser.add_argument("--datasets", default="configs/datasets.yaml")
    parser.add_argument("--models", default="configs/models.yaml")
    parser.add_argument("--project", default="configs/project.yaml")
    parser.add_argument("--index", default="data/index/records.jsonl", help="Path to indexed records")
    parser.add_argument("--output", default="data/generations/registry.jsonl")
    parser.add_argument("--checkpoint-dir", default="data/generations/checkpoints")
    parser.add_argument("--max-concurrent", type=int, default=5)
    parser.add_argument("--max-tasks", type=int, default=50, help="Limit tasks (for testing)")
    parser.add_argument("--env-file", default=".env", help="Path to .env with API keys")
    return parser.parse_args()


def _load_tasks(index_path: Path, max_tasks: int = None):
    tasks = []
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))
    if max_tasks:
        tasks = tasks[:max_tasks]
    return tasks


def main() -> None:
    args = _parse_args()
    load_dotenv(args.env_file)

    api_keys = {
        "openai": os.getenv("OPENAI_API_KEY", ""),
        "openrouter": os.getenv("OPENROUTER_API_KEY", ""),
        "google": os.getenv("GOOGLE_API_KEY", ""),
        "together": os.getenv("TOGETHER_API_KEY", ""),
        "groq": os.getenv("GROQ_API_KEY", ""),
    }

    models = load_models_config(Path(args.models))
    registry = ExperimentRegistryWriter(Path(args.output))
    cost_tracker = CostTracker()

    tasks = _load_tasks(Path(args.index), args.max_tasks)
    logger.info("Loaded %d tasks for generation", len(tasks))

    orchestrator = GenerationOrchestrator(
        models=models, api_keys=api_keys, registry=registry,
        cost_tracker=cost_tracker, checkpoint_dir=Path(args.checkpoint_dir),
        max_concurrent=args.max_concurrent,
    )

    result = asyncio.run(orchestrator.run(tasks))
    logger.info("Generation complete: %s", result)

    cost_tracker.save(Path(args.checkpoint_dir) / "cost_report.json")
    logger.info("Cost report saved.")


if __name__ == "__main__":
    main()

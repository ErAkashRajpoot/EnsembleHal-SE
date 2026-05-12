"""Experiment registry writer with JSON Schema validation."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import jsonschema

from ..generation.clients import GenerationResult


_SCHEMA_PATH = Path(__file__).resolve().parents[3] / "schemas" / "experiment_registry.schema.json"


class ExperimentRegistryWriter:
    """Append experiment entries to a JSONL registry file."""

    def __init__(self, output_path: Path, schema_path: Optional[Path] = None):
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._schema: Optional[Dict[str, Any]] = None

        schema_file = schema_path or _SCHEMA_PATH
        if schema_file.exists():
            with schema_file.open("r", encoding="utf-8") as f:
                self._schema = json.load(f)

    def _validate(self, entry: Dict[str, Any]) -> None:
        if self._schema:
            jsonschema.validate(instance=entry, schema=self._schema)

    def write_entry(
        self,
        result: GenerationResult,
        task_id: str,
        dataset_source: str,
        artefact_type: str,
        prompt_version: str,
        hallucination_label: int,
        contamination_flag: bool,
        split: str,
        generation_mode: str,
        sample_index: int = 0,
        has_logprobs: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Write a single experiment entry to the registry."""
        entry: Dict[str, Any] = {
            "model_id": result.model_id,
            "model_provider": result.provider,
            "has_logprobs": has_logprobs if has_logprobs is not None else (result.logprobs is not None),
            "temperature": result.temperature,
            "max_tokens": 512,
            "artefact_type": artefact_type,
            "dataset_source": dataset_source,
            "task_id": task_id,
            "prompt_version": prompt_version,
            "hallucination_label": hallucination_label,
            "contamination_flag": contamination_flag,
            "split": split,
            "generation_mode": generation_mode,
            "sample_index": sample_index,
            "output_text": result.text,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "logprobs": result.logprobs,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        self._validate(entry)

        with self.output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=True) + "\n")

        return entry

    def write_batch(self, entries: List[Dict[str, Any]]) -> int:
        """Write multiple pre-built entries at once."""
        count = 0
        with self.output_path.open("a", encoding="utf-8") as f:
            for entry in entries:
                self._validate(entry)
                f.write(json.dumps(entry, ensure_ascii=True) + "\n")
                count += 1
        return count

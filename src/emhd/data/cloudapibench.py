from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .records import NormalizedRecord


def _load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}") from exc


def _parse_metadata_map(values: List[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError("metadata-map must be in raw_field=metadata_key format")
        raw_field, meta_key = item.split("=", 1)
        raw_field = raw_field.strip()
        meta_key = meta_key.strip()
        if not raw_field or not meta_key:
            raise ValueError("metadata-map must have non-empty raw_field and metadata_key")
        result[raw_field] = meta_key
    return result


def _slugify(value: str) -> str:
    cleaned = value.strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    return cleaned.strip("_")


def _fallback_task_id(raw: Dict[str, Any], index: int) -> str:
    api_name = raw.get("api_name") or raw.get("target_api") or raw.get("api")
    provider = raw.get("provider") or raw.get("api_provider") or raw.get("sdk")
    parts = ["cloudapibench"]
    if provider:
        parts.append(_slugify(str(provider)))
    if api_name:
        parts.append(_slugify(str(api_name)))
    parts.append(str(index))
    return "_".join(part for part in parts if part)


def normalize_cloudapibench(
    input_path: Path,
    output_path: Path,
    dataset_source: str,
    artefact_type: str,
    contamination_flag: bool,
    task_id_field: str,
    prompt_field: str,
    reference_field: str,
    label_field: str,
    metadata_fields: List[str],
    metadata_map: Dict[str, str],
) -> Tuple[int, int]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    defaults = {
        "dataset_source": dataset_source,
        "artefact_type": artefact_type,
        "contamination_flag": contamination_flag,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    skipped = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for index, raw in enumerate(_load_jsonl(input_path), start=1):
            task_id = str(raw.get(task_id_field, "")).strip() if task_id_field else ""
            if not task_id:
                task_id = _fallback_task_id(raw, index)

            prompt = str(raw.get(prompt_field, "")).strip()
            reference = str(raw.get(reference_field, "")).strip()
            label = raw.get(label_field)

            if not task_id or not prompt or not reference or label is None:
                skipped += 1
                continue

            metadata: Dict[str, Any] = {}
            for field_name in metadata_fields:
                if field_name in raw:
                    metadata[field_name] = raw[field_name]
            for raw_field, meta_key in metadata_map.items():
                if raw_field in raw:
                    metadata[meta_key] = raw[raw_field]
            if metadata:
                metadata = {
                    key: value
                    for key, value in metadata.items()
                    if value is not None and value != ""
                }

            payload = {
                "task_id": task_id,
                "prompt": prompt,
                "reference": reference,
                "hallucination_label": label,
            }
            if metadata:
                payload["metadata"] = metadata

            record = NormalizedRecord.from_dict(payload, defaults)
            handle.write(json.dumps(record.to_index_dict(), ensure_ascii=True) + "\n")
            count += 1

    return count, skipped

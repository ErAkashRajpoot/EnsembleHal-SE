from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List

from .records import NormalizedRecord


@dataclass(frozen=True)
class FieldMap:
    task_id: str
    prompt: str
    reference: str
    hallucination_label: str
    metadata_fields: List[str] = field(default_factory=list)
    metadata_map: Dict[str, str] = field(default_factory=dict)


def _load_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}") from exc


def normalize_jsonl(
    input_path: Path,
    output_path: Path,
    dataset_source: str,
    artefact_type: str,
    contamination_flag: bool,
    field_map: FieldMap,
    task_id_prefix: str = "",
) -> int:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    defaults = {
        "dataset_source": dataset_source,
        "artefact_type": artefact_type,
        "contamination_flag": contamination_flag,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for raw in _load_jsonl(input_path):
            task_id = raw.get(field_map.task_id, "")
            if task_id_prefix:
                task_id = f"{task_id_prefix}{task_id}"

            payload = {
                "task_id": task_id,
                "prompt": raw.get(field_map.prompt, ""),
                "reference": raw.get(field_map.reference, ""),
                "hallucination_label": raw.get(field_map.hallucination_label, None),
            }

            metadata: Dict[str, object] = {}
            for field_name in field_map.metadata_fields:
                if field_name in raw:
                    metadata[field_name] = raw[field_name]
            for raw_field, meta_key in field_map.metadata_map.items():
                if raw_field in raw:
                    metadata[meta_key] = raw[raw_field]
            if metadata:
                payload["metadata"] = metadata

            record = NormalizedRecord.from_dict(payload, defaults)
            handle.write(json.dumps(record.to_index_dict(), ensure_ascii=True) + "\n")
            count += 1

    return count

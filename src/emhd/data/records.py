from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable


ALLOWED_ARTEFACT_TYPES = {"code", "api", "test", "doc", "review", "explanation"}


@dataclass
class NormalizedRecord:
    task_id: str
    artefact_type: str
    prompt: str
    reference: str
    hallucination_label: int
    dataset_source: str
    contamination_flag: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any], defaults: Dict[str, Any]) -> "NormalizedRecord":
        task_id = str(raw.get("task_id", "")).strip()
        artefact_type = str(raw.get("artefact_type", defaults.get("artefact_type", ""))).strip()
        prompt = str(raw.get("prompt", ""))
        reference = str(raw.get("reference", ""))
        dataset_source = str(raw.get("dataset_source", defaults.get("dataset_source", ""))).strip()
        contamination_flag = bool(raw.get("contamination_flag", defaults.get("contamination_flag", False)))
        label_raw = raw.get("hallucination_label", None)

        if not task_id:
            raise ValueError("task_id is required")
        if artefact_type not in ALLOWED_ARTEFACT_TYPES:
            raise ValueError(f"Invalid artefact_type: {artefact_type}")
        if not dataset_source:
            raise ValueError("dataset_source is required")
        if prompt == "":
            raise ValueError("prompt is required")
        if reference == "":
            raise ValueError("reference is required")
        if label_raw is None:
            raise ValueError("hallucination_label is required")

        try:
            label = int(label_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("hallucination_label must be an integer") from exc
        if label < 0 or label > 5:
            raise ValueError("hallucination_label must be between 0 and 5")

        metadata = raw.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a JSON object")

        return cls(
            task_id=task_id,
            artefact_type=artefact_type,
            prompt=prompt,
            reference=reference,
            hallucination_label=label,
            dataset_source=dataset_source,
            contamination_flag=contamination_flag,
            metadata=metadata,
        )

    def to_index_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "artefact_type": self.artefact_type,
            "prompt": self.prompt,
            "reference": self.reference,
            "hallucination_label": self.hallucination_label,
            "dataset_source": self.dataset_source,
            "contamination_flag": self.contamination_flag,
            "metadata": self.metadata,
        }


def ensure_unique_task_ids(records: Iterable[NormalizedRecord]) -> None:
    seen = set()
    for record in records:
        if record.task_id in seen:
            raise ValueError(f"Duplicate task_id detected: {record.task_id}")
        seen.add(record.task_id)

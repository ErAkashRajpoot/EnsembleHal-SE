from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from .io import load_yaml


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    path: Path
    artefact_type: str
    contamination_flag: bool
    format: str


def load_datasets_config(path: str | Path) -> List[DatasetConfig]:
    raw = load_yaml(path)
    defaults = raw.get("defaults", {})
    datasets = []
    for item in raw.get("datasets", []):
        dataset = DatasetConfig(
            name=str(item["name"]),
            path=Path(item["path"]),
            artefact_type=str(item.get("artefact_type", "")),
            contamination_flag=bool(item.get("contamination_flag", defaults.get("contamination_flag", False))),
            format=str(item.get("format", defaults.get("format", "normalized_jsonl"))),
        )
        if not dataset.artefact_type:
            raise ValueError(f"Dataset {dataset.name} has no artefact_type.")
        datasets.append(dataset)
    if not datasets:
        raise ValueError("No datasets configured in datasets.yaml")
    return datasets

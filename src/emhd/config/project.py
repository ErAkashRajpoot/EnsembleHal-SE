from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from .io import load_yaml


@dataclass(frozen=True)
class ProjectConfig:
    random_seed: int
    index_output_dir: Path


def load_project_config(path: str | Path) -> ProjectConfig:
    raw: Dict[str, Any] = load_yaml(path)
    return ProjectConfig(
        random_seed=int(raw.get("random_seed", 42)),
        index_output_dir=Path(raw.get("index_output_dir", "data/index")),
    )

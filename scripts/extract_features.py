"""CLI: Extract features from generation outputs."""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from emhd.features import FeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract features from generations.")
    parser.add_argument("--registry", default="data/generations/registry.jsonl")
    parser.add_argument("--index", default="data/index/records.jsonl")
    parser.add_argument("--output", default="data/features/features.csv")
    return parser.parse_args()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    args = _parse_args()

    # Load task index
    tasks = {r["task_id"]: r for r in _load_jsonl(Path(args.index))}
    logger.info("Loaded %d tasks from index", len(tasks))

    # Load generation outputs, group by task_id
    registry = _load_jsonl(Path(args.registry))
    task_outputs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for entry in registry:
        task_outputs[entry["task_id"]].append(entry)

    logger.info("Loaded %d generation entries for %d tasks", len(registry), len(task_outputs))

    # Extract features
    extractor = FeatureExtractor()
    feature_tasks = []
    for task_id, outputs in task_outputs.items():
        task_info = tasks.get(task_id, {})
        feature_tasks.append({
            "task_id": task_id,
            "outputs": [o.get("output_text", "") for o in outputs],
            "reference": task_info.get("reference", ""),
            "artefact_type": task_info.get("artefact_type", "code"),
            "prompt": task_info.get("prompt", ""),
            "dataset_source": task_info.get("dataset_source", ""),
            "hallucination_label": task_info.get("hallucination_label", 0),
            "logprobs_list": [o.get("logprobs") for o in outputs],
            "has_logprobs_flags": [o.get("has_logprobs", False) for o in outputs],
        })

    df = extractor.extract_batch(feature_tasks)
    output_path = Path(args.output)
    extractor.save_features(df, output_path)
    logger.info("Saved %d feature rows (%d columns) to %s", len(df), len(df.columns), output_path)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from .hash_utils import task_hash
from .records import NormalizedRecord, ensure_unique_task_ids
from .splitter import stratified_split


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


def load_dataset_records(
    path: Path,
    dataset_source: str,
    artefact_type: str,
    contamination_flag: bool,
) -> List[NormalizedRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    defaults = {
        "dataset_source": dataset_source,
        "artefact_type": artefact_type,
        "contamination_flag": contamination_flag,
    }
    records: List[NormalizedRecord] = []
    for raw in _load_jsonl(path):
        record = NormalizedRecord.from_dict(raw, defaults)
        records.append(record)
    return records


def build_index(
    datasets: List[Tuple[str, Path, str, bool]],
    seed: int,
    output_dir: Path,
) -> Dict[str, object]:
    all_records: List[NormalizedRecord] = []
    for dataset_source, path, artefact_type, contamination_flag in datasets:
        all_records.extend(
            load_dataset_records(path, dataset_source, artefact_type, contamination_flag)
        )

    if not all_records:
        raise ValueError("No records loaded from datasets.")

    ensure_unique_task_ids(all_records)

    hash_map: Dict[str, List[str]] = {}
    for record in all_records:
        h = task_hash(record)
        hash_map.setdefault(h, []).append(record.task_id)

    overlaps = {h: ids for h, ids in hash_map.items() if len(ids) > 1}

    splits = stratified_split(all_records, seed=seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    records_path = output_dir / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for split_name, indices in splits.items():
            for idx in indices:
                record = all_records[idx]
                payload = record.to_index_dict()
                payload["task_hash"] = task_hash(record)
                payload["split"] = split_name
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    counts = {
        "total": len(all_records),
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"]),
    }

    pd_records = pd.DataFrame([r.to_index_dict() for r in all_records])
    balance = pd_records.groupby(["dataset_source", "artefact_type", "hallucination_label"]).size()
    balance_df = balance.reset_index(name="count")
    balance_df.to_csv(output_dir / "class_balance.csv", index=False)

    with (output_dir / "overlap_report.json").open("w", encoding="utf-8") as handle:
        json.dump(overlaps, handle, ensure_ascii=True, indent=2)

    with (output_dir / "splits.json").open("w", encoding="utf-8") as handle:
        json.dump(counts, handle, ensure_ascii=True, indent=2)

    return {
        "records_path": str(records_path),
        "splits": counts,
        "overlap_count": len(overlaps),
    }

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List


def _load_jsonl(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}") from exc
    return records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build labeling set for Defects4J bug explanations.")
    parser.add_argument("--input", required=True, help="Path to raw Defects4J JSONL file")
    parser.add_argument("--output", required=True, help="Path to output labeling JSONL file")
    parser.add_argument("--sample", type=int, default=400, help="Number of records to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = _load_jsonl(input_path)
    if not records:
        raise ValueError(f"No records found in {input_path}")

    random.seed(args.seed)
    sample_size = min(args.sample, len(records))
    sample = random.sample(records, sample_size)

    with output_path.open("w", encoding="utf-8") as handle:
        for raw in sample:
            payload = {
                "project_id": raw.get("project_id"),
                "bug_id": raw.get("bug_id"),
                "report_url": raw.get("report_url"),
                "report_title": raw.get("report_title"),
                "report_body": raw.get("report_body"),
                "diff": raw.get("diff"),
                "explanation": "",
                "hallucination_label": "",
                "notes": "",
            }
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    print(f"Wrote {sample_size} labeling records to {output_path}")


if __name__ == "__main__":
    main()

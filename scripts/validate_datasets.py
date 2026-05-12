from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from emhd.config import load_datasets_config
from emhd.data.indexer import load_dataset_records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate normalized datasets.")
    parser.add_argument("--datasets", required=True, help="Path to datasets.yaml")
    parser.add_argument("--output", help="Optional CSV summary output path")
    return parser.parse_args()


def _summarize_labels(records) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for record in records:
        counts[record.hallucination_label] = counts.get(record.hallucination_label, 0) + 1
    return counts


def main() -> None:
    args = _parse_args()
    datasets = load_datasets_config(Path(args.datasets))

    summary_rows: List[Dict[str, object]] = []

    for dataset in datasets:
        row: Dict[str, object] = {
            "dataset": dataset.name,
            "path": str(dataset.path),
            "artefact_type": dataset.artefact_type,
            "contamination_flag": dataset.contamination_flag,
            "status": "ok",
            "count": 0,
            "label_counts": "",
        }
        try:
            records = load_dataset_records(
                path=dataset.path,
                dataset_source=dataset.name,
                artefact_type=dataset.artefact_type,
                contamination_flag=dataset.contamination_flag,
            )
        except FileNotFoundError:
            row["status"] = "missing"
            summary_rows.append(row)
            continue

        row["count"] = len(records)
        label_counts = _summarize_labels(records)
        row["label_counts"] = ";".join(
            f"{label}:{count}" for label, count in sorted(label_counts.items())
        )
        summary_rows.append(row)

    df = pd.DataFrame(summary_rows)
    print(df.to_string(index=False))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"Summary written to {args.output}")


if __name__ == "__main__":
    main()

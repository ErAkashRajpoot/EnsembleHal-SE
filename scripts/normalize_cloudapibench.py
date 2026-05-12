from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from emhd.data.cloudapibench import normalize_cloudapibench, _parse_metadata_map


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize CloudAPIBench JSONL tasks.")
    parser.add_argument("--input", required=True, help="Path to raw CloudAPIBench JSONL file")
    parser.add_argument("--output", required=True, help="Path to output normalized JSONL file")
    parser.add_argument("--dataset-source", default="cloudapibench", help="Dataset source name")
    parser.add_argument("--artefact-type", default="api", help="Artefact type")
    parser.add_argument("--contamination-flag", action="store_true", help="Set contamination_flag")
    parser.add_argument("--task-id-field", default="", help="Optional explicit task_id field")
    parser.add_argument("--prompt-field", default="prompt", help="Field name for prompt")
    parser.add_argument("--reference-field", default="reference", help="Field name for reference")
    parser.add_argument("--label-field", default="hallucination_label", help="Field name for label")
    parser.add_argument(
        "--metadata-field",
        action="append",
        default=[],
        help="Field name to copy into metadata (repeatable)",
    )
    parser.add_argument(
        "--metadata-map",
        action="append",
        default=[],
        help="Mapping raw_field=metadata_key (repeatable)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    metadata_map = _parse_metadata_map(args.metadata_map)
    count, skipped = normalize_cloudapibench(
        input_path=Path(args.input),
        output_path=Path(args.output),
        dataset_source=args.dataset_source,
        artefact_type=args.artefact_type,
        contamination_flag=bool(args.contamination_flag),
        task_id_field=args.task_id_field,
        prompt_field=args.prompt_field,
        reference_field=args.reference_field,
        label_field=args.label_field,
        metadata_fields=list(args.metadata_field),
        metadata_map=metadata_map,
    )
    print(f"Wrote {count} records to {args.output}")
    if skipped:
        print(f"Skipped {skipped} records with missing required fields")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from emhd.data.normalize import FieldMap, normalize_jsonl


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize a JSONL dataset.")
    parser.add_argument("--input", required=True, help="Path to raw JSONL file")
    parser.add_argument("--output", required=True, help="Path to output normalized JSONL file")
    parser.add_argument("--dataset-source", required=True, help="Dataset source name")
    parser.add_argument("--artefact-type", required=True, help="Artefact type")
    parser.add_argument("--contamination-flag", action="store_true", help="Set contamination_flag")
    parser.add_argument("--task-id-field", required=True, help="Field name for task_id")
    parser.add_argument("--prompt-field", required=True, help="Field name for prompt")
    parser.add_argument("--reference-field", required=True, help="Field name for reference")
    parser.add_argument("--label-field", required=True, help="Field name for hallucination_label")
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
    parser.add_argument("--task-id-prefix", default="", help="Prefix for task_id values")
    return parser.parse_args()


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


def main() -> None:
    args = _parse_args()
    field_map = FieldMap(
        task_id=args.task_id_field,
        prompt=args.prompt_field,
        reference=args.reference_field,
        hallucination_label=args.label_field,
        metadata_fields=args.metadata_field,
        metadata_map=_parse_metadata_map(args.metadata_map),
    )

    count = normalize_jsonl(
        input_path=Path(args.input),
        output_path=Path(args.output),
        dataset_source=args.dataset_source,
        artefact_type=args.artefact_type,
        contamination_flag=bool(args.contamination_flag),
        field_map=field_map,
        task_id_prefix=args.task_id_prefix,
    )

    print(f"Wrote {count} records to {args.output}")


if __name__ == "__main__":
    main()

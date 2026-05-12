from __future__ import annotations

import argparse
from pathlib import Path

from emhd.data.codehalu import normalize_codehalu


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize CodeHaluEval JSON files.")
    parser.add_argument("--input-dir", required=True, help="Directory with CodeHalu JSON files")
    parser.add_argument("--output", required=True, help="Path to output normalized JSONL file")
    parser.add_argument("--dataset-source", default="codehalu", help="Dataset source name")
    parser.add_argument("--artefact-type", default="code", help="Artefact type")
    parser.add_argument("--contamination-flag", action="store_true", help="Set contamination_flag")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    count, skipped = normalize_codehalu(
        input_dir=Path(args.input_dir),
        output_path=Path(args.output),
        dataset_source=args.dataset_source,
        artefact_type=args.artefact_type,
        contamination_flag=bool(args.contamination_flag),
    )
    print(f"Wrote {count} records to {args.output}")
    if skipped:
        print(f"Skipped {skipped} records with empty prompt or reference")


if __name__ == "__main__":
    main()

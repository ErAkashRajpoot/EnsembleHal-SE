from __future__ import annotations

import argparse
from pathlib import Path

from emhd.data.defects4j import normalize_defects4j


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize Defects4J bug explanation data.")
    parser.add_argument("--input", required=True, help="Path to raw Defects4J JSONL file")
    parser.add_argument("--output", required=True, help="Path to output normalized JSONL file")
    parser.add_argument("--dataset-source", default="defects4j", help="Dataset source name")
    parser.add_argument("--artefact-type", default="explanation", help="Artefact type")
    parser.add_argument("--contamination-flag", action="store_true", help="Set contamination_flag")
    parser.add_argument("--task-id-field", default="", help="Optional explicit task_id field")
    parser.add_argument("--project-id-field", default="project_id", help="Field name for project id")
    parser.add_argument("--bug-id-field", default="bug_id", help="Field name for bug id")
    parser.add_argument("--report-title-field", default="report_title", help="Field name for report title")
    parser.add_argument("--report-body-field", default="report_body", help="Field name for report body")
    parser.add_argument("--report-url-field", default="report_url", help="Field name for report URL")
    parser.add_argument("--report-id-field", default="report_id", help="Field name for report id")
    parser.add_argument("--diff-field", default="diff", help="Field name for diff text")
    parser.add_argument("--buggy-rev-field", default="revision_id_buggy", help="Field name for buggy revision")
    parser.add_argument("--fixed-rev-field", default="revision_id_fixed", help="Field name for fixed revision")
    parser.add_argument("--reference-field", default="explanation", help="Field name for explanation text")
    parser.add_argument("--label-field", default="hallucination_label", help="Field name for label")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    count, skipped = normalize_defects4j(
        input_path=Path(args.input),
        output_path=Path(args.output),
        dataset_source=args.dataset_source,
        artefact_type=args.artefact_type,
        contamination_flag=bool(args.contamination_flag),
        task_id_field=args.task_id_field,
        project_id_field=args.project_id_field,
        bug_id_field=args.bug_id_field,
        report_title_field=args.report_title_field,
        report_body_field=args.report_body_field,
        report_url_field=args.report_url_field,
        report_id_field=args.report_id_field,
        diff_field=args.diff_field,
        buggy_rev_field=args.buggy_rev_field,
        fixed_rev_field=args.fixed_rev_field,
        reference_field=args.reference_field,
        label_field=args.label_field,
    )
    print(f"Wrote {count} records to {args.output}")
    if skipped:
        print(f"Skipped {skipped} records with missing required fields")


if __name__ == "__main__":
    main()

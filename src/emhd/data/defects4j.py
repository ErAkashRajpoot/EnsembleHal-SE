from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from .records import NormalizedRecord


def _load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}") from exc


def _build_prompt(report_title: str, report_body: str, diff_text: str) -> str:
    sections = []
    report_title = report_title.strip()
    report_body = report_body.strip()
    diff_text = diff_text.strip()

    if report_title or report_body:
        report_parts = []
        if report_title:
            report_parts.append(report_title)
        if report_body:
            report_parts.append(report_body)
        sections.append("Bug report:\n" + "\n\n".join(report_parts))

    if diff_text:
        sections.append("Fix diff:\n" + diff_text)

    return "\n\n".join(sections).strip()


def normalize_defects4j(
    input_path: Path,
    output_path: Path,
    dataset_source: str,
    artefact_type: str,
    contamination_flag: bool,
    task_id_field: str,
    project_id_field: str,
    bug_id_field: str,
    report_title_field: str,
    report_body_field: str,
    report_url_field: str,
    report_id_field: str,
    diff_field: str,
    buggy_rev_field: str,
    fixed_rev_field: str,
    reference_field: str,
    label_field: str,
) -> Tuple[int, int]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    defaults = {
        "dataset_source": dataset_source,
        "artefact_type": artefact_type,
        "contamination_flag": contamination_flag,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    skipped = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for raw in _load_jsonl(input_path):
            task_id = str(raw.get(task_id_field, "")).strip() if task_id_field else ""
            project_id = str(raw.get(project_id_field, "")).strip()
            bug_id = str(raw.get(bug_id_field, "")).strip()
            if not task_id and project_id and bug_id:
                task_id = f"defects4j_{project_id}_{bug_id}"

            prompt = _build_prompt(
                str(raw.get(report_title_field, "")),
                str(raw.get(report_body_field, "")),
                str(raw.get(diff_field, "")),
            )
            reference = str(raw.get(reference_field, "")).strip()
            label = raw.get(label_field)

            if not task_id or not prompt or not reference or label is None:
                skipped += 1
                continue

            metadata = {
                "project_id": project_id,
                "bug_id": bug_id,
                "report_url": raw.get(report_url_field),
                "report_id": raw.get(report_id_field),
                "revision_id_buggy": raw.get(buggy_rev_field),
                "revision_id_fixed": raw.get(fixed_rev_field),
            }
            metadata = {
                key: value
                for key, value in metadata.items()
                if value is not None and value != ""
            }

            payload = {
                "task_id": task_id,
                "prompt": prompt,
                "reference": reference,
                "hallucination_label": label,
                "metadata": metadata,
            }

            record = NormalizedRecord.from_dict(payload, defaults)
            handle.write(json.dumps(record.to_index_dict(), ensure_ascii=True) + "\n")
            count += 1

    return count, skipped

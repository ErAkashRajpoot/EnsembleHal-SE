from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .records import NormalizedRecord

HALU_CATEGORY_LABELS = {
    "Mapping": 2,
    "Naming": 3,
    "Resource": 3,
    "Logic": 2,
}

HALU_SUBTYPE_TO_CATEGORY = {
    "data_compliance": "Mapping",
    "structural_access": "Mapping",
    "identification": "Naming",
    "external_source": "Naming",
    "physical_constraint": "Resource",
    "calculate_boundary": "Resource",
    "logic_deviation": "Logic",
    "logic_breakdown": "Logic",
}

HALU_SUBTYPE_ALIASES = {
    "data_compliance": "data_compliance",
    "structural_access": "structural_access",
    "structure_access": "structural_access",
    "identification": "identification",
    "identity": "identification",
    "external_source": "external_source",
    "external_source_hallucination": "external_source",
    "physical_constraint": "physical_constraint",
    "calculate_boundary": "calculate_boundary",
    "calculation_boundary": "calculate_boundary",
    "computational_boundary": "calculate_boundary",
    "logic_deviation": "logic_deviation",
    "logic_breakdown": "logic_breakdown",
}


def _normalize_halu_key(value: str) -> str:
    cleaned = value.lower().strip()
    cleaned = cleaned.replace("hallucination", "")
    cleaned = cleaned.replace("-", " ").replace("_", " ")
    cleaned = " ".join(cleaned.split())
    return cleaned.replace(" ", "_")


def _coerce_solution_text(solutions: Any) -> str:
    if solutions is None:
        return ""
    if isinstance(solutions, str):
        try:
            parsed = json.loads(solutions)
        except json.JSONDecodeError:
            parsed = [solutions]
    else:
        parsed = solutions
    if isinstance(parsed, list):
        parts = [str(item).strip() for item in parsed if str(item).strip()]
        return "\n\n".join(parts)
    return str(parsed).strip()


def _build_prompt(question: str, starter_code: str) -> str:
    question = question.strip()
    starter_code = starter_code.strip()
    if starter_code:
        return f"{question}\n\nStarter code:\n{starter_code}"
    return question


def _resolve_halu_subtype(halu_type: str, file_stem: str) -> str:
    if halu_type:
        key = _normalize_halu_key(halu_type)
    else:
        key = _normalize_halu_key(file_stem)
    key = key.replace("_hallucination", "")
    return HALU_SUBTYPE_ALIASES.get(key, key)


def _label_for_halu_subtype(subtype: str) -> Tuple[int, str]:
    if subtype not in HALU_SUBTYPE_TO_CATEGORY:
        raise ValueError(f"Unknown CodeHalu subtype: {subtype}")
    category = HALU_SUBTYPE_TO_CATEGORY[subtype]
    label = HALU_CATEGORY_LABELS[category]
    return label, category


def _load_json_array(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return data


def normalize_codehalu(
    input_dir: Path,
    output_path: Path,
    dataset_source: str,
    artefact_type: str,
    contamination_flag: bool,
) -> Tuple[int, int]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    json_files = sorted(p for p in input_dir.glob("*.json") if p.is_file())
    if not json_files:
        raise ValueError(f"No .json files found in {input_dir}")

    defaults = {
        "dataset_source": dataset_source,
        "artefact_type": artefact_type,
        "contamination_flag": contamination_flag,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    skipped = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for path in json_files:
            for raw in _load_json_array(path):
                halu_type_raw = str(raw.get("halu_type", "") or "").strip()
                subtype = _resolve_halu_subtype(halu_type_raw, path.stem)
                label, category = _label_for_halu_subtype(subtype)

                prompt = _build_prompt(
                    str(raw.get("question", "")),
                    str(raw.get("starter_code", "")),
                )
                reference = _coerce_solution_text(raw.get("solutions"))

                raw_id = raw.get("id")
                task_id_raw = raw.get("task_id")
                test_case_id = raw.get("test_case_id")
                suffix_parts = [
                    str(part)
                    for part in (raw_id, task_id_raw, test_case_id)
                    if part is not None and str(part).strip() != ""
                ]
                suffix = "_".join(suffix_parts) if suffix_parts else "record"
                task_id = f"codehalu_{path.stem}_{suffix}"

                metadata = {
                    "halu_type": halu_type_raw,
                    "halu_subtype": subtype,
                    "halu_category": category,
                    "difficulty": raw.get("difficulty"),
                    "fn_name": raw.get("fn_name"),
                    "task_id_raw": task_id_raw,
                    "test_case_id": test_case_id,
                    "url": raw.get("url"),
                    "input": raw.get("input"),
                    "output": raw.get("output"),
                    "source_file": path.name,
                }
                metadata = {
                    key: value
                    for key, value in metadata.items()
                    if value is not None and value != ""
                }

                if not prompt or not reference:
                    skipped += 1
                    continue

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

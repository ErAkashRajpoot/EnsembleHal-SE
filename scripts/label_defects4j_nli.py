from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from transformers import pipeline


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


def _build_report_text(title: str, body: str) -> str:
    title = (title or "").strip()
    body = (body or "").strip()
    if title and body:
        return f"{title}\n\n{body}"
    return title or body


def _truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars].rstrip() + "..."


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label Defects4J bug explanations using NLI.")
    parser.add_argument("--input", required=True, help="Path to raw Defects4J JSONL file")
    parser.add_argument("--output", required=True, help="Path to labeled JSONL file")
    parser.add_argument(
        "--model",
        default="cross-encoder/nli-deberta-v3-base",
        help="HuggingFace NLI model id",
    )
    parser.add_argument(
        "--contradiction-threshold",
        type=float,
        default=0.5,
        help="Threshold for contradiction label",
    )
    parser.add_argument(
        "--max-diff-chars",
        type=int,
        default=4000,
        help="Max characters of diff text for NLI input",
    )
    parser.add_argument(
        "--max-report-chars",
        type=int,
        default=1200,
        help="Max characters of report text for NLI input",
    )
    return parser.parse_args()


def _score_pair(nli_pipe, premise: str, hypothesis: str) -> Dict[str, float]:
    scores = nli_pipe(
        {"text": premise, "text_pair": hypothesis},
        top_k=None,
        truncation=True,
    )
    if scores and isinstance(scores[0], list):
        scores = scores[0]
    return {item["label"].lower(): float(item["score"]) for item in scores}


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nli_pipe = pipeline("text-classification", model=args.model)

    labeled = 0
    skipped = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for raw in _load_jsonl(input_path):
            report_text = _build_report_text(
                str(raw.get("report_title", "")),
                str(raw.get("report_body", "")),
            )
            diff_text = str(raw.get("diff", ""))

            report_text = _truncate(report_text, args.max_report_chars)
            diff_text = _truncate(diff_text, args.max_diff_chars)

            if not report_text or not diff_text:
                skipped += 1
                continue

            scores = _score_pair(nli_pipe, diff_text, report_text)
            contradiction = scores.get("contradiction", 0.0)
            entailment = scores.get("entailment", 0.0)

            label = 1 if contradiction >= args.contradiction_threshold and contradiction > entailment else 0

            payload = {
                **raw,
                "explanation": report_text,
                "hallucination_label": label,
                "nli_scores": scores,
            }
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
            labeled += 1

    print(f"Labeled {labeled} records to {output_path}")
    if skipped:
        print(f"Skipped {skipped} records with missing report/diff")


if __name__ == "__main__":
    main()

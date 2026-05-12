"""Download and normalize missing datasets from HuggingFace.

Acquires: HumanEval, MBPP, CodeReviewer, TL-CodeSum.
CodeMirage is handled separately (GitHub-hosted).
CloudAPIBench is excluded (unavailable).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset


# ── HumanEval ────────────────────────────────────────────────────────────────

def download_humaneval(output_dir: Path) -> int:
    """Download OpenAI HumanEval and normalize to JSONL."""
    ds = load_dataset("openai/openai_humaneval", split="test")
    output_path = output_dir / "humaneval" / "normalized.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for row in ds:
            task_id = row.get("task_id", f"humaneval_{count}")
            prompt = row.get("prompt", "")
            canonical = row.get("canonical_solution", "")
            entry_point = row.get("entry_point", "")
            test_code = row.get("test", "")

            # For HumanEval, label is unknown (needs execution-based labeling)
            # Default to 0 (no hallucination) since these are human-written references
            record = {
                "task_id": str(task_id).replace("/", "_"),
                "artefact_type": "code",
                "prompt": prompt,
                "reference": canonical,
                "hallucination_label": 0,
                "dataset_source": "humaneval",
                "contamination_flag": True,
                "metadata": {
                    "entry_point": entry_point,
                    "test": test_code,
                },
            }
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
            count += 1

    print(f"[HumanEval] Wrote {count} records to {output_path}")
    return count


# ── MBPP ─────────────────────────────────────────────────────────────────────

def download_mbpp(output_dir: Path) -> int:
    """Download Google MBPP and normalize to JSONL."""
    ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
    output_path = output_dir / "mbpp" / "normalized.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for row in ds:
            task_id = f"mbpp_{row.get('task_id', count)}"
            prompt = row.get("text", "")
            code = row.get("code", "")
            test_list = row.get("test_list", [])

            record = {
                "task_id": task_id,
                "artefact_type": "code",
                "prompt": prompt,
                "reference": code,
                "hallucination_label": 0,
                "dataset_source": "mbpp",
                "contamination_flag": True,
                "metadata": {
                    "test_list": test_list,
                },
            }
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
            count += 1

    print(f"[MBPP] Wrote {count} records to {output_path}")
    return count


# ── CodeReviewer ─────────────────────────────────────────────────────────────

def download_codereviewer(output_dir: Path) -> int:
    """Download CodeReviewer-style code review data and normalize to JSONL.

    Tries multiple HuggingFace dataset identifiers; falls back to
    code_x_glue_cc_code_refinement if none work.
    """
    ds = None
    tried = []

    # Try various known HuggingFace identifiers
    candidates = [
        ("microsoft/code-reviewer", None, "test"),
        ("microsoft/CodeReviewer", "cls", "test"),
        ("microsoft/CodeReviewer", "msg", "test"),
        # Fallback: code refinement dataset (diff → fixed code)
        ("code_x_glue_cc_code_refinement", "medium", "test"),
    ]

    for name, config, split in candidates:
        try:
            if config:
                ds = load_dataset(name, config, split=split)
            else:
                ds = load_dataset(name, split=split)
            print(f"[CodeReviewer] Loaded {name} (config={config})")
            break
        except Exception as exc:
            tried.append(f"{name}/{config}: {exc}")
            continue

    if ds is None:
        print(f"[CodeReviewer] All sources failed:")
        for t in tried:
            print(f"  - {t}")
        return 0

    output_path = output_dir / "codereviewer" / "normalized.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(ds):
            # Handle different column formats
            diff = row.get("patch", row.get("diff", row.get("buggy", "")))
            msg = row.get("msg", row.get("review_comment",
                   row.get("fixed", row.get("comment", ""))))

            if not diff or not msg:
                continue

            record = {
                "task_id": f"codereviewer_{idx}",
                "artefact_type": "review",
                "prompt": diff,
                "reference": msg,
                "hallucination_label": 0,
                "dataset_source": "codereviewer",
                "contamination_flag": False,
                "metadata": {},
            }
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
            count += 1

    print(f"[CodeReviewer] Wrote {count} records to {output_path}")
    return count


# ── TL-CodeSum ───────────────────────────────────────────────────────────────

def download_tlcodesum(output_dir: Path) -> int:
    """Download TL-CodeSum (code summarization) and normalize to JSONL."""
    try:
        ds = load_dataset("code_x_glue_ct_code_to_text", "java", split="test")
    except Exception:
        ds = load_dataset("microsoft/codexglue_code_to_text", "java", split="test")

    output_path = output_dir / "tl-codesum" / "normalized.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(ds):
            code = row.get("code", row.get("original_string", ""))
            summary = row.get("docstring", row.get("code_tokens", ""))

            if isinstance(summary, list):
                summary = " ".join(summary)
            if isinstance(code, list):
                code = " ".join(code)

            if not code or not summary:
                continue

            record = {
                "task_id": f"tlcodesum_{idx}",
                "artefact_type": "doc",
                "prompt": code,
                "reference": summary,
                "hallucination_label": 0,  # Ground truth summaries
                "dataset_source": "tl-codesum",
                "contamination_flag": False,
                "metadata": {},
            }
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
            count += 1

    print(f"[TL-CodeSum] Wrote {count} records to {output_path}")
    return count


# ── Main ─────────────────────────────────────────────────────────────────────

DOWNLOADERS = {
    "humaneval": download_humaneval,
    "mbpp": download_mbpp,
    "codereviewer": download_codereviewer,
    "tlcodesum": download_tlcodesum,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and normalize datasets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DOWNLOADERS.keys()) + ["all"],
        default=["all"],
        help="Datasets to download",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Base output directory for raw data",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    targets = list(DOWNLOADERS.keys()) if "all" in args.datasets else args.datasets

    for name in targets:
        print(f"\n{'='*60}")
        print(f"Downloading: {name}")
        print(f"{'='*60}")
        try:
            DOWNLOADERS[name](output_dir)
        except Exception as exc:
            print(f"[ERROR] Failed to download {name}: {exc}")
            continue


if __name__ == "__main__":
    main()

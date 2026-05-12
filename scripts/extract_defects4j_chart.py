from __future__ import annotations

import argparse
import csv
import json
import re
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Iterable, Tuple


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []

    def handle_data(self, data: str) -> None:
        if data:
            self._chunks.append(data)

    def text(self) -> str:
        text = " ".join(self._chunks)
        text = re.sub(r"\s+", " ", text)
        return text.strip()


def _fetch_report(url: str, timeout: float = 20.0) -> Tuple[str, str]:
    if not url or not url.lower().startswith("http"):
        return "", ""

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            html = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return "", ""

    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else ""

    parser = _TextExtractor()
    parser.feed(html)
    body = parser.text()

    if title and body.startswith(title):
        body = body[len(title) :].strip()

    return title, body


def _load_active_bugs(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield {key.strip(): (value or "").strip() for key, value in row.items()}


def _read_patch(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Defects4J Chart bug reports and diffs.")
    parser.add_argument(
        "--defects4j-root",
        required=True,
        help="Path to local defects4j checkout (contains framework/projects).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output raw JSONL file.",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Skip downloading bug reports; only include URLs.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = Path(args.defects4j_root)
    project_dir = root / "framework" / "projects" / "Chart"
    active_bugs = project_dir / "active-bugs.csv"
    patches_dir = project_dir / "patches"

    if not active_bugs.exists():
        raise FileNotFoundError(f"Missing active-bugs.csv at {active_bugs}")
    if not patches_dir.exists():
        raise FileNotFoundError(f"Missing patches directory at {patches_dir}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    skipped_missing_patch = 0
    fetched = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for row in _load_active_bugs(active_bugs):
            bug_id = row.get("bug.id", "")
            patch_path = patches_dir / f"{bug_id}.src.patch"
            if not bug_id or not patch_path.exists():
                skipped_missing_patch += 1
                continue

            diff_text = _read_patch(patch_path)
            report_url = row.get("report.url", "")
            if args.no_fetch:
                report_title, report_body = "", ""
            else:
                report_title, report_body = _fetch_report(report_url)
                if report_title or report_body:
                    fetched += 1

            payload = {
                "project_id": "Chart",
                "bug_id": bug_id,
                "report_id": row.get("report.id", ""),
                "report_url": report_url,
                "revision_id_buggy": row.get("revision.id.buggy", ""),
                "revision_id_fixed": row.get("revision.id.fixed", ""),
                "report_title": report_title,
                "report_body": report_body,
                "diff": diff_text,
                "explanation": "",
                "hallucination_label": None,
            }
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
            total += 1

    print(f"Wrote {total} Chart records to {output_path}")
    if skipped_missing_patch:
        print(f"Skipped {skipped_missing_patch} rows with missing patches")
    if not args.no_fetch:
        print(f"Fetched reports for {fetched} bugs")


if __name__ == "__main__":
    main()

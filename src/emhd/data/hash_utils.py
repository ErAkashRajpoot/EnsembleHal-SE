from __future__ import annotations

import hashlib
import json
from typing import Dict

from .records import NormalizedRecord


def _canonicalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def task_hash(record: NormalizedRecord) -> str:
    payload: Dict[str, str] = {
        "artefact_type": record.artefact_type,
        "prompt": _canonicalize_text(record.prompt),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

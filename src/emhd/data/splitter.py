from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from sklearn.model_selection import StratifiedShuffleSplit

from .records import NormalizedRecord


def stratified_split(
    records: Sequence[NormalizedRecord],
    seed: int,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> Dict[str, List[int]]:
    if abs(train_size + val_size + test_size - 1.0) > 1e-6:
        raise ValueError("Split sizes must sum to 1.0")

    labels = [f"{r.artefact_type}::{r.hallucination_label}" for r in records]
    if len(set(labels)) < 2:
        raise ValueError("Need at least two strata for stratified split")

    sss_train = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    train_idx, temp_idx = next(sss_train.split(list(range(len(records))), labels))

    temp_labels = [labels[i] for i in temp_idx]
    val_ratio = val_size / (val_size + test_size)
    sss_val = StratifiedShuffleSplit(n_splits=1, train_size=val_ratio, random_state=seed)
    val_idx_rel, test_idx_rel = next(sss_val.split(list(range(len(temp_idx))), temp_labels))

    val_idx = [temp_idx[i] for i in val_idx_rel]
    test_idx = [temp_idx[i] for i in test_idx_rel]

    return {
        "train": sorted(train_idx.tolist()),
        "val": sorted(val_idx),
        "test": sorted(test_idx),
    }

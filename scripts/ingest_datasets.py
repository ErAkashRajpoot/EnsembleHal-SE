from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from emhd.config import load_datasets_config, load_project_config
from emhd.data import build_index


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and index datasets.")
    parser.add_argument("--datasets", required=True, help="Path to datasets.yaml")
    parser.add_argument("--project", required=True, help="Path to project.yaml")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip datasets whose files are missing",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    datasets = load_datasets_config(Path(args.datasets))
    project = load_project_config(Path(args.project))

    dataset_specs: List[Tuple[str, Path, str, bool]] = []
    missing: List[str] = []

    for dataset in datasets:
        if not dataset.path.exists():
            missing.append(dataset.name)
            if args.allow_missing:
                continue
            raise FileNotFoundError(f"Dataset file not found: {dataset.path}")

        dataset_specs.append(
            (dataset.name, dataset.path, dataset.artefact_type, dataset.contamination_flag)
        )

    if missing:
        print("Missing datasets:", ", ".join(missing))

    result = build_index(
        datasets=dataset_specs,
        seed=project.random_seed,
        output_dir=project.index_output_dir,
    )

    print("Index build complete:")
    print(result)


if __name__ == "__main__":
    main()

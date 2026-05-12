from __future__ import annotations

import argparse
from pathlib import Path

from emhd.config import load_datasets_config, load_project_config
from emhd.data import build_index


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dataset index and splits.")
    parser.add_argument("--datasets", required=True, help="Path to datasets.yaml")
    parser.add_argument("--project", required=True, help="Path to project.yaml")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    datasets = load_datasets_config(Path(args.datasets))
    project = load_project_config(Path(args.project))

    dataset_specs = [
        (d.name, d.path, d.artefact_type, d.contamination_flag)
        for d in datasets
    ]

    result = build_index(
        datasets=dataset_specs,
        seed=project.random_seed,
        output_dir=project.index_output_dir,
    )

    print("Index build complete:")
    print(result)


if __name__ == "__main__":
    main()

# Normalized Dataset Format

Each dataset must be normalized into a JSONL file with one record per line.

## Required Fields

- task_id (string)
- artefact_type (string): code | api | test | doc | review | explanation
- prompt (string): task specification
- reference (string): ground-truth or target output
- hallucination_label (int): 0-5
- dataset_source (string)
- contamination_flag (bool)

## Optional Fields

- metadata (object): any extra dataset-specific fields

## Example

{"task_id":"cm_0001","artefact_type":"code","prompt":"Write a function...","reference":"def foo():...","hallucination_label":2,"dataset_source":"codemirage","contamination_flag":false,"metadata":{"split":"train"}}

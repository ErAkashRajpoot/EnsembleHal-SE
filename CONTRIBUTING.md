# Contributing to EnsembleHal-SE

Thank you for your interest in contributing to **EMHD: Ensemble Meta-Learning for Hallucination Detection in LLM-Generated Software Artefacts**!

We welcome contributions from researchers, engineers, and open-source enthusiasts. Whether you are extending the framework to new LLMs, adding custom feature extractors, or simply fixing typos, your help is appreciated.

## How to Contribute

1. **Fork the Repository**: Create your own fork of the `ErAkashRajpoot/EnsembleHal-SE` repository.
2. **Clone Locally**: Clone your fork to your local machine.
3. **Set Up the Environment**: Follow the Setup instructions in the `README.md` to ensure your local dependencies match the `pyproject.toml`.
4. **Create a Branch**: Always create a descriptive branch name (e.g., `feature/add-llama4-support` or `fix/xgboost-hyperparameters`).
5. **Make Changes**: Develop your feature or bug fix. Please adhere to the existing code style.
6. **Test Your Changes**: Ensure your changes do not break the 3-layer pipeline:
   - Layer 1: Generation (`generate.py` / `offline_client.py`)
   - Layer 2: Feature Extraction (`extract_features.py`)
   - Layer 3: Meta-Learner Training (`run_evaluation.py`)
7. **Submit a Pull Request**: Push your branch to your fork and submit a PR against our `main` branch. Provide a detailed description of your changes, what motivated them, and any evaluation metrics if applicable.

## Issues and Feature Requests

If you encounter any bugs, rate-limit edge cases with API providers, or have an idea for a new feature, please open an issue in the GitHub issue tracker. Include as much detail as possible, such as error logs, OS version, and exact API configurations used in `.env`.

Thank you for helping make LLM code generation safer and more reliable!

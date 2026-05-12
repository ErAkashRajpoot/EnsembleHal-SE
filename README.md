<div align="center">

# Ensemble Meta-Learning for Hallucination Detection in LLM-Generated Software Artefacts

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A black-box ensemble meta-learning architecture (EMHD) that detects hallucinations in LLM-generated software artefacts by measuring cross-model disagreement across semantic, syntactic, and structural feature spaces.

[**Read the Paper**](#citation) | [**Methodology**](#architecture) | [**Dataset**](#dataset)

</div>

## Overview

Large language models (LLMs) are increasingly deployed to generate software engineering artefacts—source code, API invocations, and unit tests—yet they hallucinate with frequencies that practical deployments cannot tolerate. Existing hallucination detection methods rely on single-model sampling consistency or require narrow execution environments.

**EMHD** introduces a novel three-layer framework that:
1. Queries a heterogeneous ensemble of $K$ black-box LLMs for each SE task.
2. Extracts a **43-dimensional cross-model disagreement feature vector** spanning syntactic, semantic, structural, and confidence dimensions.
3. Feeds this vector to an XGBoost meta-classifier trained with Optuna-optimised hyperparameters.

**Key Results:** Our evaluation on the CodeHalu benchmark demonstrates that EMHD achieves a mean **F1 of 0.900** and **AUROC of 0.971** under 5-fold cross-validation, significantly outperforming single-model consistency methods without requiring logit access or sandbox execution environments.

---

## Architecture

EMHD operates entirely black-box through three distinct layers:

### Layer 1: Multi-LLM Generation
Queries a heterogeneous ensemble of LLMs (e.g., GPT-4o, Claude 3.5, Llama 3) via standard API endpoints to generate candidate artefacts for a given natural-language prompt.

### Layer 2: Feature Extraction (43-Dimensions)
Computes a robust cross-model disagreement vector across four sub-spaces:
- **L1 (Cross-Model Disagreement):** 21 features including Token Jaccard variance, Edit Distance, ROUGE-L, BERTScore, and AST structure variance. *(Contributes the strongest individual signal: F1 = 0.929)*
- **L2 (Confidence Signals):** 8 features including verbalized uncertainty and token entropy (if logprobs are available).
- **L3 (Verification Signals):** 6 features assessing cross-encoder NLI (Entailment/Contradiction) between prompt and generated code comments.
- **L4 (Context Features):** 8 features detailing prompt length, test difficulty, and API rarity.

### Layer 3: Meta-Learning Aggregation
Uses an Optuna-optimised XGBoost stacked generalisation meta-classifier to aggregate the 43-dimensional feature space into a final hallucination probability score.

---

## Setup & Installation

Ensure you have Python 3.10+ installed.

```bash
# 1. Clone the repository
git clone https://github.com/ErAkashRajpoot/EnsembleHal-SE.git
cd EnsembleHal-SE

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the package locally
pip install -e .
```

### API Keys Configuration
Copy `.env.example` to `.env` and fill in your API keys for the ensemble LLMs. 
```bash
cp .env.example .env
```
Ensure you have active API keys for OpenAI, Anthropic, Groq, and OpenRouter to utilize the full heterogeneous ensemble.

---

## Usage

### 1. Generate Artefacts (Layer 1)
To generate the candidate responses from the LLM ensemble:
```bash
python scripts/generate.py --datasets configs/datasets.yaml
```
*Note: For free-tier rate limits, you can use the offline fallback generation script.*

### 2. Extract Features (Layer 2)
To compute the 43-dimensional disagreement matrix from the raw generated responses:
```bash
python scripts/extract_features.py --config configs/models.yaml
```
This produces `features.csv` inside the `data/features/` directory.

### 3. Train Meta-Learner (Layer 3)
To train the XGBoost meta-classifier and evaluate performance via 5-fold cross validation:
```bash
python scripts/run_evaluation.py
```

### 4. Reproduce Paper Figures
To generate the publication-ready PDF figures (ROC Curve, Feature Correlation Heatmap, Layer Ablation Bar Chart) found in the paper:
```bash
python scripts/generate_paper_figures.py
```
Generated figures will be saved in `Research PaperDATA/LatexTemplate/Springer Conference templates/Figures/`.

---

## Dataset

This framework was evaluated on the **CodeHalu** benchmark. The benchmark dataset consists of deterministic execution-based verification for execution hallucinations (mapping, naming, resource, and logic errors). Ensure datasets are placed in the `data/` directory and normalized via the scripts provided in `scripts/normalize_*.py`.

*(Note: Large raw datasets are excluded from this repository via `.gitignore`. You must download the base CodeHalu dataset and process it locally).*

---

## Citation

If you find this code or our research useful, please cite our paper:

```bibtex
@inproceedings{rajpoot2026emhd,
  title={Ensemble Meta-Learning for Hallucination Detection in LLM-Generated Software Artefacts: A Cross-Model Disagreement Approach},
  author={Rajpoot, Akash and Malhotra, Ruchika},
  booktitle={Proceedings of the Springer International Conference on Software Engineering},
  year={2026}
}
```

## License
This project is licensed under the [MIT License](LICENSE).

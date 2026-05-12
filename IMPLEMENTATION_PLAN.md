# Implementation Plan (Revised and Aligned)

Project: Ensemble Meta-Learning for Hallucination Detection in LLM-Generated Software Artefacts
Date: 2026-05-11

This plan is aligned to the support package and the 20-paper list. It fixes cross-cutting gaps, assigns day ranges to every phase, and defines the minimum baselines, datasets, and statistical tests needed for reviewer-grade evidence.

## Day 1: Phase 0 - Foundation and Cross-Cutting Decisions

Deliverables (must complete before any generation work):

1) Model roster (>= 5 diverse LLMs)
- Target diversity across architecture, training data, and RLHF pipelines.
- Example roster (cost-aware, confirm access and pricing):
  - GPT-4o-mini (OpenAI)
  - Gemini 1.5 Flash (Google)
  - Llama-3 70B (open-source, API via Groq/Together)
  - DeepSeek-Coder-V2 (open-source, API via Together)
  - Qwen2.5-Coder-7B (open-source, API via Together)
- Record per-model capabilities:
  - has_logprobs (bool), max_tokens, rate limits, pricing, supported context size.

2) Unified hallucination taxonomy mapping
- Goal: One merged label space across CodeMirage and CodeHalu.
- Unified labels (binary for primary; multi-class for analysis):
  - 0: No hallucination
  - 1: Syntax or invalid structure
  - 2: Semantic or logical error
  - 3: API or resource misuse
  - 4: Specification deviation
  - 5: Robustness or security issue

Suggested mapping table (update once dataset docs are confirmed):

| Source | Category | Unified label |
|---|---|---|
| CodeMirage | Syntactic incorrectness | 1 |
| CodeMirage | Logical error | 2 |
| CodeMirage | Dead code | 4 |
| CodeMirage | Robustness issue | 5 |
| CodeMirage | Security vulnerability | 5 |
| CodeHalu | Mapping | 2 |
| CodeHalu | Naming | 3 |
| CodeHalu | Resource | 3 |
| CodeHalu | Logic | 2 |

3) Experiment registry schema (required fields)
- model_id
- model_provider
- has_logprobs (bool)
- temperature
- max_tokens
- artefact_type
- dataset_source
- task_id
- prompt_version
- hallucination_label
- contamination_flag (bool)
- split (train/val/test)
- generation_mode (greedy or sampled)

4) Contamination policy
- HumanEval and MBPP must be contamination-flagged.
- If contamination_flag == true, use only for feature extraction or ablations, not for final evaluation tables.
- Add inter-dataset overlap check using task description hash.

5) Split strategy and class balance
- Primary split: 70/15/15, stratified by artefact_type AND hallucination_label.
- Report class balance tables per dataset.
- Decide imbalance handling: class-weighted loss (XGBoost scale_pos_weight, LightGBM is_unbalance) and threshold tuning on validation.

6) Compute and cost estimates
- Estimate token usage per phase and per model (assume k = 5 samples per task per model).
- Provide an API cost table before generation begins, using the cost-aware roster.

## Day 2: Phase 1 - Ensemble Diversity Validation

- Pilot with 200 stratified tasks: 50 per artefact type (code, API, test, doc).
- Compute pairwise Q-statistic and Fleiss kappa.
- Report 95% bootstrap CIs and a correlation heatmap.
- Provisional swap rule: Q > 0.7 (flagged), validated later in ablation.

## Day 3: Phase 2 - Dataset Acquisition and Indexing

Primary labeled datasets (must use):
- CodeMirage
- CodeHalu
- CloudAPIBench
- Defects4J (bug explanation artefacts)

Secondary evaluation sources:
- HumanEval, MBPP (contamination-flagged)
- CoderEval, EvoCodeBench
- CodeReviewer or TL-CodeSum for review/explanation/doc tasks

Deliverables:
- Unified dataset index with contamination_flag, overlap hash, and class balance.
- Label policy for execution-based benchmarks:
  - Hallucinated if pass@1 == 0 OR AST-check fails OR identifier check fails.
- Defects4J policy:
  - Sample 300-400 bug fix pairs (diff + developer-written description).
  - Bug explanation is hallucinated if it contradicts the diff.

## Days 4-5: Phase 3 - Prompting and Generation

- Prompt templates per artefact type.
- Generation policy:
  - Core outputs: temperature = 0.7, k = 5 samples per model per task (required for variance features).
  - Greedy output: temperature = 0, 1 sample per model per task for reproducibility (use as one of k or store separately).
  - Capture logprobs when supported for each sample; set has_logprobs accordingly.
- Store outputs with registry metadata.
- Add retry logic (tenacity) and checkpointing.
- Include API cost report at end of phase.

## Day 6: Phase 4 - Pre-labeling and Annotation

- NLI pre-labeling with DeBERTa v3 or a hallucination-specific model.
- Code-specific redundancy checks: CodeBERT similarity + AST identifier overlap.
- Persist NLI scores and AST identifier coverage for Layer 3 features.
- Manual annotation:
  - 200 spot-checks, stratified by artefact type.
  - 100 dual-annotator overlap for Cohen kappa.
  - kappa threshold: >= 0.60 acceptable, >= 0.80 preferred.
  - Conflict resolution: adjudicator or majority.
  - Annotators must have software engineering background.

## Day 7: Phase 5 - Feature Engineering

Layer 1: Disagreement features (>= 16)
- Token Jaccard mean/var, edit distance mean/var, ROUGE-1/2/L mean/var.
- BERTScore P/R/F1 mean/std.
- Cosine distance stats using explicit embedding models:
  - NL outputs: text-embedding-3-large or sentence-transformers equivalent.
  - Code outputs: CodeBERT embeddings.
- Code-specific: AST edit distance variance, identifier Jaccard variance, API call graph Jaccard.

Layer 2: Confidence features (>= 8)
- Logprob entropy mean/var where available.
- Verbalized uncertainty (normalized), disagreement entropy.
- has_logprobs flag.
- Missing data strategy: median imputation OR separate model for no-logprob models (decide and document).

Layer 3: NLI and AST verification features (>= 4)
- NLI entailment score (cross-encoder)
- NLI contradiction score (cross-encoder)
- NLI neutral score (cross-encoder)
- AST identifier coverage rho(o,s) = |ID(o) ∩ AST(s)| / |ID(o)|

Layer 4: Task context features (>= 6)
- artefact_type, prompt length, repo context depth, API rarity, test difficulty, dataset_source.

Deliverables:
- Feature correlation matrix (Spearman) and optional PCA plot.

## Day 8: Phase 6 - Baselines

Required baselines:
- SelfCheckGPT
- MetaQA (prompt mutation MR)
- DrHall (metamorphic testing)
- Functional Clustering
- AST deterministic analysis
- RAG baseline (De-Hallucinator style)

Missing baselines to add:
- Best single model baseline (use its confidence or logprob features)
- Majority voting ensemble
- Random Forest meta-learner
- Logistic Regression meta-learner (sanity check)

## Day 9: Phase 7 - Meta-Learning and Training

- XGBoost and LightGBM with Optuna HPO (TPE, time budget per model).
- Stratified 5-fold CV by artefact_type AND hallucination_label.
- Class imbalance handling enabled.

## Day 10: Phase 8 - Calibration and Statistics

- Isotonic regression on held-out calibration split.
- Report ECE and Brier (pre/post calibration).
- Statistical tests:
  - Bootstrap CIs (10,000 iterations, fixed seed)
  - Wilcoxon signed-rank across folds
  - McNemar with Holm correction
  - DeLong test for AUROC

## Day 11: Phase 9 - Ablations and Reporting

Required ablations:
- K-model sweep (K = 2, 3, 4, 5)
- Q-threshold sensitivity: Q in {0.5, 0.6, 0.7, 0.8}
- Cross-dataset generalization: train CodeMirage, test CodeHalu and vice versa
- Temperature ablation: greedy vs sampled
- Logprob vs disagreement: Layer 2 only vs Layer 1 only vs combined
- AST/NLI contribution: (Layer 1+2) vs (Layer 1+2+3) to quantify deterministic SE gains

Metrics:
- F1, AUROC, AUPRC
- pass@1, pass@3, pass@5 for code
- API validity rate
- Test pass rate

Qualitative analysis:
- Error analysis on 100 samples (50 FP, 50 FN) with taxonomy labels.

## End-to-End Deliverables

- Model roster and diversity report
- Unified taxonomy table
- Dataset index with contamination flags and overlap checks
- Feature extractor library + config
- Baseline implementations
- Trained meta-learners + calibration report
- Results tables with CIs and corrected p-values
- Ablation and error analysis report

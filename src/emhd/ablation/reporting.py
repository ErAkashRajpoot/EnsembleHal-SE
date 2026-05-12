"""Reporting: generate LaTeX tables and Matplotlib figures for the paper."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Guard matplotlib import for headless environments
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    logger.warning("matplotlib not available. Figure generation disabled.")


# ── LaTeX table generators ───────────────────────────────────────────────

def comparison_table_latex(
    results: List[Dict[str, Any]],
    caption: str = "Hallucination detection: EMHD vs baselines.",
    label: str = "tab:comparison",
) -> str:
    """Generate LaTeX table comparing methods across all metrics."""
    header = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\begin{tabular}{lccccc}\n"
        "\\toprule\n"
        "Method & Accuracy & Precision & Recall & F1 & AUROC \\\\\n"
        "\\midrule\n"
    )
    rows = []
    for r in results:
        name = r["method"].replace("_", "\\_")
        acc = f'{r.get("accuracy", 0):.4f}'
        prec = f'{r.get("precision", 0):.4f}'
        rec = f'{r.get("recall", 0):.4f}'
        f1 = f'{r.get("f1", 0):.4f}'
        auroc = f'{r.get("auroc", 0):.4f}' if "auroc" in r else "—"

        # Bold the best F1
        is_best = r.get("f1", 0) == max(x.get("f1", 0) for x in results)
        if is_best:
            f1 = f"\\textbf{{{f1}}}"

        rows.append(f"{name} & {acc} & {prec} & {rec} & {f1} & {auroc} \\\\")

    footer = (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    return header + "\n".join(rows) + "\n" + footer


def calibration_table_latex(
    calibration: Dict[str, Dict[str, float]],
    caption: str = "Calibration results: pre vs post isotonic regression.",
    label: str = "tab:calibration",
) -> str:
    """Generate LaTeX table for calibration metrics."""
    header = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\n"
        "Model & ECE (Pre) & ECE (Post) & Brier (Pre) & Brier (Post) \\\\\n"
        "\\midrule\n"
    )
    rows = []
    for model, cal in calibration.items():
        name = model.replace("_", "\\_")
        rows.append(
            f"{name} & {cal['ece_pre']:.4f} & {cal['ece_post']:.4f} "
            f"& {cal['brier_pre']:.4f} & {cal['brier_post']:.4f} \\\\"
        )
    footer = "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return header + "\n".join(rows) + "\n" + footer


def significance_table_latex(
    mcnemar: Dict[str, Dict[str, float]],
    caption: str = "McNemar test: EMHD-LightGBM vs baselines (Holm-corrected).",
    label: str = "tab:significance",
) -> str:
    """Generate LaTeX table for statistical significance results."""
    header = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\begin{tabular}{lccc}\n"
        "\\toprule\n"
        "Baseline & Statistic & $p$-value & $p$ (Holm) \\\\\n"
        "\\midrule\n"
    )
    rows = []
    for bname, vals in mcnemar.items():
        name = bname.replace("_", "\\_")
        stat = f'{vals.get("statistic", 0):.2f}'
        p_raw = vals.get("p_value", 1.0)
        p_holm = vals.get("p_value_holm", p_raw)
        sig = "*" if p_holm < 0.05 else ""
        rows.append(f"{name} & {stat} & {p_raw:.4f} & {p_holm:.4f}{sig} \\\\")
    footer = "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return header + "\n".join(rows) + "\n" + footer


def ablation_table_latex(
    ablation: List[Dict[str, Any]],
    caption: str = "Feature layer ablation study.",
    label: str = "tab:ablation",
) -> str:
    """Generate LaTeX table for ablation results."""
    header = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\begin{tabular}{lcc}\n"
        "\\toprule\n"
        "Configuration & F1 (mean$\\pm$std) & AUROC (mean$\\pm$std) \\\\\n"
        "\\midrule\n"
    )
    rows = []
    for a in ablation:
        config = a.get("config", "unknown").replace("_", "\\_")
        f1_str = f'{a.get("mean_f1", 0):.4f}$\\pm${a.get("std_f1", 0):.4f}'
        auroc_str = f'{a.get("mean_auroc", 0):.4f}$\\pm${a.get("std_auroc", 0):.4f}'
        rows.append(f"{config} & {f1_str} & {auroc_str} \\\\")
    footer = "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return header + "\n".join(rows) + "\n" + footer


# ── Figure generators ────────────────────────────────────────────────────

def plot_comparison_bar(
    results: List[Dict[str, Any]],
    output_path: Path,
    metric: str = "f1",
    title: str = "EMHD vs Baselines: F1 Score",
) -> None:
    """Bar chart comparing all methods."""
    if not HAS_MPL:
        logger.warning("Skipping bar chart: matplotlib not available.")
        return

    names = [r["method"] for r in results]
    values = [r.get(metric, 0) for r in results]
    colors = ["#2ecc71" if "EMHD" in n else "#3498db" for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names, values, color=colors, edgecolor="white", height=0.6)
    ax.set_xlabel(metric.upper(), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1.05)

    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved comparison figure: %s", output_path)


def plot_ablation_layers(
    ablation: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Grouped bar chart for layer ablation study."""
    if not HAS_MPL:
        logger.warning("Skipping ablation chart: matplotlib not available.")
        return

    configs = [a.get("config", "?") for a in ablation]
    f1_means = [a.get("mean_f1", 0) for a in ablation]
    f1_stds = [a.get("std_f1", 0) for a in ablation]
    auroc_means = [a.get("mean_auroc", 0) for a in ablation]
    auroc_stds = [a.get("std_auroc", 0) for a in ablation]

    x = np.arange(len(configs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, f1_means, width, yerr=f1_stds, label="F1", color="#3498db", capsize=3)
    ax.bar(x + width / 2, auroc_means, width, yerr=auroc_stds, label="AUROC", color="#e74c3c", capsize=3)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Feature Layer Ablation Study", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ablation figure: %s", output_path)


def plot_diversity_heatmap(
    q_matrix: List[List[float]],
    model_ids: List[str],
    output_path: Path,
) -> None:
    """Heatmap for Q-statistic diversity matrix."""
    if not HAS_MPL:
        logger.warning("Skipping heatmap: matplotlib not available.")
        return

    matrix = np.array(q_matrix)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=-1, vmax=1, aspect="auto")

    short_names = [m[:12] for m in model_ids]
    ax.set_xticks(range(len(short_names)))
    ax.set_yticks(range(len(short_names)))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short_names, fontsize=9)

    # Annotate cells
    for i in range(len(model_ids)):
        for j in range(len(model_ids)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(matrix[i, j]) > 0.5 else "black")

    ax.set_title("Pairwise Q-statistic (< 0.7 = sufficient diversity)", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved diversity heatmap: %s", output_path)


# ── All-in-one report generator ──────────────────────────────────────────

def generate_full_report(results_dir: Path, output_dir: Optional[Path] = None) -> None:
    """Load all result JSONs and generate complete LaTeX + figures.

    Expected files in results_dir:
      - comparison_table.json
      - calibration.json
      - significance_tests.json
      - ablations/feature_layer_ablation.json
      - diversity/diversity_pilot.json  (optional)
    """
    if output_dir is None:
        output_dir = results_dir / "report"
    output_dir.mkdir(parents=True, exist_ok=True)

    latex_parts = []

    # 1. Comparison table
    comp_path = results_dir / "comparison_table.json"
    if comp_path.exists():
        with comp_path.open() as f:
            comparison = json.load(f)
        latex_parts.append("% --- Main Comparison Table ---")
        latex_parts.append(comparison_table_latex(comparison))
        plot_comparison_bar(comparison, output_dir / "fig_comparison_f1.png")
        plot_comparison_bar(comparison, output_dir / "fig_comparison_auroc.png",
                           metric="auroc", title="EMHD vs Baselines: AUROC")
    else:
        logger.warning("No comparison_table.json found")

    # 2. Calibration table
    cal_path = results_dir / "calibration.json"
    if cal_path.exists():
        with cal_path.open() as f:
            calibration = json.load(f)
        latex_parts.append("% --- Calibration Table ---")
        latex_parts.append(calibration_table_latex(calibration))

    # 3. Significance table
    sig_path = results_dir / "significance_tests.json"
    if sig_path.exists():
        with sig_path.open() as f:
            significance = json.load(f)
        if "mcnemar" in significance:
            latex_parts.append("% --- Significance Table ---")
            latex_parts.append(significance_table_latex(significance["mcnemar"]))

    # 4. Ablation table + figure
    abl_path = results_dir / "ablations" / "feature_layer_ablation.json"
    if abl_path.exists():
        with abl_path.open() as f:
            ablation = json.load(f)
        latex_parts.append("% --- Ablation Table ---")
        latex_parts.append(ablation_table_latex(ablation))
        plot_ablation_layers(ablation, output_dir / "fig_ablation_layers.png")

    # 5. Diversity heatmap
    div_path = results_dir / "diversity" / "diversity_pilot.json"
    if div_path.exists():
        with div_path.open() as f:
            diversity = json.load(f)
        plot_diversity_heatmap(
            diversity["q_statistic_matrix"],
            diversity["model_ids"],
            output_dir / "fig_diversity_heatmap.png",
        )

    # Write combined LaTeX
    latex_output = output_dir / "tables.tex"
    with latex_output.open("w", encoding="utf-8") as f:
        f.write("% Auto-generated LaTeX tables for EMHD paper\n")
        f.write("% Generated by emhd.ablation.reporting\n\n")
        f.write("\n\n".join(latex_parts))
    logger.info("LaTeX tables written to %s", latex_output)
    logger.info("All figures saved to %s", output_dir)

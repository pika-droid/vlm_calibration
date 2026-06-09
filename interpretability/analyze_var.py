"""Visual Attention Ratio (VAR) diagnostic tool for M3-LLaVA.

Analyzes the attention allocation of generated answer tokens to visual tokens
relative to prompt text tokens in the middle layers (layers 5 to 18).
Uses self-contained numpy implementations for statistical diagnostics.
"""

from __future__ import annotations

import os
import sys
import json
import logging
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Configure matplotlib for headless generation
import matplotlib
matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VLM_VAR_Analysis")


def set_premium_style() -> None:
    """Set custom premium styling for matplotlib plots."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Arial"],
        "axes.edgecolor": "#E2E8F0",
        "axes.linewidth": 1.2,
        "grid.color": "#F1F5F9",
        "grid.linewidth": 0.8,
        "xtick.color": "#64748B",
        "ytick.color": "#64748B",
        "axes.labelcolor": "#1E293B",
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.titlecolor": "#0F172A",
        "figure.dpi": 300,
    })


def numpy_mannwhitneyu(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Computes the Mann-Whitney U statistic (greater alternative) and p-value approximation."""
    n_x = len(x)
    n_y = len(y)
    if n_x == 0 or n_y == 0:
        return 0.0, 1.0
        
    combined = np.concatenate([x, y])
    # Compute ranks with average ranking for ties
    argsorted = np.argsort(combined)
    ranks = np.empty_like(combined, dtype=float)
    ranks[argsorted] = np.arange(1, len(combined) + 1)
    
    # Tie adjustment
    vals, counts = np.unique(combined, return_counts=True)
    if len(vals) < len(combined):
        for val, count in zip(vals, counts):
            if count > 1:
                mask = combined == val
                ranks[mask] = ranks[mask].mean()
                
    r_x = ranks[:n_x].sum()
    u_x = r_x - (n_x * (n_x + 1)) / 2.0
    u_y = n_x * n_y - u_x
    
    # Standard normal approximation for p-value (greater alternative: x has larger values than y)
    mu = n_x * n_y / 2.0
    if len(vals) < len(combined):
        n = n_x + n_y
        var = (n_x * n_y / (n * (n - 1))) * (((n**3 - n) - np.sum(counts**3 - counts)) / 12.0)
    else:
        var = n_x * n_y * (n_x + n_y + 1) / 12.0
        
    z = (u_x - mu) / np.sqrt(var + 1e-8)
    
    # Normal CDF approximation: 1 / (1 + exp(-1.702 * z))
    p_val = 1.0 / (1.0 + np.exp(1.702 * z))
    
    return float(u_x), float(p_val)


def numpy_roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes ROC AUC score using the Mann-Whitney U relationship."""
    x = y_score[y_true == 1]
    y = y_score[y_true == 0]
    n_x = len(x)
    n_y = len(y)
    if n_x == 0 or n_y == 0:
        return 0.5
    u, _ = numpy_mannwhitneyu(x, y)
    return u / (n_x * n_y)


def numpy_roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes False Positive Rate (FPR), True Positive Rate (TPR), and thresholds using numpy."""
    desc_idx = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[desc_idx]
    y_score_sorted = y_score[desc_idx]
    
    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)
    
    total_tps = tps[-1]
    total_fps = fps[-1]
    
    tpr = tps / (total_tps + 1e-8)
    fpr = fps / (total_fps + 1e-8)
    
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    thresholds = np.concatenate([[y_score_sorted[0] + 1.0], y_score_sorted])
    
    return fpr, tpr, thresholds


def evaluate_var_diagnostics(input_dir: str, target_layers: list[int] | None = None) -> None:
    """Load pilot results, analyze Visual Attention Ratio, and generate plots.

    Args:
        input_dir: Directory containing pilot_results.jsonl.
        target_layers: Layer indices to evaluate (typically layers 5 to 18).
    """
    set_premium_style()
    
    input_path = Path(input_dir) / "pilot_results.jsonl"
    plots_dir = Path(input_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.error(f"Pilot results file not found at: {input_path}")
        return

    if target_layers is None:
        target_layers = list(range(5, 19))  # Middle layers 5 to 18

    # 1. Load data and extract VAR metrics
    records = []
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            q_id = data["question_id"]
            stratum = data["stratum"]
            results_by_m = data.get("results_by_m", {})
            
            for m_str, m_data in results_by_m.items():
                if "error" in m_data or "var_by_layer" not in m_data:
                    continue
                    
                var_by_layer = m_data["var_by_layer"]
                
                # Compute average VAR across requested layers
                var_vals = []
                for layer in target_layers:
                    val = var_by_layer.get(str(layer))
                    if val is not None:
                        var_vals.append(val)
                
                if var_vals:
                    mean_var_middle = np.mean(var_vals)
                    
                    record = {
                        "question_id": q_id,
                        "stratum": stratum,
                        "scale": int(m_str),
                        "mean_var": mean_var_middle,
                    }
                    
                    # Add layer-specific columns
                    for layer in target_layers:
                        val = var_by_layer.get(str(layer))
                        if val is not None:
                            record[f"var_layer_{layer}"] = val
                            
                    records.append(record)

    if not records:
        logger.error("No valid VAR records found in results file.")
        return

    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} records containing VAR data.")

    # 2. Statistical Analysis: stable_correct vs stable_incorrect_strict
    correct_df = df[df["stratum"] == "stable_correct"]
    incorrect_df = df[df["stratum"] == "stable_incorrect_strict"]

    if len(correct_df) > 0 and len(incorrect_df) > 0:
        c_var = correct_df["mean_var"].values
        i_var = incorrect_df["mean_var"].values
        
        # Mann-Whitney U Test
        u_stat, p_val = numpy_mannwhitneyu(c_var, i_var)
        
        # Cohen's d effect size
        cohen_d = (np.mean(c_var) - np.mean(i_var)) / np.sqrt((np.var(c_var) + np.var(i_var)) / 2.0 + 1e-8)
        
        logger.info("=== Visual Attention Ratio (VAR) statistical diagnostics ===")
        logger.info(f"Stable Correct mean VAR:            {np.mean(c_var):.4f}")
        logger.info(f"Stable Incorrect (Strict) mean VAR: {np.mean(i_var):.4f}")
        logger.info(f"Mann-Whitney U statistic:           {u_stat:.1f}")
        logger.info(f"p-value (approx):                   {p_val:.4g}")
        logger.info(f"Cohen's d effect size:              {cohen_d:.4f}")

        # Compute ROC curve and AUC using numpy
        eval_df = df[df["stratum"].isin(["stable_correct", "stable_incorrect_strict"])].copy()
        eval_df["label"] = (eval_df["stratum"] == "stable_correct").astype(int)
        
        y_true = eval_df["label"].values
        y_score = eval_df["mean_var"].values
        
        auc = numpy_roc_auc_score(y_true, y_score)
        fpr, tpr, thresholds = numpy_roc_curve(y_true, y_score)
        logger.info(f"ROC-AUC score:                      {auc:.4f}")
        
        # Save ROC plot
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color="#3B82F6", linewidth=2.5, label=f"VAR Discriminator (AUC = {auc:.3f})")
        plt.plot([0, 1], [0, 1], color="#94A3B8", linestyle="--", linewidth=1.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("VAR ROC Curve: Correct vs. Hallucinated Answers")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(plots_dir / "var_roc_curve.png")
        plt.close()
        logger.info(f"ROC curve saved to: {plots_dir / 'var_roc_curve.png'}")
    else:
        logger.warning("Insufficient classes found in pilot data for statistical classification testing.")

    # ── 1. Plot Violin Distribution of VAR by Stratum ──────────────────────
    plt.figure(figsize=(10, 6))
    strata_colors = {
        "stable_correct": "#10B981",
        "stable_incorrect_strict": "#EF4444",
        "stable_incorrect_relaxed": "#F59E0B",
        "flip": "#64748B"
    }
    
    sns.violinplot(
        data=df,
        x="stratum",
        y="mean_var",
        palette=strata_colors,
        order=["stable_correct", "stable_incorrect_strict", "stable_incorrect_relaxed", "flip"],
        cut=0
    )
    plt.title("Visual Attention Ratio (VAR) across VQA Strata")
    plt.xlabel("Prediction Stratum")
    plt.ylabel("Mean VAR (Layers 5-18)")
    plt.xticks(
        ticks=[0, 1, 2, 3],
        labels=["Stable Correct", "Stable Incorrect\n(Strict)", "Stable Incorrect\n(Relaxed)", "Flip"]
    )
    plt.tight_layout()
    plt.savefig(plots_dir / "var_distribution.png")
    plt.close()
    logger.info(f"VAR distribution violin plot saved to: {plots_dir / 'var_distribution.png'}")

    # ── 2. Plot VAR by Layer Trend Line ────────────────────────────────────
    plt.figure(figsize=(10, 6))
    
    # Restructure data to long format for lineplot
    layer_cols = [f"var_layer_{l}" for l in target_layers]
    long_records = []
    
    for _, row in df.iterrows():
        for l in target_layers:
            col = f"var_layer_{l}"
            if col in row and not pd.isna(row[col]):
                long_records.append({
                    "stratum": row["stratum"],
                    "layer": l,
                    "var_val": row[col]
                })
                
    long_df = pd.DataFrame(long_records)
    
    sns.lineplot(
        data=long_df,
        x="layer",
        y="var_val",
        hue="stratum",
        palette=strata_colors,
        hue_order=["stable_correct", "stable_incorrect_strict", "stable_incorrect_relaxed", "flip"],
        linewidth=2.5,
        marker="o"
    )
    plt.title("Visual Attention Ratio (VAR) Trajectory across Decoder Layers")
    plt.xlabel("Decoder Layer Index")
    plt.ylabel("Visual Attention Ratio (VAR)")
    plt.xticks(target_layers)
    plt.legend(
        title="Stratum",
        labels=["Stable Correct", "Stable Incorrect (Strict)", "Stable Incorrect (Relaxed)", "Flip"]
    )
    plt.tight_layout()
    plt.savefig(plots_dir / "var_by_layer.png")
    plt.close()
    logger.info(f"VAR layer trend line saved to: {plots_dir / 'var_by_layer.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Visual Attention Ratio diagnostics on pilot results.")
    parser.add_argument("--input-dir", type=str, default="results/pilot-interpretability")
    args = parser.parse_args()
    
    evaluate_var_diagnostics(args.input_dir)

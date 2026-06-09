"""Intermediate layer softmax calibration (LatentLens) diagnostic tool.

Analyzes intermediate representation vocabulary projections (LogitLens) and
computes Expected Calibration Error (ECE) across hooked decoder layers.
Performs post-hoc temperature scaling using binary log-odds calibration and
numpy-based optimization (removing scipy dependencies).
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
logger = logging.getLogger("VLM_Interpretability.latent_lens")


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


def compute_ece(confidences: np.ndarray, accuracies: np.ndarray, num_bins: int = 15) -> float:
    """Computes Expected Calibration Error (ECE).

    Args:
        confidences: Array of predicted top-1 confidence values [0, 1].
        accuracies: Binary array of correctness (1 for correct, 0 for incorrect).
        num_bins: Number of bins to partition the confidence space.

    Returns:
        ECE as a float.
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    total_samples = len(confidences)

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Identify samples falling in the current bin
        in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            confidence_in_bin = np.mean(confidences[in_bin])
            ece += prop_in_bin * np.abs(accuracy_in_bin - confidence_in_bin)

    return float(ece)


def binary_temperature_scale(confidences: np.ndarray, temp: float) -> np.ndarray:
    """Applies temperature scaling using a binary log-odds reconstruction.

    Reconstructs the log-odds (binary logits) of the top-1 prediction, divides
    them by the temperature parameter, and runs them back through sigmoid.
    """
    eps = 1e-7
    clamped_conf = np.clip(confidences, eps, 1.0 - eps)
    
    # Log-odds logit: z = log(p / (1 - p))
    logits = np.log(clamped_conf / (1.0 - clamped_conf))
    
    # Scale and apply sigmoid
    scaled_conf = 1.0 / (1.0 + np.exp(-logits / temp))
    return scaled_conf


def optimize_temperature(confidences: np.ndarray, accuracies: np.ndarray) -> float:
    """Finds the optimal temperature parameter minimizing Negative Log-Likelihood (NLL).

    Uses a robust grid search over T ∈ [0.05, 5.0] with step 0.025 to remove scipy dependency.
    """
    eps = 1e-7
    t_candidates = np.linspace(0.05, 5.00, 199)
    best_t = 1.0
    best_loss = float("inf")
    
    for t in t_candidates:
        scaled_c = binary_temperature_scale(confidences, t)
        loss = -np.mean(accuracies * np.log(scaled_c + eps) + (1.0 - accuracies) * np.log(1.0 - scaled_c + eps))
        if loss < best_loss:
            best_loss = loss
            best_t = t
            
    return float(best_t)


def evaluate_calibration(input_dir: str) -> None:
    """Load results, compute ECE across layers, run temperature scaling, and generate plots."""
    set_premium_style()
    
    input_path = Path(input_dir) / "pilot_results.jsonl"
    plots_dir = Path(input_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.error(f"Pilot results file not found at: {input_path}")
        return

    # 1. Load data
    records = []
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            q_id = data["question_id"]
            stratum = data["stratum"]
            gt_answers = [ans.strip().lower() for ans in data.get("gt_answers", [])]
            results_by_m = data.get("results_by_m", {})
            
            for m_str, m_data in results_by_m.items():
                if "error" in m_data or "logit_lens_by_layer" not in m_data:
                    continue
                    
                # H5: Use vqa_accuracy >= 0.5 for correctness if available, else fallback to in check
                if "vqa_accuracy" in m_data:
                    is_correct = int(m_data["vqa_accuracy"] >= 0.5)
                else:
                    final_ans = m_data.get("answer", "").strip().lower()
                    is_correct = int(final_ans in gt_answers)
                
                # Check layers
                logit_lens = m_data["logit_lens_by_layer"]
                for layer_str, lens_data in logit_lens.items():
                    records.append({
                        "question_id": q_id,
                        "stratum": stratum,
                        "scale": int(m_str),
                        "layer": int(layer_str),
                        "final_correct": is_correct,
                        "lens_top1_token": lens_data["top1_token"].strip().lower(),
                        "confidence": lens_data["confidence"],
                        "entropy": lens_data["entropy"]
                    })

    if not records:
        logger.error("No valid LogitLens records found.")
        return

    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} records containing calibration metrics.")

    # Unique hooked layers
    layers = sorted(df["layer"].unique())
    logger.info(f"Found hooked layers: {layers}")

    ece_results = []
    
    # 2. Compute ECE and optimize temperature for each layer
    for layer in layers:
        layer_df = df[df["layer"] == layer]
        conf = layer_df["confidence"].values
        acc = layer_df["final_correct"].values
        
        # Raw ECE
        raw_ece = compute_ece(conf, acc)
        
        # Optimize temperature
        opt_temp = optimize_temperature(conf, acc)
        
        # Scaled ECE
        scaled_conf = binary_temperature_scale(conf, opt_temp)
        scaled_ece = compute_ece(scaled_conf, acc)
        
        logger.info(
            f"Layer {layer:<2} | Raw ECE: {raw_ece*100:5.2f}% | "
            f"Opt Temp: {opt_temp:4.2f} | Scaled ECE: {scaled_ece*100:5.2f}%"
        )
        
        ece_results.append({
            "layer": layer,
            "raw_ece": raw_ece,
            "optimal_temp": opt_temp,
            "scaled_ece": scaled_ece
        })

    ece_df = pd.DataFrame(ece_results)

    # ── 1. Plot ECE Comparison Bar Chart (Before vs After Scaling) ────────
    plt.figure(figsize=(10, 6))
    x = np.arange(len(layers))
    width = 0.35
    
    plt.bar(x - width/2, ece_df["raw_ece"] * 100, width, label="Raw ECE", color="#3B82F6", alpha=0.85)
    plt.bar(x + width/2, ece_df["scaled_ece"] * 100, width, label="Temperature Scaled ECE", color="#10B981", alpha=0.85)
    
    plt.title("Expected Calibration Error (ECE) across Decoder Layers")
    plt.xlabel("Decoder Layer Index")
    plt.ylabel("ECE (%)")
    plt.xticks(x, layers)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "intermediate_ece_curve.png")
    plt.close()
    logger.info(f"ECE comparison bar chart saved to: {plots_dir / 'intermediate_ece_curve.png'}")

    # ── 2. Plot Confidence Evolution Trajectory by Stratum ──────────────────
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x="layer",
        y="confidence",
        hue="stratum",
        palette={
            "stable_correct": "#10B981",
            "stable_incorrect_strict": "#EF4444",
            "stable_incorrect_relaxed": "#F59E0B",
            "flip": "#64748B"
        },
        linewidth=2.5,
        marker="o"
    )
    plt.title("LogitLens Confidence Trajectory across Decoder Layers")
    plt.xlabel("Decoder Layer Index")
    plt.ylabel("Intermediate Softmax Confidence")
    plt.xticks(layers)
    plt.legend(
        title="Stratum",
        labels=["Stable Correct", "Stable Incorrect (Strict)", "Stable Incorrect (Relaxed)", "Flip"]
    )
    plt.tight_layout()
    plt.savefig(plots_dir / "layer_confidence_evolution.png")
    plt.close()
    logger.info(f"Confidence evolution trajectory saved to: {plots_dir / 'layer_confidence_evolution.png'}")

    # ── 3. Plot Temperature Parameter vs Layer ─────────────────────────────
    plt.figure(figsize=(10, 6))
    plt.plot(layers, ece_df["optimal_temp"], color="#8B5CF6", marker="s", linewidth=2.5, label="Optimal Temp (T)")
    plt.axhline(1.0, color="#64748B", linestyle="--", alpha=0.7)
    plt.title("Optimal Post-Hoc Temperature (T) by Decoder Layer")
    plt.xlabel("Decoder Layer Index")
    plt.ylabel("Temperature Parameter (T)")
    plt.xticks(layers)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "temperature_scaling_comparison.png")
    plt.close()
    logger.info(f"Temperature scaling parameter plot saved to: {plots_dir / 'temperature_scaling_comparison.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LogitLens calibration and temperature scaling.")
    parser.add_argument("--input-dir", type=str, default="results/pilot-interpretability")
    args = parser.parse_args()
    
    evaluate_calibration(args.input_dir)

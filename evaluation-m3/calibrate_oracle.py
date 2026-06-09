"""
Oracle-Weighted Calibration Analysis for M3-LLaVA sweeps.

Processes raw token-sweep results to simulate a sequential oracle-updating scheme:
- Succeeding token count inferences have access to previous answers, confidences,
  and actual accuracies.
- Compares Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
  before and after calibration.
- Generates comparative plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Setup premium styling or fallback
try:
    from evaluation.config import DEFAULT_CONFIG, EvalConfig
except ImportError:
    # Handle if run directly or paths differ
    class DEFAULT_CONFIG:
        figure_dpi = 150

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
        "figure.dpi": DEFAULT_CONFIG.figure_dpi,
    })

def normalize_vqa_answer(text: str) -> str:
    """Normalize VQA answers for string matching (similar to VQA eval rules)."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\b(a|an|the)\b', '', text)
    return ' '.join(text.split())

def compute_calibration_metrics(
    accuracies: list[float], confidences: list[float], num_bins: int = 15
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Compute bin-level calibration statistics, ECE, and MCE."""
    accuracies = np.array(accuracies)
    confidences = np.array(confidences)
    
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_accuracies = np.zeros(num_bins)
    bin_confidences = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)
    
    ece = 0.0
    mce = 0.0
    
    for i in range(num_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]
        
        if i == num_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            
        bin_counts[i] = np.sum(in_bin)
        
        if bin_counts[i] > 0:
            bin_accuracies[i] = np.mean(accuracies[in_bin])
            bin_confidences[i] = np.mean(confidences[in_bin])
            
            bin_weight = bin_counts[i] / len(confidences)
            bin_diff = np.abs(bin_accuracies[i] - bin_confidences[i])
            ece += bin_weight * bin_diff
            
            mce = max(mce, bin_diff)
            
    return bin_edges, bin_accuracies, bin_confidences, ece, mce

def main():
    parser = argparse.ArgumentParser(description="Oracle-weighted calibration for M3 sweeps.")
    parser.add_argument("--input-file", type=str, default=None, help="Path to raw results JSONL.")
    parser.add_argument("--output-dir", type=str, default=None, help="Path to write calibrated results.")
    parser.add_argument("--beta", type=float, default=1.0, help="Smoothing parameter (default: 1.0 = pure oracle).")
    parser.add_argument("--num-bins", type=int, default=15, help="Number of bins for ECE/reliability diagrams.")
    parser.add_argument("--penalize-new", action="store_true", help="Penalize new answers to 0 if a previous answer was correct.")
    args = parser.parse_args()
    
    # 1. Resolve paths
    project_root = Path(__file__).resolve().parent.parent.parent
    
    input_file_path = args.input_file
    if not input_file_path:
        input_file_path = project_root / "results" / "vlm-calibration-m3" / "results" / "multi_scale_results.jsonl"
    else:
        input_file_path = Path(input_file_path)
        
    output_dir_path = args.output_dir
    if not output_dir_path:
        output_dir_path = project_root / "results" / "m3-weighted-confidence"
    else:
        output_dir_path = Path(output_dir_path)
        
    output_results_dir = output_dir_path / "results"
    output_plots_dir = output_dir_path / "plots"
    
    output_results_dir.mkdir(parents=True, exist_ok=True)
    output_plots_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading raw results from: {input_file_path}")
    if not input_file_path.exists():
        print(f"Error: Raw results file not found at {input_file_path}!")
        return

    # 2. Process results and apply oracle calibration
    calibrated_entries = []
    token_sweep = [1, 9, 36, 144, 576]
    
    print(f"Applying oracle calibration (beta={args.beta}, penalize_new={args.penalize_new})...")
    with open(input_file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            
            # Apply oracle weighting logic
            history = []  # list of {'m': m, 'answer': normalized_ans, 'accuracy': acc}
            
            for m in token_sweep:
                m_str = str(m)
                m_data = entry["results_by_m"][m_str]
                
                raw_conf = m_data["avg_softmax_conf"]
                raw_ans = m_data["answer"]
                acc = m_data["vqa_accuracy"]
                
                normalized_ans = normalize_vqa_answer(raw_ans)
                
                if m == 1:
                    # First step has no history, uses raw confidence
                    oracle_weighted_conf = raw_conf
                else:
                    # Look up if answer has matches in history
                    matches = [h for h in history if h["answer"] == normalized_ans]
                    
                    if matches:
                        # Matches a previous answer. Use that answer's accuracy
                        prev_acc = matches[-1]["accuracy"]
                        oracle_weighted_conf = args.beta * prev_acc + (1.0 - args.beta) * raw_conf
                    else:
                        # New answer.
                        if args.penalize_new:
                            # Check if any previous answer was fully correct (1.0)
                            any_prev_correct = any(h["accuracy"] >= 1.0 for h in history)
                            if any_prev_correct:
                                # A previous answer was correct and we shifted to a new one, so the new one is wrong
                                oracle_weighted_conf = (1.0 - args.beta) * raw_conf
                            else:
                                oracle_weighted_conf = raw_conf
                        else:
                            # Simplified: keep raw confidence for new answers
                            oracle_weighted_conf = raw_conf
                
                # Clamp confidence between 0 and 1
                oracle_weighted_conf = float(np.clip(oracle_weighted_conf, 0.0, 1.0))
                
                # Save the new confidence score in the result dictionary
                m_data["oracle_weighted_conf"] = oracle_weighted_conf
                
                # Record to history
                # Note: VQA accuracy might be None if the dataset is unlabeled (e.g. some splits)
                # If None, treat accuracy as 0.0 for oracle simulation logic
                history.append({
                    "m": m,
                    "answer": normalized_ans,
                    "accuracy": acc if acc is not None else 0.0
                })
            
            # Compute new aggregated statistics
            weighted_confs = [entry["results_by_m"][str(m)]["oracle_weighted_conf"] for m in token_sweep]
            entry["statistics"]["oracle_weighted_confidence_mean"] = float(np.mean(weighted_confs))
            entry["statistics"]["oracle_weighted_confidence_var"] = float(np.var(weighted_confs))
            
            calibrated_entries.append(entry)

    # 3. Write outputs
    output_jsonl_path = output_results_dir / "multi_scale_results.jsonl"
    output_csv_path = output_results_dir / "summary_statistics.csv"
    
    print(f"Writing calibrated JSONL to: {output_jsonl_path}")
    with open(output_jsonl_path, "w", encoding="utf-8") as f_out:
        for entry in calibrated_entries:
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"Writing summary CSV to: {output_csv_path}")
    headers = [
        "question_id", "image_id", "question_type", "answer_type",
        "answer_stability", "mean_similarity_to_final",
        "confidence_mean_logprob", "confidence_var_logprob",
        "confidence_mean_softmax", "confidence_var_softmax",
        "oracle_weighted_confidence_mean", "oracle_weighted_confidence_var",
        "accuracy_mean", "num_unique_answers"
    ]
    for m in token_sweep:
        headers.extend([
            f"acc_m_{m}",
            f"conf_logprob_m_{m}",
            f"conf_softmax_m_{m}",
            f"conf_oracle_weighted_m_{m}"
        ])
        
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(headers)
        
        for entry in calibrated_entries:
            stats = entry["statistics"]
            row = [
                entry["question_id"],
                entry["image_id"],
                entry["question_type"],
                entry["answer_type"],
                stats["answer_stability"],
                stats["mean_similarity_to_final"],
                stats["confidence_mean_logprob"],
                stats["confidence_var_logprob"],
                stats["confidence_mean_softmax"],
                stats["confidence_var_softmax"],
                stats["oracle_weighted_confidence_mean"],
                stats["oracle_weighted_confidence_var"],
                stats["accuracy_mean"],
                stats["num_unique_answers"]
            ]
            for m in token_sweep:
                m_data = entry["results_by_m"][str(m)]
                row.extend([
                    m_data["vqa_accuracy"],
                    m_data["avg_log_prob"],
                    m_data["avg_softmax_conf"],
                    m_data["oracle_weighted_conf"]
                ])
            writer.writerow(row)

    # 4. Calibration Analysis and Metrics
    raw_ece_list = []
    raw_mce_list = []
    weighted_ece_list = []
    weighted_mce_list = []
    
    print("\n=== ECE/MCE Metrics Comparison ===")
    print(f"{'Token Count (m)':<15} | {'Raw ECE (%)':<12} | {'Oracle ECE (%)':<15} | {'Raw MCE (%)':<12} | {'Oracle MCE (%)':<15}")
    print("-" * 85)
    
    for m in token_sweep:
        m_str = str(m)
        valid_pairs = [
            (
                r["results_by_m"][m_str]["vqa_accuracy"],
                r["results_by_m"][m_str]["avg_softmax_conf"],
                r["results_by_m"][m_str]["oracle_weighted_conf"]
            )
            for r in calibrated_entries
            if r["results_by_m"][m_str]["vqa_accuracy"] is not None
        ]
        
        if not valid_pairs:
            print(f"{m:<15} | No labeled samples.")
            raw_ece_list.append(0.0)
            raw_mce_list.append(0.0)
            weighted_ece_list.append(0.0)
            weighted_mce_list.append(0.0)
            continue
            
        accs, raw_confs, weighted_confs = zip(*valid_pairs)
        
        _, _, _, raw_ece, raw_mce = compute_calibration_metrics(accs, raw_confs, num_bins=args.num_bins)
        _, _, _, weighted_ece, weighted_mce = compute_calibration_metrics(accs, weighted_confs, num_bins=args.num_bins)
        
        raw_ece_list.append(raw_ece)
        raw_mce_list.append(raw_mce)
        weighted_ece_list.append(weighted_ece)
        weighted_mce_list.append(weighted_mce)
        
        print(f"{m:<15} | {raw_ece*100:<12.4f} | {weighted_ece*100:<15.4f} | {raw_mce*100:<12.4f} | {weighted_mce*100:<15.4f}")
    
    print("=" * 85)

    # Save Markdown Table
    with open(output_plots_dir / "calibration_comparison_table.md", "w", encoding="utf-8") as f:
        f.write("# Calibration Improvement Metrics (Raw vs. Oracle Weighted)\n\n")
        f.write("| Token Count (m) | Raw ECE (%) | Oracle ECE (%) | Raw MCE (%) | Oracle MCE (%) |\n")
        f.write("|---|---|---|---|---|\n")
        for idx, m in enumerate(token_sweep):
            f.write(f"| {m} | {raw_ece_list[idx]*100:.4f}% | {weighted_ece_list[idx]*100:.4f}% | {raw_mce_list[idx]*100:.4f}% | {weighted_mce_list[idx]*100:.4f}% |\n")
            
    # 5. Plotting Comparisons
    set_premium_style()
    
    # Plot ECE comparison bar chart
    plt.figure(figsize=(10, 6))
    x = np.arange(len(token_sweep))
    width = 0.35
    
    plt.bar(x - width/2, [e * 100 for e in raw_ece_list], width, label="Raw ECE", color="#3B82F6", edgecolor="#2563EB", alpha=0.85)
    plt.bar(x + width/2, [e * 100 for e in weighted_ece_list], width, label="Oracle Weighted ECE", color="#10B981", edgecolor="#059669", alpha=0.85)
    
    # Add values on top of the bars
    for i in range(len(token_sweep)):
        plt.annotate(f"{raw_ece_list[i]*100:.2f}%", xy=(i - width/2, raw_ece_list[i]*100), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#1E293B")
        plt.annotate(f"{weighted_ece_list[i]*100:.2f}%", xy=(i + width/2, weighted_ece_list[i]*100), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#1E293B")
        
    plt.title("ECE Reduction via Oracle Sequential Calibration (M3-LLaVA sweeps)", pad=15)
    plt.xlabel("Visual Token Scale (m)")
    plt.ylabel("Expected Calibration Error (%)")
    plt.xticks(x, [str(m) for m in token_sweep])
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_plots_dir / "ece_comparison.png", dpi=150)
    plt.close()
    
    # Plot side-by-side reliability diagrams
    nrows = len(token_sweep)
    ncols = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4 * nrows), sharex=True, sharey=True)
    
    for idx, m in enumerate(token_sweep):
        m_str = str(m)
        valid_pairs = [
            (
                r["results_by_m"][m_str]["vqa_accuracy"],
                r["results_by_m"][m_str]["avg_softmax_conf"],
                r["results_by_m"][m_str]["oracle_weighted_conf"]
            )
            for r in calibrated_entries
            if r["results_by_m"][m_str]["vqa_accuracy"] is not None
        ]
        
        accs, raw_confs, weighted_confs = zip(*valid_pairs)
        
        # Raw reliability diagram
        ax_raw = axes[idx, 0]
        bin_edges, bin_accs, bin_confs, ece, mce = compute_calibration_metrics(accs, raw_confs, num_bins=args.num_bins)
        
        ax_raw.plot([0, 1], [0, 1], linestyle="--", color="#64748B", linewidth=1.2)
        bin_width = 1.0 / args.num_bins
        ax_raw.bar(bin_edges[:-1], bin_accs, width=bin_width, align="edge", color="#3B82F6", edgecolor="#2563EB", linewidth=0.5, alpha=0.8, label="Accuracy")
        gap = np.clip(bin_confs - bin_accs, 0, None)
        ax_raw.bar(bin_edges[:-1], gap, bottom=bin_accs, width=bin_width, align="edge", color="#EF4444", edgecolor="#DC2626", linewidth=0.5, alpha=0.2, label="Calibration Gap")
        
        ax_raw.set_title(f"m = {m} Tokens (Raw)", fontsize=11, fontweight="bold")
        ax_raw.text(0.05, 0.90, f"ECE: {ece:.2%}\nMCE: {mce:.2%}", transform=ax_raw.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="#F8FAFC", edgecolor="#E2E8F0", alpha=0.9), fontsize=9, verticalalignment="top")
        
        ax_raw.set_xlim(0, 1)
        ax_raw.set_ylim(0, 1)
        if idx == nrows - 1:
            ax_raw.set_xlabel("Raw Softmax Confidence")
        ax_raw.set_ylabel("Empirical Accuracy")
        
        # Oracle reliability diagram
        ax_oracle = axes[idx, 1]
        bin_edges, bin_accs, bin_confs, ece_w, mce_w = compute_calibration_metrics(accs, weighted_confs, num_bins=args.num_bins)
        
        ax_oracle.plot([0, 1], [0, 1], linestyle="--", color="#64748B", linewidth=1.2)
        ax_oracle.bar(bin_edges[:-1], bin_accs, width=bin_width, align="edge", color="#10B981", edgecolor="#059669", linewidth=0.5, alpha=0.8, label="Accuracy")
        gap_w = np.clip(bin_confs - bin_accs, 0, None)
        ax_oracle.bar(bin_edges[:-1], gap_w, bottom=bin_accs, width=bin_width, align="edge", color="#EF4444", edgecolor="#DC2626", linewidth=0.5, alpha=0.2, label="Calibration Gap")
        
        ax_oracle.set_title(f"m = {m} Tokens (Oracle Weighted)", fontsize=11, fontweight="bold")
        ax_oracle.text(0.05, 0.90, f"ECE: {ece_w:.2%}\nMCE: {mce_w:.2%}", transform=ax_oracle.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="#F8FAFC", edgecolor="#E2E8F0", alpha=0.9), fontsize=9, verticalalignment="top")
        
        ax_oracle.set_xlim(0, 1)
        ax_oracle.set_ylim(0, 1)
        if idx == nrows - 1:
            ax_oracle.set_xlabel("Calibrated Oracle Confidence")
            
    plt.suptitle("M3-LLaVA Calibration Optimization: Raw vs. Oracle-Weighted Reliability Diagrams", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(output_plots_dir / "reliability_diagrams_comparison.png", dpi=150)
    plt.close()
    
    print(f"\nPlots generated successfully:")
    print(f"- {output_plots_dir / 'ece_comparison.png'}")
    print(f"- {output_plots_dir / 'reliability_diagrams_comparison.png'}")
    print(f"- {output_plots_dir / 'calibration_comparison_table.md'}")

if __name__ == "__main__":
    main()

"""
Visualization scripts to analyze prediction variance and stability.

Generates:
1. Variance Distribution Histogram (answer stability distribution).
2. Accuracy vs. Token Count Curve (with error bands).
3. Detailed tables and grid figures of the highest and lowest variance questions.
4. Answer transition heatmaps showing how predictions shift across scales.
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from evaluation.config import DEFAULT_CONFIG, EvalConfig


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


def generate_variance_plots(config: EvalConfig) -> None:
    """Read results and generate stability and variance plots."""
    set_premium_style()
    
    jsonl_path = config.results_jsonl_path()
    plots_dir = Path(config.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    if not jsonl_path.exists():
        print(f"Error: Results file not found at {jsonl_path}")
        return
        
    # Load all results
    results = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))
            
    df = pd.DataFrame([
        {
            "question_id": r["question_id"],
            "question": r["question"],
            "question_type": r["question_type"],
            "answer_type": r["answer_type"],
            "answer_stability": r["statistics"]["answer_stability"],
            "mean_similarity_to_final": r["statistics"]["mean_similarity_to_final"],
            "confidence_var_logprob": r["statistics"]["confidence_var_logprob"],
            "confidence_var_softmax": r["statistics"]["confidence_var_softmax"],
            "num_unique_answers": r["statistics"]["num_unique_answers"],
        }
        for r in results
    ])
    
    # ── 1. Variance Distribution Histogram ─────────────────────────────────
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df,
        x="answer_stability",
        kde=True,
        color="#3B82F6",
        edgecolor="#2563EB",
        linewidth=1.2,
        alpha=0.6,
        bins=20
    )
    plt.title("Distribution of VQA Answer Stability across Token Scale Sweeps")
    plt.xlabel("Answer Stability (Mean Pairwise Cosine Similarity)")
    plt.ylabel("Number of VQA Questions")
    plt.tight_layout()
    plt.savefig(plots_dir / "variance_distribution.png", dpi=config.figure_dpi)
    plt.close()
    
    # ── 2. Accuracy vs. Token Count Curve ──────────────────────────────────
    m_values = config.token_sweep
    accuracies = {m: [] for m in m_values}
    confidences = {m: [] for m in m_values}
    
    for r in results:
        for m in m_values:
            m_str = str(m)
            vqa_acc = r["results_by_m"][m_str]["vqa_accuracy"]
            if vqa_acc is not None:
                accuracies[m].append(vqa_acc)
            confidences[m].append(r["results_by_m"][m_str]["avg_softmax_conf"])
            
    # Check if we have any accuracy values (we might be running on a split with no labels)
    has_accuracy = all(len(accuracies[m]) > 0 for m in m_values)
    
    if has_accuracy:
        m_means = [np.mean(accuracies[m]) for m in m_values]
        m_sems = [np.std(accuracies[m]) / np.sqrt(len(accuracies[m])) for m in m_values]
    else:
        m_means = [0.0] * len(m_values)
        m_sems = [0.0] * len(m_values)
        
    m_conf_means = [np.mean(confidences[m]) for m in m_values]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    if has_accuracy:
        # Plot accuracy on left axis
        color = "#10B981"
        ax1.set_xlabel("Visual Token Footprint (m)")
        ax1.set_ylabel("VQA Accuracy", color=color, fontweight="bold")
        line1 = ax1.plot(m_values, m_means, marker="o", color=color, linewidth=2.5, label="VQA Accuracy")
        ax1.fill_between(
            m_values,
            [m - 1.96 * s for m, s in zip(m_means, m_sems)],
            [m + 1.96 * s for m, s in zip(m_means, m_sems)],
            color=color,
            alpha=0.15
        )
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.set_xscale("log")
        ax1.set_xticks(m_values)
        ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        
        # Plot confidence on right axis
        ax2 = ax1.twinx()
        color = "#3B82F6"
        ax2.set_ylabel("Avg Softmax Confidence", color=color, fontweight="bold")
        line2 = ax2.plot(m_values, m_conf_means, marker="s", linestyle="--", color=color, linewidth=2, label="Confidence")
        ax2.tick_params(axis="y", labelcolor=color)
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="lower right")
    else:
        # Only plot confidence
        color = "#3B82F6"
        ax1.set_xlabel("Visual Token Footprint (m)")
        ax1.set_ylabel("Avg Softmax Confidence", color=color, fontweight="bold")
        ax1.plot(m_values, m_conf_means, marker="s", linestyle="-", color=color, linewidth=2.5, label="Confidence")
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.set_xscale("log")
        ax1.set_xticks(m_values)
        ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax1.legend(loc="lower right")
        
    plt.title("VQA Model Performance and Softmax Confidence vs. Visual Token Scale")
    plt.tight_layout()
    plt.savefig(plots_dir / "performance_vs_tokens.png", dpi=config.figure_dpi)
    plt.close()
    
    # ── 3. High vs. Low Variance Galleries ──────────────────────────────────
    # Sort results by stability to get highest and lowest variance
    sorted_results = sorted(results, key=lambda x: x["statistics"]["answer_stability"])
    
    # Highest variance = lowest stability
    highest_variance = sorted_results[:config.gallery_top_k]
    # Lowest variance = highest stability
    lowest_variance = sorted_results[-config.gallery_top_k:]
    
    # Save a Markdown gallery of highest and lowest variance examples
    gallery_path = plots_dir / "variance_gallery.md"
    with open(gallery_path, "w", encoding="utf-8") as f:
        f.write("# Prediction Variance & Stability Analysis Gallery\n\n")
        f.write("Below are the most and least stable VQA predictions across the token sweep scale.\n\n")
        
        f.write("## 🔴 Top 10 Highest-Variance (Unstable) Predictions\n")
        f.write("These predictions represent instances where the model's output fluctuated heavily as visual token scale changed, signaling epistemic uncertainty and potential hallucinations.\n\n")
        
        for i, r in enumerate(highest_variance[:10]):
            f.write(f"### Example {i+1} (QID: {r['question_id']})\n")
            f.write(f"- **Question:** {r['question']}\n")
            gt_ans_list = r.get('gt_answers', [])
            gt_str = str(gt_ans_list[:3]) if gt_ans_list else "N/A"
            f.write(f"- **Annotator GT:** {gt_str}\n")
            f.write(f"- **Answer Stability Score:** `{r['statistics']['answer_stability']:.4f}`\n")
            f.write("| Token Count (m) | Generated Answer | Softmax Conf | VQA Acc |\n")
            f.write("|---|---|---|---|\n")
            for m in m_values:
                m_str = str(m)
                m_data = r["results_by_m"][m_str]
                vqa_acc = m_data.get("vqa_accuracy")
                acc_str = f"{vqa_acc:.2f}" if vqa_acc is not None else "N/A"
                f.write(f"| {m} | `{m_data['answer']}` | {m_data['avg_softmax_conf']:.4f} | {acc_str} |\n")
            f.write("\n---\n\n")
            
        f.write("## 🟢 Top 10 Lowest-Variance (Highly Stable) Predictions\n")
        f.write("These predictions remained structurally identical across token dimensions, signifying high epistemic confidence and a well-grounded visual answer.\n\n")
        
        for i, r in enumerate(reversed(lowest_variance[-10:])):
            f.write(f"### Example {i+1} (QID: {r['question_id']})\n")
            f.write(f"- **Question:** {r['question']}\n")
            gt_ans_list = r.get('gt_answers', [])
            gt_str = str(gt_ans_list[:3]) if gt_ans_list else "N/A"
            f.write(f"- **Annotator GT:** {gt_str}\n")
            f.write(f"- **Answer Stability Score:** `{r['statistics']['answer_stability']:.4f}`\n")
            f.write("| Token Count (m) | Generated Answer | Softmax Conf | VQA Acc |\n")
            f.write("|---|---|---|---|\n")
            for m in m_values:
                m_str = str(m)
                m_data = r["results_by_m"][m_str]
                vqa_acc = m_data.get("vqa_accuracy")
                acc_str = f"{vqa_acc:.2f}" if vqa_acc is not None else "N/A"
                f.write(f"| {m} | `{m_data['answer']}` | {m_data['avg_softmax_conf']:.4f} | {acc_str} |\n")
            f.write("\n---\n\n")
            
    print(f"Variance galleries generated at: {plots_dir / 'variance_gallery.md'}")
    print(f"Variance plots saved to:        {plots_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate variance and stability plots.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_CONFIG.output_dir)
    parser.add_argument("--plots-dir", type=str, default=DEFAULT_CONFIG.plots_dir)
    args = parser.parse_args()
    
    cfg = EvalConfig(output_dir=args.output_dir, plots_dir=args.plots_dir)
    generate_variance_plots(cfg)

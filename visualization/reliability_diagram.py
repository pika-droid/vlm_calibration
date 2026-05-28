"""
Calibration analysis: Reliability diagrams and Expected Calibration Error (ECE).

Generates:
1. An 8-panel figure showing the reliability diagrams at each token depth (2 to 256).
2. Computes and plots ECE and MCE values for each scale.
3. Overlays the perfect calibration diagonal and shows bin density.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from evaluation.config import DEFAULT_CONFIG, EvalConfig
from visualization.variance_plots import set_premium_style


def compute_calibration_metrics(
    accuracies: list[float], confidences: list[float], num_bins: int = 15
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Compute bin-level calibration statistics, ECE, and MCE.

    Args:
        accuracies: List of sample accuracies (VQA scores ∈ [0, 1]).
        confidences: List of sample confidence scores (softmax probabilities ∈ [0, 1]).
        num_bins: Number of confidence bins.

    Returns:
        Tuple of:
            - bin_edges: edges of the bins (length num_bins + 1)
            - bin_accuracies: mean accuracy per bin (length num_bins)
            - bin_confidences: mean confidence per bin (length num_bins)
            - ECE: Expected Calibration Error
            - MCE: Maximum Calibration Error
    """
    accuracies = np.array(accuracies)
    confidences = np.array(confidences)
    
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_accuracies = np.zeros(num_bins)
    bin_confidences = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)
    
    ece = 0.0
    mce = 0.0
    
    for i in range(num_bins):
        # Determine which samples fall into the current bin
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]
        
        # Keep upper bound inclusive for the last bin
        if i == num_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            
        bin_counts[i] = np.sum(in_bin)
        
        if bin_counts[i] > 0:
            bin_accuracies[i] = np.mean(accuracies[in_bin])
            bin_confidences[i] = np.mean(confidences[in_bin])
            
            # Weighted absolute difference for ECE
            bin_weight = bin_counts[i] / len(confidences)
            bin_diff = np.abs(bin_accuracies[i] - bin_confidences[i])
            ece += bin_weight * bin_diff
            
            # Maximum difference for MCE
            mce = max(mce, bin_diff)
            
    return bin_edges, bin_accuracies, bin_confidences, ece, mce


def plot_reliability_diagrams(config: EvalConfig) -> None:
    """Generate reliability diagrams for all token depths in an 8-panel layout."""
    set_premium_style()
    
    jsonl_path = config.results_jsonl_path()
    plots_dir = Path(config.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    if not jsonl_path.exists():
        print(f"Error: Results file not found at {jsonl_path}")
        return
        
    # Load results
    results = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))
            
    m_values = config.token_sweep
    num_panels = len(m_values)
    
    # Dynamically compute grid layout based on number of token scales
    ncols = min(4, num_panels)
    nrows = math.ceil(num_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), sharex=True, sharey=True)
    axes_flat = np.atleast_1d(axes).flatten()
    
    ece_values = []
    mce_values = []
    
    print("\n=== Calibration Error Analysis ===")
    for idx, m in enumerate(m_values):
        ax = axes_flat[idx]
        m_str = str(m)
        
        # Extract accuracies and confidences for this m, filtering out unlabeled samples
        valid_pairs = [
            (r["results_by_m"][m_str]["vqa_accuracy"], r["results_by_m"][m_str]["avg_softmax_conf"])
            for r in results
            if r["results_by_m"][m_str]["vqa_accuracy"] is not None
        ]
        
        if not valid_pairs:
            print(f"Token count (m) = {m:<3} | No labeled samples. Skipping reliability diagram.")
            ece_values.append(0.0)
            mce_values.append(0.0)
            ax.text(0.5, 0.5, "No Labels Available", ha="center", va="center", color="#94A3B8")
            ax.set_title(f"m = {m} Tokens", fontsize=12, fontweight="bold")
            continue
            
        accs, confs = zip(*valid_pairs)
        
        # Compute metrics
        bin_edges, bin_accs, bin_confs, ece, mce = compute_calibration_metrics(
            list(accs), list(confs), num_bins=config.ece_num_bins
        )
        
        ece_values.append(ece)
        mce_values.append(mce)
        print(f"Token count (m) = {m:<3} | ECE: {ece:.4f} | MCE: {mce:.4f}")
        
        # ── Draw Reliability Diagram Panel ──────────────────────────────────
        # Draw perfect calibration diagonal
        ax.plot([0, 1], [0, 1], linestyle="--", color="#64748B", linewidth=1.5, label="Perfect Calibration")
        
        # Plot empirical bin accuracies (bars)
        bin_width = 1.0 / config.ece_num_bins
        # Fill gaps with a light red to visualize the gap/error
        ax.bar(
            bin_edges[:-1],
            bin_accs,
            width=bin_width,
            align="edge",
            color="#3B82F6",
            edgecolor="#2563EB",
            linewidth=0.8,
            alpha=0.85,
            label="Accuracy"
        )
        
        # Visualize the calibration gap as red bars on top of accuracy
        gap = np.clip(bin_confs - bin_accs, 0, None)
        ax.bar(
            bin_edges[:-1],
            gap,
            bottom=bin_accs,
            width=bin_width,
            align="edge",
            color="#EF4444",
            edgecolor="#DC2626",
            linewidth=0.8,
            alpha=0.25,
            label="Calibration Gap"
        )
        
        # Add labels and annotations inside the panel
        ax.set_title(f"m = {m} Tokens", fontsize=12, fontweight="bold")
        ax.text(
            0.05, 0.90,
            f"ECE: {ece:.4%}\nMCE: {mce:.2%}",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8FAFC", edgecolor="#E2E8F0", alpha=0.9),
            fontsize=10,
            verticalalignment="top"
        )
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Show labels only on outer panels to clean up presentation
        if idx >= ncols * (nrows - 1):
            ax.set_xlabel("Softmax Confidence")
        if idx % ncols == 0:
            ax.set_ylabel("Empirical Accuracy")
            
    plt.suptitle("Reliability Diagrams & Expected Calibration Error (ECE) across Scale Sweeps", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(plots_dir / "reliability_diagrams_multi.png", dpi=config.figure_dpi)
    plt.close()
    
    # Save the ECE and MCE data to JSON for the summary script
    calib_summary = {
        "m_values": m_values,
        "ece": ece_values,
        "mce": mce_values
    }
    with open(plots_dir / "calibration_stats.json", "w") as f:
        json.dump(calib_summary, f, indent=2)
        
    print(f"Reliability diagrams saved to: {plots_dir / 'reliability_diagrams_multi.png'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot reliability diagrams.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_CONFIG.output_dir)
    parser.add_argument("--plots-dir", type=str, default=DEFAULT_CONFIG.plots_dir)
    args = parser.parse_args()
    
    cfg = EvalConfig(output_dir=args.output_dir, plots_dir=args.plots_dir)
    plot_reliability_diagrams(cfg)

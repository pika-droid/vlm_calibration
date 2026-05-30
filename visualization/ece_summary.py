"""
Summarize calibration metrics (ECE and MCE) across token depths.

Generates:
1. Comparison bar charts for ECE and MCE values at each token count.
2. Tabulates the numerical calibration metrics for reports.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

# Configure matplotlib to use non-interactive Agg backend and suppress font warnings
import matplotlib
matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from evaluation.config import DEFAULT_CONFIG, EvalConfig
from visualization.variance_plots import set_premium_style


def generate_ece_summary(config: EvalConfig) -> None:
    """Load calibration statistics and generate summary ECE/MCE charts."""
    set_premium_style()
    
    plots_dir = Path(config.plots_dir)
    stats_path = plots_dir / "calibration_stats.json"
    
    if not stats_path.exists():
        print(f"Error: Calibration stats file not found at {stats_path}. Run reliability_diagram.py first.")
        return
        
    with open(stats_path, "r") as f:
        data = json.load(f)
        
    m_values = data["m_values"]
    ece_vals = data["ece"]
    mce_vals = data["mce"]
    
    # Create pandas DataFrame for easy plotting and exporting
    df = pd.DataFrame({
        "Token Count (m)": m_values,
        "ECE": ece_vals,
        "MCE": mce_vals
    })
    
    if not ece_vals:
        print("No calibration data available to plot.")
        return
    
    # ── 1. Plot ECE vs Token Count Bar Chart ───────────────────────────────
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Bar plot for ECE
    bars = ax1.bar(
        [str(m) for m in m_values],
        [e * 100 for e in ece_vals],  # convert to percentages
        color="#3B82F6",
        edgecolor="#2563EB",
        width=0.6,
        alpha=0.85,
        label="ECE"
    )
    
    # Add values on top of the bars
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(
            f"{height:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="#1E293B"
        )
        
    ax1.set_title("Expected Calibration Error (ECE) across Visual Token Scales")
    ax1.set_xlabel("Visual Token Footprint (m)")
    ax1.set_ylabel("Expected Calibration Error (%)")
    ax1.set_ylim(0, max([e * 100 for e in ece_vals]) * 1.15)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "ece_comparison.png", dpi=config.figure_dpi)
    plt.close()
    
    # ── 2. Display Numerical Table ──────────────────────────────────────────
    print("\n" + "=" * 50)
    print(" VLM Calibration Summary (ECE & MCE)")
    print("=" * 50)
    print(f"{'Token Count (m)':<17} | {'ECE (%)':<12} | {'MCE (%)':<12}")
    print("-" * 50)
    for m, ece, mce in zip(m_values, ece_vals, mce_vals):
        print(f"{m:<17} | {ece*100:<12.4f} | {mce*100:<12.4f}")
    print("=" * 50)
    
    # Save markdown table
    with open(plots_dir / "calibration_metrics_table.md", "w") as f:
        f.write("# Calibration Metrics Summary Table\n\n")
        f.write("| Token Count (m) | ECE (%) | MCE (%) |\n")
        f.write("|---|---|---|\n")
        for m, ece, mce in zip(m_values, ece_vals, mce_vals):
            f.write(f"| {m} | {ece*100:.4f}% | {mce*100:.4f}% |\n")
            
    print(f"Summary table exported to: {plots_dir / 'calibration_metrics_table.md'}")
    print(f"ECE comparison bar chart saved to: {plots_dir / 'ece_comparison.png'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate ECE comparison summary.")
    parser.add_argument("--plots-dir", type=str, default=DEFAULT_CONFIG.plots_dir)
    args = parser.parse_args()
    
    cfg = EvalConfig(plots_dir=args.plots_dir)
    generate_ece_summary(cfg)

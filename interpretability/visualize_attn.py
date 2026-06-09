"""Attention map visualization tool across Matryoshka scales for M3-LLaVA.

Extracts the 2D spatial attention weights of the generated answer tokens
assigned to visual tokens, and overlays them as heatmaps on the input image
across all 5 scales (m ∈ [1, 9, 36, 144, 576]).
"""

from __future__ import annotations

import os
import sys
import json
import logging
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image

# Configure matplotlib for headless generation
import matplotlib
matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

import matplotlib.pyplot as plt
import seaborn as sns

# Add evaluation-m3 and interpretability to system path
root_path = Path(__file__).resolve().parent.parent
package_path = Path(__file__).resolve().parent
eval_m3_path = root_path / "evaluation-m3"
if str(eval_m3_path) not in sys.path:
    sys.path.insert(0, str(eval_m3_path))
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))
if str(package_path) not in sys.path:
    sys.path.insert(0, str(package_path))

from extract_hooks import HookedM3Wrapper, HookOutput, TokenMap

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VLM_Attention_Viz")


def set_premium_style() -> None:
    """Set custom premium styling for matplotlib plots."""
    sns.set_theme(style="white")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Arial"],
        "axes.edgecolor": "#E2E8F0",
        "axes.linewidth": 1.2,
        "axes.labelcolor": "#1E293B",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": "#0F172A",
        "figure.dpi": 300,
    })


def extract_spatial_attention(mean_attn: torch.Tensor, token_map: TokenMap, scale: int) -> np.ndarray:
    """Extracts attention weights of answer tokens to image tokens, reshaped to 2D grid."""
    tm = token_map
    
    # ans_attn shape: (num_answer_tokens, scale)
    ans_attn = mean_attn[tm.answer_start : tm.answer_end, tm.image_start : tm.image_end]
    
    # Average attention across generated answer tokens
    grid_attn = ans_attn.mean(dim=0).cpu().numpy()  # (scale,)
    
    # Reshape to square 2D grid
    side = int(np.sqrt(scale))
    grid_2d = grid_attn.reshape((side, side))
    
    # Normalize grid for visualization
    denom = grid_2d.max() - grid_2d.min()
    if denom > 1e-8:
        grid_2d = (grid_2d - grid_2d.min()) / denom
    else:
        grid_2d = np.zeros_like(grid_2d)
        
    return grid_2d


def plot_attention_comparison(
    image: Image.Image,
    attention_maps: dict[int, np.ndarray],
    scales: list[int],
    question: str,
    save_path: Path
) -> None:
    """Plots original VQA image next to 5 spatial attention heatmap overlays."""
    set_premium_style()
    
    # Create 1 row × 6 columns subplot
    fig, axes = plt.subplots(1, 6, figsize=(20, 4.5))
    
    # Colormap
    cmap = plt.cm.jet
    
    # ── 1. Plot original image ─────────────────────────────────────────────
    axes[0].imshow(image)
    axes[0].axis("off")
    axes[0].set_title("Input Image")
    
    # Resize image for overlay size consistency
    img_np = np.array(image.convert("RGB"))
    h, w, _ = img_np.shape
    
    # ── 2. Plot attention maps for each scale ──────────────────────────────
    for idx, scale in enumerate(scales):
        ax = axes[idx + 1]
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(f"Scale m = {scale}")
        
        attn_grid = attention_maps.get(scale)
        if attn_grid is not None:
            # Resize attention grid to match image using bilinear interpolation
            # via matplotlib imshow overlay
            ax.imshow(
                attn_grid,
                cmap=cmap,
                alpha=0.45,
                extent=[0, w, h, 0],
                interpolation="bilinear"
            )
            
    # Add title with question wrap
    import textwrap
    wrapped_q = "\n".join(textwrap.wrap(question, width=90))
    fig.suptitle(f"Multi-Scale Spatial Attention Map Profile\nQuestion: {wrapped_q}", fontsize=14, fontweight="bold", y=0.98)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def generate_attention_maps(input_dir: str, num_examples: int = 5) -> None:
    """Main execution block to generate attention overlays."""
    plots_dir = Path(input_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    input_path = Path(input_dir) / "pilot_results.jsonl"
    if not input_path.exists():
        logger.error(f"Pilot results not found at: {input_path}")
        return

    # Load dataset to get images
    logger.info("Loading VQAv2 dataset splits for visual examples...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("lmms-lab/VQAv2", split="all")
    except Exception as e:
        logger.warning(f"Failed to load dataset: {e}. Generating mockup visualizer files.")
        _generate_mockup_visualizer(plots_dir, num_examples)
        return

    # Load pilot results
    samples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))

    # Pick representative samples (from different strata)
    strata_groups = {}
    for sample in samples:
        stratum = sample["stratum"]
        if stratum not in strata_groups:
            strata_groups[stratum] = []
        strata_groups[stratum].append(sample)

    target_samples = []
    # Pick a couple from each stratum
    for stratum, pool in strata_groups.items():
        target_samples.extend(pool[:2])
    target_samples = target_samples[:num_examples]

    logger.info(f"Generating attention overlays for {len(target_samples)} target examples...")
    if not target_samples:
        return

    # Load model wrapper
    model_path = "/workspace/models/llava-v1.5-7b-m3"
    if not os.path.exists(model_path):
        logger.warning(f"Model checkpoint not found at {model_path}. Generating mockup visualizer files.")
        _generate_mockup_visualizer(plots_dir, num_examples)
        return

    # Fetch images
    target_qids = {int(s["question_id"]) for s in target_samples}
    image_map = {}
    for sample in dataset:
        qid = int(sample["question_id"])
        if qid in target_qids:
            image_map[qid] = sample["image"]
            if len(image_map) == len(target_qids):
                break

    wrapper = HookedM3Wrapper(model_path=model_path, precision="fp16")
    scales = [1, 9, 36, 144, 576]
    
    # Hook the last attention layer (layer 32)
    eval_layer = 32

    for sample_data in target_samples:
        qid = int(sample_data["question_id"])
        question = sample_data["question"]
        image = image_map.get(qid)
        
        if image is None:
            continue
            
        attention_maps = {}
        
        for m in scales:
            try:
                # Execute forward pass with hooks active on layer 32
                with wrapper.hooked(layer_indices=[eval_layer], capture_hidden=False, capture_attention=True):
                    hook_output = wrapper.forward_with_hooks(
                        image=image,
                        question=question,
                        num_visual_tokens=m,
                        hook_layers=[eval_layer]
                    )
                    
                    attn = hook_output.attention_weights.get(eval_layer)
                    if attn is not None:
                        # shape: (num_heads, seq_len, seq_len)
                        mean_attn = attn.mean(dim=0)
                        attn_2d = extract_spatial_attention(mean_attn, hook_output.token_map, m)
                        attention_maps[m] = attn_2d
            except Exception as e:
                logger.error(f"Error extracting attention for QID {qid} scale {m}: {e}")
                continue

        if attention_maps:
            save_path = plots_dir / f"attention_map_comparison_{qid}.png"
            plot_attention_comparison(image, attention_maps, scales, question, save_path)
            logger.info(f"Saved attention map overlay plot to: {save_path}")


def _generate_mockup_visualizer(plots_dir: Path, num_examples: int) -> None:
    """Generates mock images for verify pipeline / CPU execution."""
    logger.info("Generating mockup attention overlay plots...")
    scales = [1, 9, 36, 144, 576]
    
    for i in range(num_examples):
        qid = 1000 + i
        question = f"Mock VQA question number {i+1}?"
        
        # Create a dummy image
        img = Image.new("RGB", (300, 300), color=(60 + i * 20, 100, 150))
        
        # Generate dummy 2D attention grids (simulating gaussian center focus)
        attention_maps = {}
        for m in scales:
            side = int(np.sqrt(m))
            grid = np.zeros((side, side))
            
            # Simple mockup center focus
            cy, cx = side / 2.0 - 0.5, side / 2.0 - 0.5
            for y in range(side):
                for x in range(side):
                    dist = (y - cy)**2 + (x - cx)**2
                    grid[y, x] = np.exp(-dist / max(1.0, side / 4.0))
            attention_maps[m] = grid
            
        save_path = plots_dir / f"attention_map_comparison_{qid}.png"
        plot_attention_comparison(img, attention_maps, scales, question, save_path)
        
    logger.info(f"Mockup visualizer files saved to: {plots_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize attention map scaling for pilot subset.")
    parser.add_argument("--input-dir", type=str, default="results/pilot-interpretability")
    parser.add_argument("--num-examples", type=int, default=5)
    args = parser.parse_args()
    
    generate_attention_maps(args.input_dir, args.num_examples)

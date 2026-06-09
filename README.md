# interpretability

Interpretability and mid-layer diagnostics package for **M3-LLaVA** (Matryoshka Multimodal Model), branched from `m3-test`. This suite extracts hidden states, token attention weights, and maps intermediate calibration/interpretability features across visual scales ($m \in [1, 9, 36, 144, 576]$) on a representative 1,000-sample pilot subset of VQAv2.

---

## Overview

Modern VLMs like M3-LLaVA exhibit answer shift and potential hallucinations when visual token scale is compressed. This diagnostics suite implements four advanced mechanistic interpretability techniques to probe the model's internal representation space across scales:

1. **Visual Attention Ratio (VAR):** Measures the relative attention weight allocated to visual tokens versus text query tokens in middle layers (layers 5–18) to identify hallucinations.
2. **Intermediate Softmax Calibration (LogitLens):** Projects intermediate hidden states $\mathbf{h}^{(\ell)}$ directly to the vocabulary using the language model head, evaluating calibration quality (ECE) across layers.
3. **Activation Back-patching:** A training-free intervention that records visual token key/value states in later layers (e.g., layer 24) and patches them into earlier layers (e.g., layer 10) to evaluate accuracy and calibration recovery.
4. **Attention Map Overlays:** Renders 2D spatial attention grids of the generated response token over the input image across all 5 scales.

---

## Directory Structure

```
interpretability/
├── __init__.py
├── extract_hooks.py      # HookedM3Wrapper: PyTorch forward hooks manager
├── run_pilot.py          # Pilot selection (4 strata) and inference runner
├── analyze_var.py        # Visual Attention Ratio (VAR) evaluation and plot generation
├── latent_lens.py        # LogitLens projection, ECE tracking, and temp scaling
├── backpatching.py       # Activation back-patching experiment grid and heatmaps
├── visualize_attn.py     # 2D attention heatmap overlay visualizer
└── verify_pipeline.py    # Local CPU-only end-to-end mock verification script
```

---

## Strata Definition & Selection

The pilot subset of **1,000 samples** is selected using a seeded (`seed=42`) stratified sample over the full 67,583-sample VQAv2 evaluation results:

| Stratum | Definition | Count | Diagnostic Goal |
|---|---|---|---|
| **Stable Correct** | Accuracy $\ge 0.5$ at all 5 scales | 400 | Baseline for optimal representation space |
| **Stable Incorrect (Strict)** | Accuracy $< 0.5$ at all scales, identical answer string | 200 | Probes stable failure modes/hallucinations |
| **Stable Incorrect (Relaxed)** | Accuracy $< 0.5$ at all scales, answer changes | 200 | Probes high-uncertainty failure modes |
| **Flip** | Accuracy $\ge 0.5$ at some scales, $< 0.5$ at others | 200 | Probes scale-dependent instability |

---

## Interpretability Metrics

### 1. Visual Attention Ratio (VAR)
Defined for a decoder layer $\ell$ as:
$$\text{VAR}^{(\ell)} = \frac{\sum \text{Attention}(\text{Answer Token} \rightarrow \text{Image Tokens})}{\sum \text{Attention}(\text{Answer Token} \rightarrow \text{Text Tokens})}$$
Calculated across layers 5–18. Correct stable answers show a significantly higher VAR than wrong stable answers.

### 2. LogitLens & Temperature Scaling
Hidden state $\mathbf{h}^{(\ell)}$ at the final prefill token position is projected as:
$$\mathbf{p}^{(\ell)} = \text{softmax}\left(\text{LM\_Head}(\text{RMSNorm}(\mathbf{h}^{(\ell)}))\right)$$
We compute Expected Calibration Error (ECE) on $\mathbf{p}^{(\ell)}$. Since we capture only top-1 confidence $p$, we run post-hoc temperature scaling using a binary log-odds logit:
$$z = \log\left(\frac{p}{1-p}\right) \quad \rightarrow \quad p_{\text{scaled}} = \sigma\left(\frac{z}{T}\right)$$
We fit $T$ by minimizing Negative Log-Likelihood (NLL).

### 3. Activation Back-patching
We execute two-pass inference:
* **Pass 1:** Run inference normally, caching visual token hidden states at layer $L_{\text{source}}$.
* **Pass 2:** Re-run inference, placing a forward hook at layer $L_{\text{dest}}$ that overrides the visual token hidden states with the cached representations.

---

## Usage

### 1. Local CPU Verification (End-to-End Mock Run)
Verify the entire pipeline's logic, mathematics, shape compatibility, and plotting without loading weights or requiring GPU VRAM:
```bash
python -m interpretability.verify_pipeline
```

### 2. Stratified Pilot Extraction (GPU required)
Select the 1,000 pilot samples and run inference with active hooks, extracting VAR and LogitLens statistics:
```bash
python -m interpretability.run_pilot --model-path /workspace/models/llava-v1.5-7b-m3 \
    --source-results results/vlm-calibration-m3/results/multi_scale_results.jsonl \
    --output-dir results/pilot-interpretability
```
*Note: Debug mode is supported for quick GPU testing on a single sample:*
```bash
python -m interpretability.run_pilot --debug --sample-idx 0
```

### 3. Run Interpretability Diagnostics
Analyze the captured statistics and generate plots:
```bash
# 1. Analyze VAR classification capacity
python -m interpretability.analyze_var --input-dir results/pilot-interpretability

# 2. Analyze LogitLens calibration and ECE per layer
python -m interpretability.latent_lens --input-dir results/pilot-interpretability

# 3. Execute the back-patching experiment grid
python -m interpretability.backpatching --input-dir results/pilot-interpretability --num-samples 50

# 4. Generate 2D spatial attention overlays
python -m interpretability.visualize_attn --input-dir results/pilot-interpretability --num-examples 10
```

---

## Output Plots

Under `results/pilot-interpretability/plots/`:

| Plot | Description |
|---|---|
| `var_distribution.png` | Violin plots showing VAR values across the 4 strata |
| `var_roc_curve.png` | ROC curve of VAR as a classifier for stable correctness |
| `var_by_layer.png` | Trajectory of VAR values across layers 5–18 |
| `intermediate_ece_curve.png` | ECE bar charts per decoder layer (raw vs temperature scaled) |
| `layer_confidence_evolution.png` | Confidence buildup trajectory across decoder layers |
| `temperature_scaling_comparison.png` | Fitted temperature parameter $T$ values per layer |
| `backpatching_results.png` | Heatmap of accuracy shift per source-destination combination |
| `backpatching_ece.png` | Heatmap of answer modification rate |
| `attention_map_comparison_{qid}.png` | Overlaid attention heatmaps on original image across the 5 scales |

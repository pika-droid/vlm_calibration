# evaluation-m3

Multi-scale calibration evaluation pipeline for **M3-LLaVA** (Matryoshka Multimodal Model), adapted from the base `evaluation/` pipeline. This harness sweeps across nested visual token counts, extracts token-level confidence scores, and measures VQA calibration on the VQAv2 dataset.

---

## Overview

M3-LLaVA supports **elastic visual token inference** via a Resampler (cross-attention pooling). Instead of always using all 576 visual tokens (a full 24×24 grid), the model can pool them into progressively smaller grids:

| Token count (m) | Grid | Pooling factor |
|---|---|---|
| 576 | 24×24 | None (full resolution) |
| 144 | 12×12 | 2×2 |
| 36 | 6×6 | 4×4 |
| 9 | 3×3 | 8×8 |
| 1 | 1×1 | 24×24 (maximum compression) |

For each VQAv2 sample, the pipeline runs one greedy forward pass at each token depth, collects the generated answer and confidence score, and computes calibration statistics across the sweep.

---

## Directory Structure

```
evaluation-m3/
├── __init__.py
├── config.py                    # All tunable hyperparameters (EvalConfig dataclass)
├── model_wrapper.py             # M3LLaVAWrapper: model loading, prompt formatting, inference
├── multi_scale_harness.py       # Main evaluation loop with checkpointing and CSV export
├── smoke_test.py                # Quick sanity check on a single VQAv2 sample
├── load_vqav2.py                # Dataset loading helpers
├── verify_pipeline.py           # End-to-end pipeline verification script
├── calibrate_oracle.py          # Oracle-weighted calibration analysis runner
└── visualization/
    ├── reliability_diagram.py   # Reliability diagrams and ECE/MCE per token depth
    ├── variance_plots.py        # Confidence and stability variance plots across scales
    └── ece_summary.py           # ECE summary bar chart across all token depths

```

---

## How It Works

### 1. Confidence Score Extraction

Confidence is derived entirely from the model's own logits — no sampling is used.

- `model.generate()` runs **greedy decoding** (`do_sample=False`, `temperature=0.0`) and, because `output_scores=True` is set, also returns the raw logit vectors at each generation step.
- For each generated token, we apply `torch.softmax(logits, dim=-1)` over the full 32,000-token vocabulary.
- The confidence at that step is `probs[chosen_token_id]` — the probability assigned to the token that was actually selected via argmax.
- The final per-sample confidence is the **geometric mean** of all non-EOS token probabilities across the generated sequence.

### 2. VQA Accuracy

Predicted answers are compared against VQAv2 ground-truth annotations using the standard VQA scoring rule:

```
accuracy = min(1.0, number_of_matching_annotators / 3)
```

Answers are normalized before comparison (lowercased, punctuation stripped, articles removed).

### 3. Answer Stability

After all five token depths are swept for one sample, the five generated answers are encoded using `all-MiniLM-L6-v2` (sentence-transformers, run on CPU to conserve GPU VRAM). The pairwise cosine similarities across the five embeddings are averaged to produce an **answer stability score** (1.0 = identical answers at all scales).

### 4. ECE Calculation

ECE (Expected Calibration Error) is computed per token depth. Samples are binned by confidence, and the average |confidence − accuracy| gap per bin is weighted by bin population. Since VQA produces free-form text (not fixed classes), confidence is the token-level softmax probability described above — equivalent to treating the 32k-vocabulary as the class set for a single-token classification at each decoding step.

### 5. Checkpointing

Results are written line-by-line to a JSONL file (`multi_scale_results.jsonl`). If the run is interrupted, it resumes from where it stopped by counting completed lines. Intermediate plots are regenerated every 500 samples, and full archive snapshots are saved every 2000 samples.

---

## Usage

### Smoke Test (single sample, 2 token depths)

```bash
python -m evaluation-m3.smoke_test
```

### Full Evaluation

```bash
# Run on a subset of 500 samples
python -m evaluation-m3.multi_scale_harness --subset-size 500

# Run on full VQAv2 (~1.1M samples), in the background
nohup python -m evaluation-m3.multi_scale_harness > logs/full_run.log 2>&1 &
```

### Generate Plots from Existing Results

```bash
# Reliability diagrams
python -m evaluation-m3.visualization.reliability_diagram

# Variance plots
python -m evaluation-m3.visualization.variance_plots

# ECE summary
python -m evaluation-m3.visualization.ece_summary
```

### Oracle-Weighted Calibration (Sequential Optimization)

Process raw token-sweep results to simulate a sequential oracle-updating scheme:
```bash
python -m evaluation-m3.calibrate_oracle \
    --input-file results/vlm-calibration-m3/results/multi_scale_results.jsonl \
    --output-dir results/m3-weighted-confidence \
    --beta 1.0 \
    --penalize-new
```
*Note: This will perform post-hoc temperature/sequential scaling and output ECE/MCE metrics comparisons, reliability diagrams, and comparison markdown tables.*


---

## Configuration

All parameters are in [`config.py`](./config.py) inside the `EvalConfig` dataclass. Key settings:

| Parameter | Default | Description |
|---|---|---|
| `model_path` | `mucai/llava-v1.5-7b-m3` | HuggingFace model ID or local path |
| `token_sweep` | `[1, 9, 36, 144, 576]` | Visual token counts to evaluate |
| `dataset_name` | `lmms-lab/VQAv2` | HuggingFace dataset ID |
| `subset_size` | `None` (full dataset) | Limit evaluation to N samples |
| `temperature` | `0.0` | Greedy decoding |
| `max_new_tokens` | `64` | Max generated tokens per answer |
| `checkpoint_interval` | `500` | Regenerate plots every N samples |
| `archive_interval` | `2000` | Archive snapshot every N samples |
| `ece_num_bins` | `15` | Bins for ECE/reliability diagrams |
| `embedding_model` | `all-MiniLM-L6-v2` | Sentence encoder for stability |

---

## Issues Faced During Deployment (RunPod)

The following issues were encountered while setting up and running this pipeline on the **RunPod PyTorch 2.8.0** template.

---

### Issue 1 — PyTorch Version Conflict

**Problem:** `matryoshka-mm` had hard pins to `torch==2.1.2` and `torchvision==0.16.2`. No Python 3.12 wheels exist for these versions, causing pip to crash during installation.

**Fix:** Recreated the virtual environment with `--system-site-packages` to inherit the RunPod pre-installed `torch 2.8.0`, then used `sed` to relax the pins in `pyproject.toml` and `requirements.txt` to `torch>=2.1.2`.

---

### Issue 2 — CUDA Diagnostic Typo (`props.total_mem`)

**Problem:** Setup script's final diagnostic failed with `AttributeError: 'torch._C._CudaDeviceProperties' object has no attribute 'total_mem'`.

**Fix:** Corrected the attribute name to `props.total_memory`. This was aesthetic-only; core installations had already completed.

---

### Issue 3 — NumPy 2.x Binary Incompatibility

**Problem:** `lmms-eval` pulled in NumPy 2.x, which caused binary incompatibility crashes (`ValueError: numpy.dtype size changed`) in compiled C-extensions like `scikit-learn` when importing `transformers`.

**Fix:**
```bash
pip install "numpy<2.0"
```

---

### Issue 4 — Relocated `apply_chunking_to_forward` Import

**Problem:** Newer `transformers` v4.x moved `apply_chunking_to_forward` from `transformers.modeling_utils` to `transformers.pytorch_utils`. The `matryoshka-mm` projector builder still used the old import path, causing an `ImportError`.

**Fix:** Patched `builder.py` to use a `try...except ImportError` fallback between the two module locations.

---

### Issue 5 — Removed Pruning Helpers in Transformers v5.x

**Problem:** The environment defaulted to `transformers` v5.x, which completely removed `find_pruneable_heads_and_indices` and `prune_linear_layer`, causing an `ImportError` on startup.

**Fix:**
```bash
pip install "transformers<5.0.0" "tokenizers<0.20.0" "accelerate<1.0.0"
```
Also patched `builder.py` with import fallback logic for these helpers.

---

### Issue 6 — `**kwargs` Syntax Error in `llava_llama.py` (M3 fork)

**Problem:** The `matryoshka-mm` fork's `LlavaLlamaForCausalLM.forward()` had a malformed signature — `**kwargs` appeared before the `matryoshka_vis_token_scale` keyword argument, causing a `SyntaxError: arguments cannot follow var-keyword argument` at import time.

**Root cause:** A patch applied to add `**kwargs` to the signature was inserted at the wrong position in the function definition.

**Fix:** Rewrote the `forward()` signature to place `matryoshka_vis_token_scale: Optional[int] = None` before `**kwargs`, and ensured `**kwargs` appeared only once.

---

### Issue 7 — `cache_position` Passed Twice to `forward()` (TypeError)

**Problem:** `transformers` v4.39.0+ passes `cache_position` as an explicit keyword argument when calling `model.forward()` inside `.generate()`. Since `LlavaLlamaForCausalLM.forward()` already received it via `**kwargs` and then also tried to pass it explicitly to `super().forward()`, this caused a `TypeError: got multiple values for argument 'cache_position'`.

**Fix:** Added `cache_position: Optional[torch.LongTensor] = None` explicitly to the `forward()` signature and passed it directly to `super().forward()`, filtering it out of `**kwargs` before the call.

---

### Issue 8 — TorchCodec Binary Incompatibility

**Problem:** `sentence-transformers` loaded `torchcodec==0.13.0`, which was binary-incompatible with `torch 2.8.0`, throwing `undefined symbol: torch_from_blob` and crashing the `SentenceTransformer` import.

**Fix:**
```bash
pip install "torchcodec==0.7.0"
```

---

### Issue 9 — Harness Exits Immediately (Stale JSONL Checkpoint)

**Problem:** After a previous partial run, the JSONL checkpoint file already had ≥ N lines, causing the harness to detect "all samples already processed" and exit immediately on the next run with a smaller `--subset-size`.

**Fix:**
```bash
rm /workspace/vlm-calibration-m3/results/multi_scale_results.jsonl
python -m evaluation-m3.multi_scale_harness --subset-size <N>
```

---

## Output Files

After a run and post-processing, the following files are produced:

| File / Folder | Description |
|---|---|
| `results/multi_scale_results.jsonl` | One JSON line per sample with full per-scale results |
| `results/summary_statistics.csv` | Flat CSV with per-sample stats for all token depths |
| `plots/reliability_diagrams_multi.png` | Reliability diagram grid (one panel per token depth) |
| `plots/calibration_stats.json` | ECE and MCE values per token depth |
| `plots/variance_*.png` | Confidence and accuracy variance plots across scales |
| `plots/ece_summary.png` | ECE summary bar chart across all token depths |
| `logs/evaluation.log` | Full run log with timestamps |
| `results/m3-weighted-confidence/` | **[Oracle Calibration]** Folder containing calibrated JSONL, summary statistics CSV, and ECE/MCE reduction comparison plots/reliability diagrams |


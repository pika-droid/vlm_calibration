# VLM Calibration: Matryoshka Visual Tokens & Confidence Estimation

This repository contains the evaluation harness and visualization pipeline for investigating uncertainty calibration in Vision-Language Models (VLMs), specifically focusing on elastic visual token configurations using the **Matryoshka Query Transformer LLaVA (MQT-LLaVA)**.

---

## What Has Been Done

### 1. Multi-Scale Evaluation Loop & Setup
* **Environment Provisioning**: Created `setup/runpod_setup.sh` to fully configure dependencies (lmms-eval, mqt-llava, sentence-transformers, etc.) inside RunPod's persistent volume.
- **Harness Implementation**: Built `evaluation/multi_scale_harness.py` to sweep across visual token depths ($m \in [2, 4, 8, 16, 36, 64, 144, 256]$) per sample.
- **Backwards Compatibility & Checkpointing**: Added robust checkpointing (`results/multi_scale_results.jsonl`) to support resuming interrupted sweeps. Added support for unlabeled splits (like test splits) where ground-truth answers are missing.
- **Intermediate Updates & Snapshots**: Added hooks to automatically update the live visualizations every 500 samples (`checkpoint_interval`), and create archived snapshots (figures, markdown table, ECE bar chart, and snapshot copy of CSV statistics) in unique folders (e.g. `plots/snapshot_10000/`) every 10k samples (`archive_interval`).
- **Configurability**: Structured settings within `evaluation/config.py`.

### 2. PyTorch & RunPod Optimizations
* **Image & Token Caching (`sweep_optimized`)**: Reduced image preprocessing (PIL -> Tensor conversion) and prompt tokenization frequency from 8 times per sample to **exactly once**.
* **Mixed Precision (`torch.amp.autocast`)**: Implemented half-precision contexts for faster Tensor Core execution on NVIDIA GPUs.
* **Resampler Compilation (`torch.compile`)**: Applied JIT compilation with `dynamic=True` and `mode="reduce-overhead"` to MQT-LLaVA's query abstractor module.
* **VRAM and CPU Pinning**: Pinned the sentence embeddings model (`SentenceTransformer`) to CPU to conserve GPU VRAM for the 7B LLM. Optimized dataloader parameters (`pin_memory=True`, `prefetch_factor=2`) and dynamically scale dataloader threads (`num_workers` set to half of available vCPUs) to fit your RunPod's hardware footprint.
* **Reproducibility**: All random seeds (`torch`, `CUDA`, `numpy`, `random`, `cudnn.deterministic`) are set via `set_seed(config.seed)` at the start of each evaluation run, following PyTorch best practices.

> **Security Note**: MQT-LLaVA's `load_pretrained_model()` internally calls `torch.load()` without `weights_only=True`. This is inherited from the upstream LLaVA codebase. Only load model checkpoints from trusted sources (e.g. the official HuggingFace repo `gordonhu/MQT-LLaVA-7b`).

### 3. Visualizations & Analytical Tools
- **Visual Variance Profiling**: Updates `visualization/variance_plots.py` to retrieve target PIL Images from the dataset, generate dual-panel stability plots (source image side-by-side with a detailed token-sweep results table), and embeds them directly inside `variance_gallery.md`.
- **Reliability Diagrams**: Generates 8-panel empirical accuracy vs confidence calibration charts (`visualization/reliability_diagram.py`).
- **ECE Summary**: Compares Expected Calibration Error (ECE) across different scale selections to visualize how calibration changes under token reduction (`visualization/ece_summary.py`).

---

## What Has Been Deferred

- **Full Vision Tower Caching**: The vision encoder (ViT) forward pass still runs inside `model.generate()` for each token scale sweep step. Since it operates on the full image, it produces the same visual features. Fully caching these features requires overriding MQT-LLaVA's internal multimodal embedding layers. The simpler caching approach (tensors and prompt tokenization) was prioritized for safety and compatibility.
- **Distributed/Multi-GPU Batching**: The current harness evaluates sequentially (batch size 1) to ensure precision and compatibility with variable-length outputs. Distributed evaluation (via PyTorch DDP or HuggingFace Accelerate) is deferred.

---

## Project Structure

```
.
├── evaluation/
│   ├── config.py                 # Evaluation parameters and settings
│   ├── model_wrapper.py          # Wrapper for MQT-LLaVA loading and inference
│   ├── load_vqav2.py             # Dataset inspector script
│   ├── smoke_test.py             # Sanity check for token sweeps
│   ├── verify_pipeline.py        # CPU-only mock verification pipeline
│   └── multi_scale_harness.py    # Main evaluation harness
├── setup/
│   └── runpod_setup.sh           # RunPod dependency setup script
├── visualization/
│   ├── variance_plots.py         # Histograms & variance markdown summaries
│   ├── reliability_diagram.py    # Calibration reliability plots (8 panels)
│   └── ece_summary.py            # ECE comparison charts
├── .gitignore                    # Local environment and cache exclusions
└── README.md                     # Project documentation
```

---

## Local / CPU-only Verification (No GPU Required)

If you do not have a GPU locally or do not want to download the multi-gigabyte models and VQAv2 dataset, you can run the mock-based verification pipeline to verify that the evaluation harness, statistics compilation, and plotting tools execute correctly:

1. **Create and Activate a Local Virtual Environment**:
   ```bash
   python -m venv venv_local
   # On Windows:
   venv_local\Scripts\activate.bat
   # On macOS/Linux:
   source venv_local/bin/activate
   ```

2. **Install Lightweight Dependencies**:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install datasets sentence-transformers matplotlib pandas seaborn tqdm Pillow
   ```

3. **Run the Verification Script**:
   ```bash
   python -m evaluation.verify_pipeline
   ```

This mock pipeline executes the harness over synthetic samples, compiles summary statistics, and runs all 3 visualization generators. The generated test outputs and charts will be saved under the `test_run_output/` directory for inspection.

---

## How to Use in RunPod

To run this pipeline on a RunPod instance (e.g. RTX A6000 with the **PyTorch 2.8 + CUDA 12.8** template):

### 1. Initialize the VM Environment
Navigate to your `/workspace` volume and pull the latest changes, then run the setup script to provision paths and build library wrappers:
```bash
cd /workspace
git pull origin main
bash setup/runpod_setup.sh
```

### 2. Activate the Virtual Environment
```bash
source /workspace/venv/bin/activate
```

### 3. Run the Smoke Test
Verify that the model wrapper, tokenizer, and confidence extraction are functional on the GPU:
```bash
python -m evaluation.smoke_test
```

### 4. Launch the Evaluation Harness
You can run a quick check on a subset, or launch the full long-running evaluation loop in the background:

* **Foreground subset run (e.g., 500 samples)**:
  ```bash
  python -m evaluation.multi_scale_harness --subset-size 500
  ```

* **Background full evaluation run (recommended)**:
  Since the harness automatically logs timestamps and milestones to `logs/evaluation.log`, you can run it in the background to ensure it continues even if your terminal session disconnects:
  ```bash
  nohup python -m evaluation.multi_scale_harness > logs/terminal.log 2>&1 &
  ```

* **Monitor run status**:
  Watch high-level milestones:
  ```bash
  tail -f /workspace/vlm-calibration/logs/evaluation.log
  ```
  Watch detailed terminal outputs:
  ```bash
  tail -f /workspace/vlm-calibration/logs/terminal.log
  ```

*Note: The harness supports automated checkpointing and recovery. If interrupted, simply relaunching the harness command will detect existing records in `multi_scale_results.jsonl` and resume progress.*

### 5. Generate Calibration and Stability Plots
Compile results to generate diagrams, charts, and markdown summaries:
```bash
# Generate variance distribution and markdown galleries
python -m visualization.variance_plots

# Generate reliability diagrams with ECE metrics
python -m visualization.reliability_diagram

# Create ECE bar chart summary
python -m visualization.ece_summary
```
Generated charts and comparison tables will be saved under `/workspace/vlm-calibration/plots/`.


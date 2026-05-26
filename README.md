# VLM Calibration: Matryoshka Visual Tokens & Confidence Estimation

This repository contains the evaluation harness and visualization pipeline for investigating uncertainty calibration in Vision-Language Models (VLMs), specifically focusing on elastic visual token configurations using the **Matryoshka Query Transformer LLaVA (MQT-LLaVA)**.

---

## What Has Been Done

### 1. Multi-Scale Evaluation Loop & Setup
* **Environment Provisioning**: Created `setup/runpod_setup.sh` to fully configure dependencies (lmms-eval, mqt-llava, sentence-transformers, etc.) inside RunPod's persistent volume.
- **Harness Implementation**: Built `evaluation/multi_scale_harness.py` to sweep across visual token depths ($m \in [2, 4, 8, 16, 36, 64, 144, 256]$) per sample.
- **Backwards Compatibility & Checkpointing**: Added robust checkpointing (`results/multi_scale_results.jsonl`) to support resuming interrupted sweeps. Added support for unlabeled splits (like test splits) where ground-truth answers are missing, automatically logging accuracies as `None` to prevent runtime crashes.
- **Configurability**: Structured settings (precision, dataset splits, sweeps, generation temp) within `evaluation/config.py`.

### 2. PyTorch & RunPod Optimizations
* **Image & Token Caching (`sweep_optimized`)**: Reduced image preprocessing (PIL -> Tensor conversion) and prompt tokenization frequency from 8 times per sample to **exactly once**.
* **Mixed Precision (`torch.amp.autocast`)**: Implemented half-precision contexts for faster Tensor Core execution on NVIDIA GPUs.
* **Resampler Compilation (`torch.compile`)**: Applied JIT compilation with `dynamic=True` and `mode="reduce-overhead"` to MQT-LLaVA's query abstractor module.
* **VRAM and CPU Pinning**: Pinned the sentence embeddings model (`SentenceTransformer`) to CPU to conserve GPU VRAM for the 7B LLM. Optimized dataloader threads (`num_workers=8`, `pin_memory=True`, `prefetch_factor=2`) to fit RunPod's 16 vCPU setup.

### 3. Visualizations & Analytical Tools
- **Variance Distribution**: Plots answer stability distribution histograms and tabulates the highest/lowest variance question-answer galleries in Markdown (`visualization/variance_plots.py`).
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

## How to Use in RunPod

To run this pipeline on a RunPod instance (e.g. RTX A6000 with the **PyTorch 2.8 + CUDA 12.8** template):

### 1. Initialize the VM Environment
Run the setup script to clone models, configure paths, and build editable library wrappers:
```bash
bash setup/runpod_setup.sh
```

### 2. Activate the Virtual Environment
```bash
source /workspace/venv/bin/activate
```

### 3. Run the Smoke Test
Verify that the model wrapper, tokenizer, and confidence extraction are functional:
```bash
python -m evaluation.smoke_test
```

### 4. Launch the Evaluation Harness
Run the harness on a small subset (e.g., 500 samples) to verify runtime performance before running on the full 1.1M dataset:
```bash
python -m evaluation.multi_scale_harness --subset-size 500
```

### 5. Generate Calibration and Stability Plots
Compile the JSONL results to generate charts and markdown comparison tables:
```bash
# Generate variance distribution and markdown galleries
python -m visualization.variance_plots

# Generate reliability diagrams with ECE metrics
python -m visualization.reliability_diagram

# Create ECE bar chart summary
python -m visualization.ece_summary
```
Generated charts and tables will be located under `/workspace/vlm-calibration/plots/`.

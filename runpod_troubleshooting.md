# RunPod Deployment & Dependency Troubleshooting Report

This document compiles the issues faced while setting up and running the VLM Calibration evaluation harness on the **RunPod PyTorch 2.8.0** template, the local workarounds applied to resolve them on the pod, and the corresponding changes pushed to the remote repository.

---

## 1. PyTorch Version & Environment Conflict
* **Issue**: The `mqt-llava` repository had strict requirements pinned to `torch==2.1.2` and `torchvision==0.16.2` in `pyproject.toml`. PyTorch 2.1.2 wheels do not exist for Python 3.12 (the default on this RunPod template), and trying to build or compile it caused the pip installation to crash.
* **Impact**: Setup aborted during `mqt-llava` installation.
* **Fix on Pod**: 
  1. Recreated the virtual environment with `--system-site-packages` so that the venv inherited the pre-compiled, GPU-optimized `torch 2.8.0` pre-installed in the RunPod image.
  2. Used `sed` to replace `"torch==2.1.2"` and `"torchvision==0.16.2"` with relaxed versions (`torch>=2.1.2` and `torchvision>=0.16.2`) in the cloned `mqt-llava` `pyproject.toml` and `requirements.txt`.
* **Repository Changes**: Updated `setup/runpod_setup.sh` to automatically use `--system-site-packages` and dynamically relax these pins during setup.

---

## 2. CUDA Diagnostic Typo (`props.total_mem`)
* **Issue**: The setup script's final diagnostic verification failed with `AttributeError: 'torch._C._CudaDeviceProperties' object has no attribute 'total_mem'`.
* **Impact**: Aesthetic failure at the very end of the setup script.
* **Fix on Pod**: Bypassed, as the actual package installations had already completed successfully.
* **Repository Changes**: Corrected the diagnostic code in `setup/runpod_setup.sh` to access `props.total_memory` (the correct property name).

---

## 3. NumPy 2.x Binary Incompatibility
* **Issue**: `lmms-eval` pulled in **NumPy 2.x** (`numpy-2.4.6`). This caused binary incompatibility crashes (`ValueError: numpy.dtype size changed, may indicate binary incompatibility`) in compiled C-extensions like `scikit-learn` when importing `transformers` inside the smoke test.
* **Impact**: Visual model loading aborted on startup.
* **Fix on Pod**: Force-downgraded NumPy to the stable 1.x series inside the virtual environment:
  ```bash
  pip install "numpy<2.0"
  ```
* **Repository Changes**: Added `"numpy<2.0"` to the additional dependencies installation block in `setup/runpod_setup.sh`.

---

## 4. Relocated `apply_chunking_to_forward` Import
* **Issue**: Hugging Face `transformers` relocated the helper function `apply_chunking_to_forward` from `transformers.modeling_utils` to `transformers.pytorch_utils` in newer v4.x releases.
* **Impact**: `ImportError` when importing `mqt-llava` projector builder.
* **Fix on Pod**: Patched the import statement in `/workspace/mqt-llava/llava/model/multimodal_projector/builder.py` to use a `try...except ImportError` fallback.
* **Repository Changes**: Added a Python regex-patching step to `setup/runpod_setup.sh` to apply this import fallback right after cloning `mqt-llava`.

---

## 5. Removed Attention Pruning Helpers in Transformers v5.x
* **Issue**: The environment pulled in `transformers` v5.x by default, which completely removed internal helper functions `find_pruneable_heads_and_indices` and `prune_linear_layer`.
* **Impact**: `ImportError` on startup as `builder.py` could no longer load these functions from any module.
* **Fix on Pod**: 
  1. Downgraded `transformers`, `tokenizers`, and `accelerate` to v4.x versions:
     ```bash
     pip install "transformers<5.0.0" "tokenizers<0.20.0" "accelerate<1.0.0"
     ```
  2. Added import fallback patching inside `/workspace/mqt-llava/llava/model/multimodal_projector/builder.py` to try importing them from `transformers.modeling_utils` first, with a fallback to `transformers.pytorch_utils`.
* **Repository Changes**: Added pins (`"transformers<5.0.0"`, `"tokenizers<0.20.0"`, and `"accelerate<1.0.0"`) to `setup/runpod_setup.sh`, along with the corresponding auto-patch logic for the pruning imports.

---

## 6. Model Forward Signature Clash (`cache_position` in LLaVA)
* **Issue**: In `transformers` v4.39.0+, `.generate()` passes `cache_position` to the model's `forward()` method. Since `LlavaLlamaForCausalLM` overrides `forward()` but did not accept `**kwargs` or `cache_position` in its signature, it threw a `TypeError`.
* **Impact**: Generation crashed on the very first forward pass during the token sweep.
* **Fix on Pod**: Patched `/workspace/mqt-llava/llava/model/language_model/llava_llama.py` to add `**kwargs` to the `forward()` signature and pass them down to `super().forward(**kwargs)`.
* **Repository Changes**: Added Python search-and-replace patch logic for `llava_llama.py` to `setup/runpod_setup.sh`.

---

## 7. TorchCodec Binary Incompatibility
* **Issue**: `sentence-transformers` loaded `torchcodec==0.13.0` which was binary incompatible with the pod's `torch 2.8.0` library, throwing `undefined symbol: torch_from_blob` and crashing the import of `SentenceTransformer`.
* **Impact**: Evaluation harness crashed during startup.
* **Fix on Pod**: Installed the version of `torchcodec` that matches the PyTorch 2.8.0 API structure:
  ```bash
  pip install "torchcodec==0.7.0"
  ```
* **Repository Changes**: Added `"torchcodec==0.7.0"` to the additional dependencies section in `setup/runpod_setup.sh`.

#!/bin/bash
# =============================================================================
# RunPod Environment Setup — Week 3: VLM Calibration Project
# Target: RunPod PyTorch 2.8.0, 1x RTX A6000 (48GB VRAM), 62GB RAM, 16 vCPU
# =============================================================================

set -euo pipefail

echo "=========================================="
echo " VLM Calibration — RunPod Setup"
echo "=========================================="

# ---------------------------------------------------------------------------
# 1. Work in the persistent /workspace volume
# ---------------------------------------------------------------------------
cd /workspace

# ---------------------------------------------------------------------------
# 2. Create and activate virtual environment
# ---------------------------------------------------------------------------
if [ ! -d "venv" ]; then
    echo "[1/7] Creating virtual environment..."
    python3 -m venv venv --system-site-packages
else
    echo "[1/7] Virtual environment already exists, reusing."
fi
source venv/bin/activate
pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# 3. Clone and install MQT-LLaVA
# ---------------------------------------------------------------------------
if [ ! -d "mqt-llava" ]; then
    echo "[2/7] Cloning MQT-LLaVA..."
    git clone https://github.com/gordonhu608/mqt-llava.git
else
    echo "[2/7] MQT-LLaVA already cloned, pulling latest..."
    cd mqt-llava && git pull && cd ..
fi
cd mqt-llava
# Relax strict torch/torchvision dependency pins to allow using the RunPod preinstalled PyTorch
if [ -f "pyproject.toml" ]; then
    echo "Relaxing torch & torchvision constraints in pyproject.toml..."
    sed -i 's/"torch==2.1.2"/"torch>=2.1.2"/g' pyproject.toml
    sed -i 's/"torchvision==0.16.2"/"torchvision>=0.16.2"/g' pyproject.toml
fi
if [ -f "requirements.txt" ]; then
    echo "Relaxing torch & torchvision constraints in requirements.txt..."
    sed -i 's/torch==2.1.2/torch>=2.1.2/g' requirements.txt
    sed -i 's/torchvision==0.16.2/torchvision>=0.16.2/g' requirements.txt
fi
# Patch transformers import compatibility for older/newer version transitions
if [ -f "llava/model/multimodal_projector/builder.py" ]; then
    echo "Patching apply_chunking_to_forward import in builder.py..."
    python3 -c '
path = "llava/model/multimodal_projector/builder.py"
with open(path, "r") as f:
    text = f.read()
if "apply_chunking_to_forward" in text:
    text = text.replace("apply_chunking_to_forward,", "")
    patch = "try:\n    from transformers.modeling_utils import apply_chunking_to_forward\nexcept ImportError:\n    from transformers.pytorch_utils import apply_chunking_to_forward\n"
    text = patch + text
    with open(path, "w") as f:
        f.write(text)
    print("Patch applied successfully!")
'
fi
pip install -e .
cd /workspace

# ---------------------------------------------------------------------------
# 4. Clone and install lmms-eval
# ---------------------------------------------------------------------------
if [ ! -d "lmms-eval" ]; then
    echo "[3/7] Cloning lmms-eval..."
    git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
else
    echo "[3/7] lmms-eval already cloned, pulling latest..."
    cd lmms-eval && git pull && cd ..
fi
cd lmms-eval
pip install -e ".[all]"
cd /workspace

# ---------------------------------------------------------------------------
# 5. Install additional Python dependencies
# ---------------------------------------------------------------------------
echo "[4/7] Installing additional dependencies..."
pip install \
    "numpy<2.0" \
    sentence-transformers \
    matplotlib \
    seaborn \
    pandas \
    scipy \
    tqdm \
    jsonlines \
    Pillow

# ---------------------------------------------------------------------------
# 6. Create project directory structure
# ---------------------------------------------------------------------------
echo "[5/7] Creating project directories..."
mkdir -p /workspace/vlm-calibration/{results,plots,logs,configs,checkpoints}

# ---------------------------------------------------------------------------
# 7. Copy project scripts into workspace
# ---------------------------------------------------------------------------
echo "[6/7] Preparing project scripts..."
# If you've uploaded the project files to /workspace/project-src, copy them:
if [ -d "/workspace/project-src/evaluation" ]; then
    cp -r /workspace/project-src/evaluation /workspace/vlm-calibration/
fi
if [ -d "/workspace/project-src/visualization" ]; then
    cp -r /workspace/project-src/visualization /workspace/vlm-calibration/
fi

# ---------------------------------------------------------------------------
# 8. Verify GPU and environment
# ---------------------------------------------------------------------------
echo "[7/7] Verifying environment..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU device:      {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'VRAM:            {props.total_memory / 1e9:.1f} GB')
    print(f'CUDA version:    {torch.version.cuda}')
else:
    print('WARNING: No GPU detected!')
"
nvidia-smi

echo ""
echo "=========================================="
echo " Setup complete!"
echo " Activate with: source /workspace/venv/bin/activate"
echo " Project dir:   /workspace/vlm-calibration/"
echo "=========================================="

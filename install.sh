#!/bin/bash
# ============================================================================
# Real-AVSE Challenge Baseline — one-shot environment setup.
#
# Creates a conda env, installs PyTorch (CUDA 12.6) + all Python deps needed to
# train / test / evaluate the baseline, then reminds you to fetch pretrained
# weights with download_models.sh.
#
# Usage:
#   bash install.sh                 # create env `real-avse`, install everything
#   ENV_NAME=myenv bash install.sh  # use a different env name
#   GPU_ONNX=1 bash install.sh      # install onnxruntime-gpu instead of CPU build
#
# Requirements: a working conda/mamba, and (for download_models.sh afterwards)
# a machine with outbound internet. The compute cluster itself may be offline —
# run download_models.sh on an internet-connected node.
# ============================================================================

set -euo pipefail

ENV_NAME="${ENV_NAME:-real-avse}"
PY_VERSION="3.11"
# CUDA 12.6 wheels match the cluster driver (535.x) and the reference env.
TORCH_INDEX="https://download.pytorch.org/whl/cu126"
UTMOS_GIT="git+https://github.com/sarulab-speech/UTMOSv2.git"

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJ_DIR"

echo "=================================================="
echo "Real-AVSE baseline setup"
echo "  env name : $ENV_NAME"
echo "  python   : $PY_VERSION"
echo "  project  : $PROJ_DIR"
echo "=================================================="

# ---------------------------------------------------------------------------
# 0. Locate conda and enable `conda activate` inside this non-interactive shell.
# ---------------------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH. Install Miniconda/Anaconda first." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
# Older conda (<=4.10) dereferences $PS1 in its activation hook. Under `set -u`
# an unbound PS1 aborts the script ("PS1: unbound variable"), so default it.
: "${PS1:=}"

# ---------------------------------------------------------------------------
# 1. Create (or reuse) the conda env.
# ---------------------------------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[1/4] conda env '$ENV_NAME' already exists — reusing it."
else
    echo "[1/4] Creating conda env '$ENV_NAME' (python $PY_VERSION) ..."
    conda create -y -n "$ENV_NAME" "python=$PY_VERSION"
fi
conda activate "$ENV_NAME"
python -m pip install --upgrade pip

# ---------------------------------------------------------------------------
# 2. PyTorch + torchaudio + torchvision (CUDA 12.6 wheels).
#    Pinned to the reference env; torchvision is needed by look2hear video nets.
# ---------------------------------------------------------------------------
echo "[2/4] Installing PyTorch 2.7.1 (cu126) ..."
python -m pip install --index-url "$TORCH_INDEX" \
    "torch==2.7.1" "torchaudio==2.7.1" "torchvision==0.22.1"

# ---------------------------------------------------------------------------
# 3. Baseline + evaluation Python dependencies.
# ---------------------------------------------------------------------------
echo "[3/4] Installing project dependencies (requirements.txt) ..."
python -m pip install -r requirements.txt

# onnxruntime: default CPU build is in requirements.txt. For the GPU build
# (faster DNSMOS), set GPU_ONNX=1 — falls back to CPU if the GPU wheel fails.
if [ "${GPU_ONNX:-0}" = "1" ]; then
    echo "      Installing onnxruntime-gpu (GPU_ONNX=1) ..."
    python -m pip uninstall -y onnxruntime >/dev/null 2>&1 || true
    python -m pip install "onnxruntime-gpu" || {
        echo "      onnxruntime-gpu failed; keeping CPU onnxruntime."
        python -m pip install "onnxruntime"
    }
fi

# UTMOSv2 (no-reference MOS) ships only from GitHub, not PyPI.
echo "      Installing UTMOSv2 from git ..."
python -m pip install "$UTMOS_GIT"

# ---------------------------------------------------------------------------
# 4. Sanity check: import the core stack.
# ---------------------------------------------------------------------------
echo "[4/4] Verifying imports ..."
python - <<'PYEOF'
import importlib
mods = ["torch", "torchaudio", "torchvision", "pytorch_lightning",
        "soundfile", "librosa", "pesq", "pystoi", "torchmetrics",
        "albumentations", "skimage", "cv2", "swanlab",
        "funasr", "modelscope", "transformers", "onnxruntime",
        "jiwer", "utmosv2"]
import torch
print(f"  torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append((m, str(e).splitlines()[0]))
if missing:
    print("  MISSING / FAILED imports:")
    for m, err in missing:
        print(f"    - {m}: {err}")
    raise SystemExit(1)
print("  All core imports OK.")
PYEOF

echo
echo "=================================================="
echo "Environment '$ENV_NAME' is ready."
echo
echo "Next steps:"
echo "  conda activate $ENV_NAME"
echo "  # Fetch pretrained weights (run on an internet-connected machine):"
echo "  bash download_models.sh"
echo "  # Then train / test / evaluate:"
echo "  bash run_train.sh   |   bash run_test.sh   |   bash run_eval_real.sh"
echo "=================================================="

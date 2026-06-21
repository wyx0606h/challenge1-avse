#!/bin/bash
# Download all pretrained weights needed by the real-test evaluation (eval_real.py).
#
# IMPORTANT: the compute cluster has NO outbound internet (huggingface.co /
# github time out). Run THIS script on a node/machine that DOES have internet
# (or with a proxy / HF mirror configured). It writes weights to fixed local
# cache locations that eval_real.py then reads fully offline.
#
# Covered models:
#   - UTMOSv2          (no-reference MOS)       -> ~/.cache/utmosv2 + HF wav2vec2-base
#   - DNSMOS           (no-reference P.835)     -> ~/.torchmetrics/DNSMOS  (3 onnx files)
#   - Fun-ASR-Nano     (Chinese ASR for WER/CER)-> ~/.cache/modelscope/hub
#   - WeSpeaker ResNet34-LM (speaker embedding) -> pretrained/wespeaker_cnceleb_resnet34
#
# Usage:
#   bash download_models.sh
#   # optional HF mirror (if huggingface.co is blocked but a mirror is reachable):
#   HF_ENDPOINT=https://hf-mirror.com bash download_models.sh

set -e

# Use the currently-activated env's python (set up by install.sh). Run
# `conda activate real-avse` first, or override with PY=/path/to/python.
PY="${PY:-python}"
PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=================================================="
echo "Real-AVSE evaluation: downloading pretrained models"
echo "Python : $PY"
echo "Project: $PROJ_DIR"
echo "=================================================="

# ---------------------------------------------------------------------------
# 0. Python dependencies are installed by install.sh into the conda env.
#    Run `bash install.sh` and `conda activate real-avse` before this script.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 1. UTMOSv2 weights.
#    create_model(pretrained=True) downloads fold0_s42_best_model.pth into
#    ~/.cache/utmosv2/models/fusion_stage3/ and pulls the SSL backbone
#    facebook/wav2vec2-base from HF. Running it once here primes both caches.
# ---------------------------------------------------------------------------
echo
echo "[1/4] UTMOSv2 (MOS) weights + wav2vec2-base backbone ..."
$PY - <<'PYEOF'
import utmosv2
# Triggers HF download of fold0_s42_best_model.pth and the wav2vec2-base backbone.
model = utmosv2.create_model(pretrained=True)
print("  UTMOSv2 ready. cache: ~/.cache/utmosv2")
PYEOF

# ---------------------------------------------------------------------------
# 2. DNSMOS onnx models.
#    torchmetrics downloads 3 onnx files from microsoft/DNS-Challenge into
#    ~/.torchmetrics/DNSMOS the first time DNSMOS runs. Prime it with a dummy
#    1s signal so eval_real.py never needs the network.
# ---------------------------------------------------------------------------
echo
echo "[2/4] DNSMOS onnx models -> ~/.torchmetrics/DNSMOS ..."
$PY - <<'PYEOF'
import torch
from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score
dummy = torch.randn(16000)  # 1s @ 16kHz
score = deep_noise_suppression_mean_opinion_score(dummy, fs=16000, personalized=False)
print("  DNSMOS ready (p808, sig, bak, ovr):", [round(float(x), 3) for x in score])
PYEOF

# ---------------------------------------------------------------------------
# 3. Fun-ASR-Nano (Chinese ASR for WER/CER).
#    Pulls FunAudioLLM/Fun-ASR-Nano-2512 from ModelScope into
#    ~/.cache/modelscope/hub. trust_remote_code uses Fun-ASR/model.py (already
#    cloned locally). Also pulls the fsmn-vad model used for long-form decoding.
# ---------------------------------------------------------------------------
echo
echo "[3/4] Fun-ASR-Nano-2512 ASR weights -> ~/.cache/modelscope ..."
$PY - <<PYEOF
import torch
from funasr import AutoModel

model = AutoModel(
    model="FunAudioLLM/Fun-ASR-Nano-2512",
    trust_remote_code=True,
    remote_code="${PROJ_DIR}/Fun-ASR/model.py",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    hub="ms",
)
print("  Fun-ASR-Nano ready. model_path:", model.model_path)
PYEOF

# ---------------------------------------------------------------------------
# 4. WeSpeaker cnceleb-resnet34-LM speaker-embedding weights.
#    The repo ships model_5.pt (raw torch state_dict) + config.yaml. We only
#    need model_5.pt; the ResNet34 architecture is reproduced in
#    look2hear/models/wespeaker_resnet34.py (no wespeaker install required).
# ---------------------------------------------------------------------------
echo
echo "[4/4] WeSpeaker cnceleb-resnet34-LM -> pretrained/wespeaker_cnceleb_resnet34 ..."
WESPK_DIR="${PROJ_DIR}/pretrained/wespeaker_cnceleb_resnet34"
mkdir -p "$WESPK_DIR"
$PY - <<PYEOF
from huggingface_hub import hf_hub_download
import shutil, os

repo = "Wespeaker/wespeaker-cnceleb-resnet34-LM"
dst = "${WESPK_DIR}"
# model_5.pt is the torch checkpoint (eval_real.py default); config.yaml
# documents the architecture (ResNet34, feat_dim=80, embed_dim=256, TSTP).
for fname in ["model_5.pt", "config.yaml"]:
    try:
        p = hf_hub_download(repo_id=repo, filename=fname)
        shutil.copy(p, os.path.join(dst, fname))
        print(f"  downloaded {fname}")
    except Exception as e:
        print(f"  WARN: could not fetch {fname}: {e}")
print("  WeSpeaker weights in:", dst)
PYEOF

echo
echo "=================================================="
echo "Done. Downloaded artifacts:"
echo "  UTMOSv2     : ~/.cache/utmosv2/models/fusion_stage3/fold0_s42_best_model.pth"
echo "  DNSMOS      : ~/.torchmetrics/DNSMOS/*.onnx"
echo "  Fun-ASR-Nano: ~/.cache/modelscope/hub/models/FunAudioLLM/Fun-ASR-Nano-2512"
echo "  WeSpeaker   : ${PROJ_DIR}/pretrained/wespeaker_cnceleb_resnet34/model_5.pt"
echo "=================================================="

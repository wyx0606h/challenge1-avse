#!/usr/bin/env bash
# Minimal Track 2 inference smoke test.
#
# Required:
#   DATA_ROOT=/path/to/Real-World-AVSE
#
# Optional:
#   GPU=0
#   CKPT=JusperLee/Real-World-AVSE-Baseline-Track2
#   PYTHON=python
#   SAVE_DIR=enhanced_out/smoke-track2

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

: "${DATA_ROOT:?Set DATA_ROOT to the directory containing track2/}"

GPU="${GPU:-0}"
CKPT="${CKPT:-JusperLee/Real-World-AVSE-Baseline-Track2}"
PYTHON="${PYTHON:-python}"
SAVE_DIR="${SAVE_DIR:-enhanced_out/smoke-track2}"

"$PYTHON" tools/check_track2_setup.py --data-root "$DATA_ROOT"

echo "Running one Track 2 dev clip per scene with metrics disabled."
echo "Checkpoint: $CKPT"
echo "Data root:  $DATA_ROOT"
echo "GPU:        $GPU"
echo "Output:     $SAVE_DIR"

"$PYTHON" eval_real.py \
  --conf_dir configs/track2_av_convtasnet.yml \
  --ckpt "$CKPT" \
  --data_root "$DATA_ROOT" \
  --track track2 \
  --scene both \
  --split dev \
  --metrics none \
  --mode enhance \
  --save_dir "$SAVE_DIR" \
  --gpus "$GPU" \
  --limit 1

echo "Track 2 smoke test completed."

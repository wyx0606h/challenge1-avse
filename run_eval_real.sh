#!/bin/bash
# Run the real-world AVSE evaluation fully offline, with optional multi-GPU
# sharding. Each GPU runs one shard of the item list as its own process; when
# all shards finish, their per-shard CSVs are merged into the final tables.
#
# Prereq: run download_models.sh once on an internet-connected machine so the
# UTMOS / DNSMOS / Fun-ASR-Nano / WeSpeaker weights are cached locally.
#
# Usage:
#   bash run_eval_real.sh [CKPT] [TRACK] [SCENE] [METRICS] [GPUS] [LIMIT] [MODE]
#   TRACK   : track1 | track2 | both
#   SCENE   : mix (real, no GT) | remix (synthetic, has GT) | both
#   METRICS : comma list of {utmos,dnsmos,asr,spk,objective} | all | none
#   GPUS    : comma-separated GPU ids, e.g. "0" or "0,1,2,3" (one shard each)
#   MODE    : both (enhance+score) | enhance (save wavs only) | eval (score saved)
#
#   Env-var overrides (keep the positional args short):
#     SPLIT       : dev (default) | test          -- which split to evaluate
#     SAVE_DIR    : enhanced_out (default)         -- where enhanced wavs go
#     ENROLL_CKPT : path to precomputed voiceprints (.pt) for spk-sim; if unset,
#                   they are rebuilt from the split's clean remix sources
#     DATA_ROOT   : REAL-AVSE corpus root override
# Examples:
#   bash run_eval_real.sh                                  # dev, latest ckpt, 1 GPU, all
#   bash run_eval_real.sh "" both both all 0,1,2,3        # dev, 4-GPU sharded eval
#   SPLIT=test bash run_eval_real.sh "" both both all 0,1 # organizer test eval
#   ENROLL_CKPT=pretrained/enroll_dev.pt bash run_eval_real.sh "" both both spk 0
#   bash run_eval_real.sh "" both both none 0 5           # 5-clip smoke (pipeline only)

# Activate the same conda env that install.sh created (override with ENV_NAME=...).
ENV_NAME="${ENV_NAME:-real-avse}"
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH. Run install.sh first." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
: "${PS1:=}"   # older conda (<=4.10) dereferences $PS1 in its activate hook
conda activate "$ENV_NAME"
PYTHON_ENV="python"

CKPT=${1:-""}
TRACK=${2:-"both"}
SCENE=${3:-"both"}
METRICS=${4:-"all"}
GPUS=${5:-"0"}
LIMIT=${6:-""}
MODE=${7:-"eval"}
SPLIT=${SPLIT:-"dev"}                   # default to dev (participant-facing split)
SAVE_DIR=${SAVE_DIR:-"enhanced_out"}    # override via env: SAVE_DIR=... bash run_eval_real.sh
ENROLL_CKPT=${ENROLL_CKPT:-""}          # precomputed voiceprints; empty => rebuild
DATA_ROOT=${DATA_ROOT:-""}

# Force offline: weights were pre-fetched by download_models.sh.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MODELSCOPE_OFFLINE=1

# Split the GPU list into an array; number of shards = number of GPUs.
IFS=',' read -ra GPU_ARR <<< "$GPUS"
NUM_SHARDS=${#GPU_ARR[@]}

# Common args shared by every shard.
COMMON="--track $TRACK --scene $SCENE --metrics $METRICS --mode $MODE --split $SPLIT --save_dir $SAVE_DIR"
[ -n "$CKPT" ]        && COMMON="$COMMON --ckpt $CKPT"
[ -n "$LIMIT" ]       && COMMON="$COMMON --limit $LIMIT"
[ -n "$ENROLL_CKPT" ] && COMMON="$COMMON --enroll_ckpt $ENROLL_CKPT"
[ -n "$DATA_ROOT" ]   && COMMON="$COMMON --data_root $DATA_ROOT"

echo "=================================================="
echo "Real-world AVSE evaluation"
echo "  ckpt    : ${CKPT:-<latest epoch=*.ckpt>}"
echo "  track   : $TRACK   scene: $SCENE   metrics: $METRICS"
echo "  split   : $SPLIT   mode: $MODE   save_dir: $SAVE_DIR"
echo "  enroll  : ${ENROLL_CKPT:-<rebuild from clean sources>}"
echo "  GPUs    : $GPUS   (shards: $NUM_SHARDS, offline mode)"
echo "=================================================="

if [ "$NUM_SHARDS" -le 1 ]; then
    # Single GPU: one process, no sharding suffix.
    $PYTHON_ENV eval_real.py $COMMON --gpus "${GPU_ARR[0]}"
    echo "Evaluation finished!"
    exit 0
fi

# Multi-GPU: launch one sharded process per GPU in the background.
# Each shard runs in its own session/process group (setsid) so that on Ctrl+C
# we can kill the whole subtree -- python plus its DataLoader/CUDA workers --
# instead of leaving orphans holding the GPUs.
PIDS=()

cleanup() {
    trap '' INT TERM            # ignore repeated signals while we tear down
    echo
    echo "Interrupted -- terminating all shards..."
    for pid in "${PIDS[@]}"; do
        kill -TERM -"$pid" 2>/dev/null   # negative pid => whole process group
    done
    sleep 3
    for pid in "${PIDS[@]}"; do
        kill -KILL -"$pid" 2>/dev/null
    done
    exit 130
}
trap cleanup INT TERM

for i in "${!GPU_ARR[@]}"; do
    gpu="${GPU_ARR[$i]}"
    echo "  -> shard $i on GPU $gpu"
    setsid $PYTHON_ENV eval_real.py $COMMON --gpus "$gpu" \
        --num_shards "$NUM_SHARDS" --shard_id "$i" &
    PIDS+=($!)
done

# Wait for all shards; abort the merge if any shard failed.
FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=1
done
if [ "$FAIL" -ne 0 ]; then
    echo "ERROR: at least one shard failed; skipping merge."
    exit 1
fi

# Merge per-shard CSVs into the final total/per-group/summary tables.
if [ "$MODE" != "enhance" ]; then
    echo "Merging $NUM_SHARDS shards..."
    $PYTHON_ENV eval_real.py $COMMON --merge_shards --num_shards "$NUM_SHARDS"
fi
echo "Evaluation finished!"

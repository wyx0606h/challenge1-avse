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
#     CONF_DIR    : model/data config override; inferred from TRACK when unset
# Examples:
#   # Score already-enhanced Track 2 dev wavs (default MODE=eval):
#   bash run_eval_real.sh "" track2 both all 0,1,2,3
#   # Enhance + score with an already-cached gated HF checkpoint:
#   bash run_eval_real.sh JusperLee/Real-World-AVSE-Baseline-Track2 \
#     track2 both all 0,1,2,3 "" both
#   # Model-only limited run (checkpoint must already be cached):
#   bash run_eval_real.sh JusperLee/Real-World-AVSE-Baseline-Track2 \
#     track2 both none 0 5 enhance

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
CONF_DIR=${CONF_DIR:-""}

# eval_real.py always reads a YAML for sample rate / face size and uses it to
# rebuild local Lightning checkpoints. A clean checkout does not contain the
# script's historical Experiments/... default, so select a tracked config here.
if [ -z "$CONF_DIR" ]; then
    if [ "$TRACK" = "track2" ]; then
        CONF_DIR="configs/track2_av_convtasnet.yml"
    else
        # Track 1 and "both" share the same AV-ConvTasNet architecture. Runs
        # using a released HF checkpoint rebuild the architecture from its own
        # config.json; this YAML still supplies sample rate and face size.
        CONF_DIR="configs/track1_av_convtasnet.yml"
    fi
fi

# Force offline: weights were pre-fetched by download_models.sh.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MODELSCOPE_OFFLINE=1

# Split the GPU list into an array; number of shards = number of GPUs.
IFS=',' read -ra GPU_ARR <<< "$GPUS"
NUM_SHARDS=${#GPU_ARR[@]}

# Cap CPU math-library threads to avoid oversubscription. The metric models
# (UTMOS especially) fire many small CPU ops; with the BLAS/OpenMP defaults a
# single process spawns one thread per logical core (40 here), and threading
# overhead then DOMINATES -- measured UTMOS was 3.6 s/clip at 40 threads vs
# 1.0 s/clip at 4. Under sharding it is worse: N shards x 40 threads all fight
# over the same cores. So budget total threads ~= core count and split across
# shards (floor, min 1). Override by exporting THREADS_PER_SHARD yourself.
if [ -z "$THREADS_PER_SHARD" ]; then
    NCORES=$(python -c "import os; print(os.cpu_count() or 8)" 2>/dev/null || echo 8)
    THREADS_PER_SHARD=$(( NCORES / NUM_SHARDS ))
    [ "$THREADS_PER_SHARD" -lt 1 ] && THREADS_PER_SHARD=1
    # A single process does best around 4 threads, not all cores (see above).
    [ "$NUM_SHARDS" -le 1 ] && [ "$THREADS_PER_SHARD" -gt 4 ] && THREADS_PER_SHARD=4
fi
export OMP_NUM_THREADS="$THREADS_PER_SHARD"
export MKL_NUM_THREADS="$THREADS_PER_SHARD"
export OPENBLAS_NUM_THREADS="$THREADS_PER_SHARD"
export NUMEXPR_NUM_THREADS="$THREADS_PER_SHARD"
export VECLIB_MAXIMUM_THREADS="$THREADS_PER_SHARD"
export TOKENIZERS_PARALLELISM=false   # HF tokenizers fork-safety + no thread storm

# Common args shared by every shard.
COMMON="--conf_dir $CONF_DIR --track $TRACK --scene $SCENE --metrics $METRICS --mode $MODE --split $SPLIT --save_dir $SAVE_DIR"
[ -n "$CKPT" ]        && COMMON="$COMMON --ckpt $CKPT"
[ -n "$LIMIT" ]       && COMMON="$COMMON --limit $LIMIT"
[ -n "$ENROLL_CKPT" ] && COMMON="$COMMON --enroll_ckpt $ENROLL_CKPT"
[ -n "$DATA_ROOT" ]   && COMMON="$COMMON --data_root $DATA_ROOT"

echo "=================================================="
echo "Real-world AVSE evaluation"
echo "  ckpt    : ${CKPT:-<latest epoch=*.ckpt>}"
echo "  track   : $TRACK   scene: $SCENE   metrics: $METRICS"
echo "  split   : $SPLIT   mode: $MODE   save_dir: $SAVE_DIR"
echo "  config  : $CONF_DIR"
echo "  enroll  : ${ENROLL_CKPT:-<rebuild from clean sources>}"
echo "  GPUs    : $GPUS   (shards: $NUM_SHARDS, offline mode)"
echo "  threads : $THREADS_PER_SHARD per shard (OMP/MKL/BLAS capped)"
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

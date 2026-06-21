#!/bin/bash
# Testing script with automatic environment setup

# Set Python environment
export PYTHON_ENV="/home/likai/data/anaconda3/envs/klyro/bin/python"

# Configuration file
CONF_DIR=${1:-"Experiments/checkpoint/AVConvTasNet-Baseline/conf.yml"}
CHECKPOINT=${2:-""}
SAVE_DIR=${3:-""}
GPU=${4:-"0"}

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$GPU

# Build command
CMD="$PYTHON_ENV test.py --conf_dir $CONF_DIR --gpus $GPU"

if [ ! -z "$CHECKPOINT" ]; then
    CMD="$CMD --checkpoint $CHECKPOINT"
fi

if [ ! -z "$SAVE_DIR" ]; then
    CMD="$CMD --save_dir $SAVE_DIR"
fi

# Run testing
echo "Starting testing with config: $CONF_DIR"
echo "Using GPU: $GPU"
echo "Python: $PYTHON_ENV"

$CMD

echo "Testing finished!"

#!/bin/bash
# Training script with automatic environment setup

# Set Python environment
export PYTHON_ENV="/home/likai/data/anaconda3/envs/klyro/bin/python"

# Configuration file
CONFIG=${1:-"configs/av_convtasnet.yml"}

# Check if config exists
if [ ! -f "$CONFIG" ]; then
    echo "Error: Configuration file $CONFIG not found!"
    exit 1
fi

# Set CUDA devices (modify as needed)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Run training
echo "Starting training with config: $CONFIG"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "Python: $PYTHON_ENV"

$PYTHON_ENV train.py --conf_dir $CONFIG

echo "Training finished!"

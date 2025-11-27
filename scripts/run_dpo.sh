#!/bin/bash
# Run Direct Preference Optimization (DPO)
#
# DPO trains the model to prefer "chosen" responses over "rejected" responses.
# Run this AFTER SFT training.
#
# Usage:
#   ./scripts/run_dpo.sh --model ./checkpoints/sft/epoch_3
#   ./scripts/run_dpo.sh --model gpt2  # Or start from base model

set -e

cd "$(dirname "$0")/.."

echo "============================================"
echo "Direct Preference Optimization (DPO)"
echo "============================================"
echo ""

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv
    echo ""
fi

# Run DPO
python run_dpo.py "$@"


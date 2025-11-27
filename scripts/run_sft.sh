#!/bin/bash
# Run Supervised Fine-Tuning (SFT) on GPT-2
#
# This script fine-tunes GPT-2 to follow instructions.
# Run this BEFORE DPO training.
#
# Usage:
#   ./scripts/run_sft.sh              # Default settings
#   ./scripts/run_sft.sh --epochs 5   # Custom epochs

set -e

cd "$(dirname "$0")/.."

echo "============================================"
echo "Supervised Fine-Tuning (SFT)"
echo "============================================"
echo ""

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv
    echo ""
fi

# Run SFT
python run_sft.py "$@"


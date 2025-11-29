#!/bin/bash
#
# BasicGPT Training Setup & Launch Script
#
# This script:
# 1. Installs dependencies
# 2. Detects GPU configuration
# 3. Logs into Weights & Biases
# 4. Tests checkpoint directory setup
# 5. Launches training with Accelerate
#
# Usage:
#   ./start.sh                 # Full setup + training
#   ./start.sh --skip-deps     # Skip dependency installation
#   ./start.sh --dry-run       # Setup only, don't train
#

set -e  # Exit on error

# ============================================
# Load environment variables from .env
# ============================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
    echo "Loading API keys from .env..."
    set -a  # Export all variables
    source "$SCRIPT_DIR/.env"
    set +a
else
    echo "⚠️  No .env file found. Copy .env.example to .env and add your API keys."
    echo "   cp .env.example .env"
fi

# Optional: Set a W&B run ID to resume (get from wandb.ai URL)
# Example: WANDB_RUN_ID="abc123xyz"
# Leave empty to auto-detect from checkpoint or start new run
WANDB_RUN_ID="${WANDB_RUN_ID:-}"

# ============================================
# Parse arguments
# ============================================
SKIP_DEPS=false
DRY_RUN=false

for arg in "$@"; do
    case $arg in
        --skip-deps)
            SKIP_DEPS=true
            ;;
        --dry-run)
            DRY_RUN=true
            ;;
        --wandb-id=*)
            WANDB_RUN_ID="${arg#*=}"
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-deps        Skip dependency installation"
            echo "  --dry-run          Setup only, don't start training"
            echo "  --wandb-id=ID      Resume specific W&B run (e.g., --wandb-id=abc123)"
            echo ""
            exit 0
            ;;
    esac
done

# ============================================
# Functions
# ============================================

print_header() {
    echo ""
    echo "============================================"
    echo "$1"
    echo "============================================"
}

detect_gpus() {
    print_header "Detecting GPU Configuration"
    
    if command -v nvidia-smi &> /dev/null; then
        echo "CUDA detected!"
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
        
        GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
        echo ""
        echo "Total GPUs: $GPU_COUNT"
        
        # Get total and per-GPU VRAM
        TOTAL_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{sum+=$1} END {print sum}')
        PER_GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        echo "Total VRAM: ${TOTAL_VRAM}MB ($(echo "scale=1; $TOTAL_VRAM/1024" | bc)GB)"
        echo "Per-GPU VRAM: ${PER_GPU_VRAM}MB ($(echo "scale=1; $PER_GPU_VRAM/1024" | bc)GB)"
        
        # Set PyTorch memory allocation config to avoid fragmentation
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        echo ""
        echo "✓ Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
        
        # Warn if GPUs have less than 48GB VRAM (may need smaller batch size)
        if [ "$PER_GPU_VRAM" -lt 48000 ]; then
            echo ""
            echo "⚠️  WARNING: GPUs have < 48GB VRAM each"
            echo "   You may need to reduce batch_size in config.py"
            echo "   Recommended: batch_size=16 for 24GB GPUs"
            echo "   Recommended: batch_size=24 for 40GB GPUs"
        fi
    else
        echo "No NVIDIA GPU detected. Training will use CPU."
        GPU_COUNT=0
    fi
}

install_dependencies() {
    print_header "Installing Dependencies"
    
    echo "Installing from requirements.txt..."
    pip install -r requirements.txt
    
    # Install optional CUDA dependencies
    if command -v nvidia-smi &> /dev/null; then
        echo ""
        echo "Installing CUDA-optimized packages..."
        pip install ninja packaging
    fi
    
    echo ""
    echo "✓ Dependencies installed"
}

setup_wandb() {
    print_header "Setting Up Weights & Biases"
    
    # Check if API key is set
    if [ "$WANDB_API_KEY" = "your-api-key-here" ] || [ -z "$WANDB_API_KEY" ]; then
        echo "⚠️  W&B API key not set in start.sh"
        echo "Training will continue without W&B logging."
        echo ""
        echo "To enable W&B, edit start.sh and set:"
        echo "  WANDB_API_KEY=\"your-key-from-wandb.ai/settings\""
        return 0
    fi
    
    # Login to W&B
    echo "Logging into Weights & Biases..."
    wandb login "$WANDB_API_KEY" --relogin
    
    if [ $? -eq 0 ]; then
        echo "✓ W&B login successful"
        echo "  Metrics will be logged to: https://wandb.ai"
    else
        echo "⚠️  W&B login failed, continuing without logging"
    fi
}

setup_huggingface() {
    print_header "Setting Up HuggingFace"
    
    # Check if token is set
    if [ -z "$HF_TOKEN" ]; then
        echo "⚠️  HuggingFace token not set in start.sh"
        echo "You may experience rate limiting (HTTP 429 errors)."
        echo ""
        echo "To fix, get a token from https://huggingface.co/settings/tokens"
        echo "Then edit start.sh and set: HF_TOKEN=\"your-token\""
        return 0
    fi
    
    # Set HuggingFace token as environment variable
    export HF_TOKEN
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    
    # Also login via CLI if available
    if command -v huggingface-cli &> /dev/null; then
        echo "Logging into HuggingFace..."
        huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true
    fi
    
    echo "✓ HuggingFace token set"
    echo "  Higher rate limits enabled"
}

setup_checkpoints() {
    print_header "Setting Up Checkpoint Directory"
    
    # Create checkpoint directory
    mkdir -p ./checkpoints
    
    # Create a test file to verify the directory works
    TEST_FILE="./checkpoints/.test_write"
    echo "Write test $(date)" > "$TEST_FILE"
    
    if [ -f "$TEST_FILE" ]; then
        echo "✓ Checkpoint directory is writable"
        rm "$TEST_FILE"
    else
        echo "✗ Cannot write to checkpoint directory!"
        exit 1
    fi
    
    # Show disk space
    echo ""
    echo "Disk space available:"
    df -h ./checkpoints | tail -1
    
    # Estimate checkpoint sizes
    echo ""
    echo "Estimated checkpoint size per save: ~200-500MB"
    echo "Recommended free space: 10GB+ for a full training run"
    
    # Warn if low on space
    AVAIL_GB=$(df -BG ./checkpoints | tail -1 | awk '{print $4}' | sed 's/G//')
    if [ "$AVAIL_GB" -lt 10 ]; then
        echo ""
        echo "⚠️  WARNING: Less than 10GB available!"
        echo "Consider reducing checkpoint frequency or cleaning up."
    fi
}

show_sync_instructions() {
    print_header "Checkpoint Sync Instructions"
    
    echo "To download checkpoints to your LOCAL machine, run this"
    echo "FROM YOUR LOCAL MACHINE (not here):"
    echo ""
    echo "  # One-time sync"
    echo "  ./scripts/sync_checkpoints.sh"
    echo ""
    echo "  # Continuous sync during training"
    echo "  ./scripts/sync_checkpoints.sh --watch"
    echo ""
    echo "  # Sync and delete from remote to save disk"
    echo "  ./scripts/sync_checkpoints.sh --watch --delete"
    echo ""
    echo "First, edit scripts/sync_checkpoints.sh with your Vast.ai connection details:"
    echo "  REMOTE_HOST=\"your-vast-ip\""
    echo "  REMOTE_PORT=\"your-port\""
    echo ""
    
    # Show current connection info for easy copying
    echo "Your current connection info (for sync script):"
    echo "  IP: $(hostname -I | awk '{print $1}')"
    echo "  Suggested SSH: ssh -p PORT root@$(hostname -I | awk '{print $1}')"
}

run_training() {
    print_header "Starting Training"
    
    # Export W&B run ID if specified (for resuming runs)
    if [ -n "$WANDB_RUN_ID" ]; then
        export WANDB_RUN_ID
        echo "Resuming W&B run: $WANDB_RUN_ID"
        echo ""
    fi
    
    # Determine launch command based on GPU count
    # Using -m flag to run as module from project root
    if [ "$GPU_COUNT" -gt 1 ]; then
        echo "Multi-GPU training with $GPU_COUNT GPUs"
        echo "Command: accelerate launch --num_processes=$GPU_COUNT -m training.pretrain.train"
        echo ""
        accelerate launch --num_processes="$GPU_COUNT" -m training.pretrain.train
    elif [ "$GPU_COUNT" -eq 1 ]; then
        echo "Single GPU training"
        echo "Command: accelerate launch -m training.pretrain.train"
        echo ""
        accelerate launch -m training.pretrain.train
    else
        echo "CPU training (this will be slow!)"
        echo "Command: python -m training.pretrain.train"
        echo ""
        python3 -m training.pretrain.train
    fi
}

# ============================================
# Main
# ============================================

print_header "BasicGPT Training Setup"
echo "Arguments: $@"
echo "Working directory: $(pwd)"

# Step 1: Detect GPUs
detect_gpus

# Step 2: Install dependencies (unless skipped)
if [ "$SKIP_DEPS" = false ]; then
    install_dependencies
else
    echo ""
    echo "Skipping dependency installation (--skip-deps)"
fi

# Step 3: Setup W&B
setup_wandb

# Step 4: Setup HuggingFace (for higher rate limits)
setup_huggingface

# Step 5: Setup checkpoint directory
setup_checkpoints

# Step 6: Show sync instructions
show_sync_instructions

# Step 7: Run training (unless dry-run mode)
if [ "$DRY_RUN" = true ]; then
    print_header "Dry Run Complete"
    echo "Everything is set up. Run without --dry-run to start training."
    echo ""
    echo "To start training manually:"
    if [ "$GPU_COUNT" -gt 1 ]; then
        echo "  accelerate launch --num_processes=$GPU_COUNT -m training.pretrain.train"
    else
        echo "  accelerate launch -m training.pretrain.train"
    fi
else
    run_training
fi

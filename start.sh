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
# Configuration - W&B API Key
# ============================================
# Set your W&B API key here (get it from https://wandb.ai/settings)
WANDB_API_KEY="8ab5900baef009c2bd6c9ed237c3319f3b476283"

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
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-deps   Skip dependency installation"
            echo "  --dry-run     Setup only, don't start training"
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
        
        # Get total VRAM
        TOTAL_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{sum+=$1} END {print sum}')
        echo "Total VRAM: ${TOTAL_VRAM}MB ($(echo "scale=1; $TOTAL_VRAM/1024" | bc)GB)"
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
    
    # Determine launch command based on GPU count
    if [ "$GPU_COUNT" -gt 1 ]; then
        echo "Multi-GPU training with $GPU_COUNT GPUs"
        echo "Command: accelerate launch --num_processes=$GPU_COUNT train.py"
        echo ""
        accelerate launch --num_processes="$GPU_COUNT" train.py
    elif [ "$GPU_COUNT" -eq 1 ]; then
        echo "Single GPU training"
        echo "Command: accelerate launch train.py"
        echo ""
        accelerate launch train.py
    else
        echo "CPU training (this will be slow!)"
        echo "Command: python train.py"
        echo ""
        python3 train.py
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

# Step 4: Setup checkpoint directory
setup_checkpoints

# Step 5: Show sync instructions
show_sync_instructions

# Step 6: Run training (unless dry-run mode)
if [ "$DRY_RUN" = true ]; then
    print_header "Dry Run Complete"
    echo "Everything is set up. Run without --dry-run to start training."
    echo ""
    echo "To start training manually:"
    if [ "$GPU_COUNT" -gt 1 ]; then
        echo "  accelerate launch --num_processes=$GPU_COUNT train.py"
    else
        echo "  accelerate launch train.py"
    fi
else
    run_training
fi

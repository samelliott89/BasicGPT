#!/bin/bash
#
# Vast.ai Connection Configuration
# Loads from .env file if available
#

# Get script directory
_VASTCONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_PROJECT_ROOT="$(dirname "$_VASTCONFIG_DIR")"

# Load from .env if exists
if [ -f "$_PROJECT_ROOT/.env" ]; then
    set -a
    source "$_PROJECT_ROOT/.env"
    set +a
fi

# Remote (Vast.ai) settings - can be overridden by .env
VAST_USER="root"
VAST_HOST="${VAST_HOST:-your-vast-ip}"
VAST_PORT="${VAST_PORT:-22}"
VAST_PATH="/root/BasicGPT"

# Local settings
LOCAL_CHECKPOINT_PATH="$_PROJECT_ROOT/checkpoints"

# SSH key (update if using a different key)
SSH_KEY="~/.ssh/id_ed25519"


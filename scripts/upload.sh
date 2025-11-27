#!/bin/bash
#
# Upload code to Vast.ai
# Uses configuration from vastconfig.sh
#

set -e

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
source "$SCRIPT_DIR/vastconfig.sh"

# Change to project root
cd "$PROJECT_ROOT"

echo "============================================"
echo "Uploading to Vast.ai"
echo "============================================"
echo "From: $PROJECT_ROOT"
echo "To:   $VAST_USER@$VAST_HOST:$VAST_PORT:$VAST_PATH/"
echo ""

rsync -avz --delete --checksum -e "ssh -p $VAST_PORT -i $SSH_KEY" \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='checkpoints' \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='wandb' \
    ./ "$VAST_USER@$VAST_HOST:$VAST_PATH/"

echo ""
echo "✓ Upload complete!"
echo ""
echo "Next: ssh -p $VAST_PORT $VAST_USER@$VAST_HOST"
echo "Then: cd $VAST_PATH && ./start.sh"


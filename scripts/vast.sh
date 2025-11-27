#!/bin/bash
#
# Vast.ai helper commands
# Sources configuration from vastconfig.sh
#

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/vastconfig.sh"

echo "============================================"
echo "Vast.ai Commands"
echo "============================================"
echo ""
echo "Configuration (edit scripts/vastconfig.sh):"
echo "  Host: $VAST_HOST"
echo "  Port: $VAST_PORT"
echo "  Path: $VAST_PATH"
echo ""
echo "Commands:"
echo ""
echo "# SSH into Vast.ai"
echo "ssh -p $VAST_PORT -i $SSH_KEY $VAST_USER@$VAST_HOST"
echo ""
echo "# Upload code to Vast.ai"
echo "rsync -avz -e \"ssh -p $VAST_PORT -i $SSH_KEY\" \\"
echo "    --exclude='__pycache__' \\"
echo "    --exclude='*.pyc' \\"
echo "    --exclude='checkpoints' \\"
echo "    --exclude='.git' \\"
echo "    --exclude='.venv' \\"
echo "    ./ $VAST_USER@$VAST_HOST:$VAST_PATH/"
echo ""

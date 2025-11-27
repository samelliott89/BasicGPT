#!/bin/bash
#
# Test Vast.ai connectivity BEFORE starting training
# Run this on your LOCAL machine to verify you can download checkpoints
#
# Usage:
#   ./scripts/test_remote_sync.sh
#

set -e

# ============================================
# Load configuration
# ============================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/vastconfig.sh"

# Map config to local variables
REMOTE_USER="$VAST_USER"
REMOTE_HOST="$VAST_HOST"
REMOTE_PORT="$VAST_PORT"
REMOTE_PATH="$VAST_PATH/checkpoints/"
LOCAL_PATH="$LOCAL_CHECKPOINT_PATH/"

# ============================================
# Main
# ============================================

echo "============================================"
echo "Vast.ai Sync Test"
echo "============================================"
echo ""
echo "Testing connectivity to:"
echo "  Host: $REMOTE_USER@$REMOTE_HOST"
echo "  Port: $REMOTE_PORT"
echo "  Path: $REMOTE_PATH"
echo ""

# Test 1: SSH connectivity
echo "Step 1: Testing SSH connection..."
if ssh -p "$REMOTE_PORT" -i "$SSH_KEY" -o ConnectTimeout=10 -o StrictHostKeyChecking=no \
    "$REMOTE_USER@$REMOTE_HOST" "echo 'SSH OK'" 2>/dev/null; then
    echo "  ✓ SSH connection successful"
else
    echo "  ✗ SSH connection FAILED"
    echo ""
    echo "  Please check:"
    echo "    - Is the Vast.ai machine running?"
    echo "    - Is VAST_HOST correct? (currently: $REMOTE_HOST)"
    echo "    - Is VAST_PORT correct? (currently: $REMOTE_PORT)"
    echo "    - Is your SSH key set up?"
    echo ""
    echo "  Edit: scripts/vastconfig.sh"
    exit 1
fi
echo ""

# Test 2: Check remote directory
echo "Step 2: Checking remote checkpoint directory..."
if ssh -p "$REMOTE_PORT" -i "$SSH_KEY" \
    "$REMOTE_USER@$REMOTE_HOST" "ls -la $REMOTE_PATH 2>/dev/null || mkdir -p $REMOTE_PATH && ls -la $REMOTE_PATH"; then
    echo "  ✓ Remote checkpoint directory accessible"
else
    echo "  ✗ Cannot access remote checkpoint directory"
    exit 1
fi
echo ""

# Test 3: Create test file on remote
echo "Step 3: Creating test checkpoint on remote..."
TEST_ID="test-$(date +%s)"
TEST_DIR="$REMOTE_PATH/$TEST_ID"

ssh -p "$REMOTE_PORT" -i "$SSH_KEY" \
    "$REMOTE_USER@$REMOTE_HOST" "mkdir -p $TEST_DIR && echo 'Sync test from $(hostname)' > $TEST_DIR/test.txt"
echo "  ✓ Created test file on remote"
echo ""

# Test 4: Download test file
echo "Step 4: Downloading test checkpoint..."
mkdir -p "$LOCAL_PATH"

if rsync -avz \
    -e "ssh -p $REMOTE_PORT -i $SSH_KEY" \
    "$REMOTE_USER@$REMOTE_HOST:$TEST_DIR" \
    "$LOCAL_PATH" 2>&1; then
    echo "  ✓ Download successful"
else
    echo "  ✗ Download FAILED"
    exit 1
fi
echo ""

# Test 5: Verify local file
echo "Step 5: Verifying downloaded file..."
if [ -f "$LOCAL_PATH/$TEST_ID/test.txt" ]; then
    echo "  ✓ File verified locally"
    cat "$LOCAL_PATH/$TEST_ID/test.txt"
else
    echo "  ✗ File not found locally"
    exit 1
fi
echo ""

# Cleanup
echo "Step 6: Cleaning up test files..."
rm -rf "$LOCAL_PATH/$TEST_ID"
ssh -p "$REMOTE_PORT" -i "$SSH_KEY" \
    "$REMOTE_USER@$REMOTE_HOST" "rm -rf $TEST_DIR"
echo "  ✓ Cleanup complete"
echo ""

# Success!
echo "============================================"
echo "✓ ALL TESTS PASSED!"
echo "============================================"
echo ""
echo "Your sync setup is working correctly."
echo ""
echo "Next steps:"
echo "  1. Start training on Vast.ai:"
echo "     ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST"
echo "     cd BasicGPT && ./start.sh"
echo ""
echo "  2. Run checkpoint sync on this machine:"
echo "     ./scripts/sync_checkpoints.sh --watch --delete"
echo ""

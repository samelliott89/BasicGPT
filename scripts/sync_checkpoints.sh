#!/bin/bash
#
# Sync checkpoints from remote GPU server to local machine
# Optionally delete from remote after successful download
#
# Usage:
#   ./scripts/sync_checkpoints.sh                    # Download only
#   ./scripts/sync_checkpoints.sh --delete           # Download and delete from remote
#   ./scripts/sync_checkpoints.sh --watch            # Watch and sync continuously
#   ./scripts/sync_checkpoints.sh --watch --delete   # Watch, sync, and delete
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
# Parse arguments
# ============================================
DELETE_AFTER_SYNC=false
WATCH_MODE=false

for arg in "$@"; do
    case $arg in
        --delete)
            DELETE_AFTER_SYNC=true
            shift
            ;;
        --watch)
            WATCH_MODE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--delete] [--watch]"
            echo ""
            echo "Options:"
            echo "  --delete    Delete checkpoints from remote after successful download"
            echo "  --watch     Continuously watch for new checkpoints (every 60s)"
            echo ""
            echo "Configuration (edit scripts/vastconfig.sh):"
            echo "  VAST_HOST: $VAST_HOST"
            echo "  VAST_PORT: $VAST_PORT"
            echo "  VAST_PATH: $VAST_PATH"
            exit 0
            ;;
    esac
done

# ============================================
# Functions
# ============================================

sync_checkpoints() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Syncing checkpoints..."
    
    # Create local directory if it doesn't exist
    mkdir -p "$LOCAL_PATH"
    
    # Show what we're syncing
    echo "From: $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
    echo "To:   $LOCAL_PATH"
    echo ""
    
    # First, list what's on remote with sizes
    echo "Remote checkpoints:"
    ssh -p "$REMOTE_PORT" -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" \
        "ls -lah $REMOTE_PATH/*/checkpoint.pt 2>/dev/null || echo 'No checkpoint.pt files found'" || true
    echo ""
    
    # Use scp to copy each checkpoint folder (more reliable than rsync for some setups)
    # First get list of checkpoint folders
    FOLDERS=$(ssh -p "$REMOTE_PORT" -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" \
        "ls -d ${REMOTE_PATH}data-* 2>/dev/null" || true)
    
    if [ -z "$FOLDERS" ]; then
        echo "No checkpoint folders found on remote"
        return 0
    fi
    
    for folder in $FOLDERS; do
        folder_name=$(basename "$folder")
        local_folder="$LOCAL_PATH/$folder_name"
        
        # Check if we already have this checkpoint locally
        if [ -f "$local_folder/checkpoint.pt" ]; then
            echo "  ⏭️  Skipping $folder_name (already exists locally)"
            continue
        fi
        
        echo "  📥 Downloading $folder_name..."
        mkdir -p "$local_folder"
        
        # Copy the checkpoint.pt file
        scp -P "$REMOTE_PORT" -i "$SSH_KEY" \
            "$REMOTE_USER@$REMOTE_HOST:$folder/checkpoint.pt" \
            "$local_folder/checkpoint.pt"
        
        if [ $? -eq 0 ]; then
            echo "     ✓ Downloaded checkpoint.pt"
        else
            echo "     ✗ Failed to download"
        fi
    done
    
    echo ""
    echo "Local checkpoints:"
    ls -lah "$LOCAL_PATH"/*/checkpoint.pt 2>/dev/null || echo "No local checkpoints yet"
    echo ""
    echo "✓ Sync completed"
    
    if [ "$DELETE_AFTER_SYNC" = true ]; then
        delete_remote_checkpoints
    fi
}

delete_remote_checkpoints() {
    echo "Deleting old checkpoints from remote (keeping checkpoint_best.pt)..."
    
    # Get list of checkpoint folders (excluding checkpoint_best.pt)
    FOLDERS=$(ssh -p "$REMOTE_PORT" -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" \
        "find $REMOTE_PATH -maxdepth 1 -type d -name 'data-*' 2>/dev/null")
    
    if [ -z "$FOLDERS" ]; then
        echo "  No checkpoint folders to delete"
        return 0
    fi
    
    # Count and list folders
    FOLDER_COUNT=$(echo "$FOLDERS" | wc -l)
    echo "  Found $FOLDER_COUNT checkpoint folder(s) to delete"
    
    # Delete each folder
    for folder in $FOLDERS; do
        # Check if we have this locally before deleting
        folder_name=$(basename "$folder")
        if [ -d "$LOCAL_PATH/$folder_name" ]; then
            echo "  Deleting: $folder_name (verified local copy exists)"
            ssh -p "$REMOTE_PORT" -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" \
                "rm -rf $folder"
        else
            echo "  SKIPPING: $folder_name (no local copy found!)"
        fi
    done
    
    # Show remaining disk usage on remote
    echo ""
    echo "Remote disk usage after cleanup:"
    ssh -p "$REMOTE_PORT" -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" \
        "df -h $REMOTE_PATH | tail -1"
}

show_status() {
    echo ""
    echo "============================================"
    echo "Checkpoint Sync Status"
    echo "============================================"
    echo "Remote: $REMOTE_USER@$REMOTE_HOST:$REMOTE_PORT"
    echo "Path: $REMOTE_PATH"
    echo "Delete after sync: $DELETE_AFTER_SYNC"
    echo "Watch mode: $WATCH_MODE"
    echo "============================================"
    echo ""
}

# ============================================
# Main
# ============================================

show_status

if [ "$WATCH_MODE" = true ]; then
    echo "Starting watch mode (Ctrl+C to stop)..."
    echo ""
    
    while true; do
        sync_checkpoints
        echo ""
        echo "Waiting 60 seconds before next sync..."
        echo "(Press Ctrl+C to stop)"
        sleep 60
    done
else
    sync_checkpoints
fi

echo ""
echo "Done!"

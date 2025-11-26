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
# CONFIGURATION - Edit these for your setup
# ============================================
REMOTE_USER="root"
REMOTE_HOST="your-vast-ai-ip"
REMOTE_PORT="your-port"
REMOTE_PATH="/root/BasicGPT/checkpoints/"
LOCAL_PATH="./checkpoints/"
SSH_KEY="~/.ssh/id_ed25519"

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
            echo "Configuration (edit script):"
            echo "  REMOTE_HOST: $REMOTE_HOST"
            echo "  REMOTE_PORT: $REMOTE_PORT"
            echo "  REMOTE_PATH: $REMOTE_PATH"
            echo "  LOCAL_PATH:  $LOCAL_PATH"
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
    
    # Rsync checkpoints from remote
    # -a: archive mode (preserves permissions, timestamps)
    # -v: verbose
    # -z: compress during transfer
    # --progress: show progress
    rsync -avz --progress \
        -e "ssh -p $REMOTE_PORT -i $SSH_KEY" \
        "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH" \
        "$LOCAL_PATH"
    
    if [ $? -eq 0 ]; then
        echo "✓ Sync completed successfully"
        
        if [ "$DELETE_AFTER_SYNC" = true ]; then
            delete_remote_checkpoints
        fi
    else
        echo "✗ Sync failed!"
        return 1
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


#!/usr/bin/env bash
# Sync DPO/GRPO data + scripts to kxkm-ai and launch training.
#
# Usage:
#   ./scripts/sync_and_train_kxkm.sh dpo [--domain kicad-dsl]
#   ./scripts/sync_and_train_kxkm.sh grpo [--domain spice]
#   ./scripts/sync_and_train_kxkm.sh dpo --all
#   ./scripts/sync_and_train_kxkm.sh grpo --all --dry-run
#   ./scripts/sync_and_train_kxkm.sh sync-only
#   ./scripts/sync_and_train_kxkm.sh kill-llama   # free 16.5GB VRAM for 35B
#
# Requires: ssh kxkm-ai (passwordless)
set -euo pipefail

REMOTE="kxkm-ai"
REMOTE_DIR="/home/kxkm/micro-kiki"
REMOTE_VENV="/home/kxkm/KIKI-models-tuning/.venv/bin/activate"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

MODE="${1:-help}"
shift || true

log() { echo "[$(date +%H:%M:%S)] $*"; }

sync_data() {
    log "Syncing data to $REMOTE:$REMOTE_DIR ..."

    # Ensure remote dirs exist
    ssh "$REMOTE" "mkdir -p $REMOTE_DIR/data/dpo $REMOTE_DIR/data/grpo $REMOTE_DIR/data/merged $REMOTE_DIR/scripts $REMOTE_DIR/output"

    # Sync DPO data
    rsync -avz --progress "$LOCAL_DIR/data/dpo/" "$REMOTE:$REMOTE_DIR/data/dpo/"

    # Sync merged data (for GRPO prompts)
    if [ -d "$LOCAL_DIR/data/merged" ]; then
        rsync -avz --progress "$LOCAL_DIR/data/merged/" "$REMOTE:$REMOTE_DIR/data/merged/"
    fi

    # Sync training scripts
    rsync -avz "$LOCAL_DIR/scripts/train_dpo_kxkm.py" "$REMOTE:$REMOTE_DIR/scripts/"
    rsync -avz "$LOCAL_DIR/scripts/train_grpo_kxkm.py" "$REMOTE:$REMOTE_DIR/scripts/"

    log "Sync complete."
}

check_gpu() {
    log "GPU status on $REMOTE:"
    ssh "$REMOTE" "nvidia-smi --query-gpu=memory.used,memory.total,memory.free --format=csv,noheader"
    log "Running processes:"
    ssh "$REMOTE" "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || echo 'No compute processes'"
}

kill_llama() {
    log "Killing llama-server on $REMOTE to free VRAM..."
    ssh "$REMOTE" "pkill -f llama-server || echo 'No llama-server running'"
    sleep 2
    check_gpu
}

run_training() {
    local script="$1"
    shift
    log "Launching $script on $REMOTE with args: $*"
    # Use nohup + tmux so training survives SSH disconnect
    ssh "$REMOTE" "tmux new-session -d -s training 'source $REMOTE_VENV && cd $REMOTE_DIR && python3 scripts/$script $*' 2>/dev/null || ssh $REMOTE 'source $REMOTE_VENV && cd $REMOTE_DIR && nohup python3 scripts/$script $* > output/training.log 2>&1 &'"
    log "Training launched in tmux session 'training'. Monitor with: ssh $REMOTE 'tmux attach -t training'"
}

case "$MODE" in
    sync-only)
        sync_data
        check_gpu
        ;;
    dpo)
        sync_data
        check_gpu
        run_training "train_dpo_kxkm.py" "$@"
        ;;
    grpo)
        sync_data
        check_gpu
        run_training "train_grpo_kxkm.py" "$@"
        ;;
    kill-llama)
        kill_llama
        ;;
    gpu)
        check_gpu
        ;;
    help|*)
        echo "Usage: $0 {sync-only|dpo|grpo|kill-llama|gpu} [training args...]"
        echo ""
        echo "Examples:"
        echo "  $0 sync-only                    # just sync data + scripts"
        echo "  $0 dpo --domain kicad-dsl       # DPO on one domain (4B model)"
        echo "  $0 dpo --all                    # DPO on all 5 domains"
        echo "  $0 grpo --domain spice          # GRPO on one domain"
        echo "  $0 grpo --all --dry-run         # dry-run GRPO"
        echo "  $0 kill-llama                   # free GPU VRAM"
        echo "  $0 gpu                          # check GPU status"
        echo ""
        echo "For 35B model (needs ~11GB in 4-bit):"
        echo "  $0 kill-llama"
        echo "  $0 dpo --all --model Qwen/Qwen3.6-35B-A3B --lora-rank 8"
        echo ""
        echo "For local 4B model (no HF download):"
        echo "  $0 dpo --domain kicad-dsl --local-model /home/kxkm/models/qwen3.5-4b/bf16"
        ;;
esac

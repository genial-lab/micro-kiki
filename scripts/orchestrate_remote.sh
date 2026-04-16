#!/usr/bin/env bash
# orchestrate_remote.sh — coordinate micro-kiki across 3 machines
#
# Usage: ./scripts/orchestrate_remote.sh <command> [args]
#
# Commands:
#   status          Show status of all machines
#   sync            Pull latest main on all machines
#   train <stack>   Train a stack on Studio (BF16 LoRA, M3 Ultra)
#   eval <stack>    Evaluate a stack on kxkm-ai (RTX 4090)
#   distill <domain> Run distillation on Studio (teacher LLM)
#   ralph <machine> Start Ralph loop on target machine
#   kill-zombie     Kill stalled jobs on kxkm-ai
#
# Machines:
#   grosmac  — local (M5), orchestration hub
#   studio   — ssh studio (M3 Ultra 512GB), training + teacher
#   kxkm-ai  — ssh kxkm@kxkm-ai (RTX 4090 24GB), inference + eval

set -euo pipefail

REPO="micro-kiki"
STUDIO_SSH="studio"
KXKM_SSH="kxkm@kxkm-ai"
STUDIO_DIR="/Users/clems/${REPO}"
KXKM_DIR="/home/kxkm/${REPO}"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${CYAN}[orch]${NC} $*"; }
ok()   { echo -e "${GREEN}[  OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[ ERR]${NC} $*" >&2; }

cmd_status() {
    log "=== GrosMac (local) ==="
    cd "$LOCAL_DIR"
    echo "  branch: $(git branch --show-current)"
    echo "  HEAD:   $(git log --oneline -1)"
    echo "  dirty:  $(git status --porcelain | wc -l | tr -d ' ') files"

    log "=== Studio ($STUDIO_SSH) ==="
    ssh -o ConnectTimeout=5 "$STUDIO_SSH" "
        cd $STUDIO_DIR 2>/dev/null || { echo '  repo: NOT FOUND'; exit 0; }
        echo \"  branch: \$(git branch --show-current)\"
        echo \"  HEAD:   \$(git log --oneline -1)\"
        echo \"  dirty:  \$(git status --porcelain | wc -l | tr -d ' ') files\"
        echo \"  claude: \$(ps aux | grep -c '[c]laude') sessions\"
        echo \"  download: \$(du -sh models/qwen3.5-35b-a3b/ 2>/dev/null | cut -f1 || echo 'none')\"
    " 2>/dev/null || warn "Studio unreachable"

    log "=== kxkm-ai ($KXKM_SSH) ==="
    ssh -o ConnectTimeout=5 "$KXKM_SSH" "
        cd $KXKM_DIR 2>/dev/null || { echo '  repo: NOT FOUND'; exit 0; }
        echo \"  branch: \$(git branch --show-current)\"
        echo \"  HEAD:   \$(git log --oneline -1)\"
        echo \"  gpu:    \$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo 'N/A')\"
        echo \"  claude: \$(ps aux | grep -c '[c]laude') sessions\"
        echo \"  llama:  \$(ps aux | grep '[l]lama-server' | grep -oP 'port \K\d+' || echo 'not running')\"
    " 2>/dev/null || warn "kxkm-ai unreachable"
}

cmd_sync() {
    log "Syncing all machines to latest main..."

    log "GrosMac: git pull"
    cd "$LOCAL_DIR" && git pull --rebase origin main

    log "Studio: git pull"
    ssh "$STUDIO_SSH" "cd $STUDIO_DIR && git stash && git pull --rebase origin main && git stash pop 2>/dev/null || true" 2>&1 | tail -5

    log "kxkm-ai: git pull"
    ssh "$KXKM_SSH" "cd $KXKM_DIR && git pull --rebase origin main" 2>&1 | tail -5

    ok "All machines synced"
}

cmd_train() {
    local stack="${1:?Usage: orchestrate_remote.sh train <stack-id>}"
    local config="configs/${stack}.yaml"

    if [ ! -f "$LOCAL_DIR/$config" ]; then
        err "Config not found: $config"
        exit 1
    fi

    log "Training $stack on Studio (BF16 LoRA, M3 Ultra)..."
    log "Syncing Studio first..."
    ssh "$STUDIO_SSH" "cd $STUDIO_DIR && git pull --rebase origin main" 2>&1 | tail -3

    log "Launching training..."
    ssh "$STUDIO_SSH" "
        cd $STUDIO_DIR
        export UNSLOTH_COMPILE_DISABLE=1
        nohup uv run python -m src.stacks.trainer \
            --config $config \
            > outputs/train-${stack}.log 2>&1 &
        echo \"PID: \$!\"
        echo \"Log: outputs/train-${stack}.log\"
    "
    ok "Training launched on Studio. Monitor: ssh studio tail -f $STUDIO_DIR/outputs/train-${stack}.log"
}

cmd_eval() {
    local stack="${1:?Usage: orchestrate_remote.sh eval <stack-id>}"

    log "Evaluating $stack on kxkm-ai (RTX 4090)..."
    ssh "$KXKM_SSH" "
        cd $KXKM_DIR
        uv run python -m src.eval.stack_eval --stack $stack 2>&1
    " | tee "$LOCAL_DIR/results/${stack}-eval.json"
    ok "Eval results saved to results/${stack}-eval.json"
}

cmd_distill() {
    local domain="${1:?Usage: orchestrate_remote.sh distill <domain>}"
    local script="scripts/distill_${domain}.py"

    log "Running distillation for $domain on Studio..."
    ssh "$STUDIO_SSH" "
        cd $STUDIO_DIR
        nohup uv run python $script \
            --teacher-url http://localhost:8000 \
            --max-examples 2000 \
            > outputs/distill-${domain}.log 2>&1 &
        echo \"PID: \$!\"
    "
    ok "Distillation launched. Monitor: ssh studio tail -f $STUDIO_DIR/outputs/distill-${domain}.log"
}

cmd_ralph() {
    local machine="${1:?Usage: orchestrate_remote.sh ralph <grosmac|studio|kxkm-ai>}"

    case "$machine" in
        grosmac)
            log "Starting Ralph loop locally..."
            cd "$LOCAL_DIR"
            MAX_ITERATIONS="${MAX_ITERATIONS:-10}" uv run .ralph/loop.py
            ;;
        studio)
            log "Starting Ralph loop on Studio..."
            ssh "$STUDIO_SSH" "
                cd $STUDIO_DIR
                git pull --rebase origin main
                nohup env MAX_ITERATIONS=${MAX_ITERATIONS:-10} \
                    /Users/clems/.local/bin/uv run .ralph/loop.py \
                    > outputs/ralph-loop.log 2>&1 &
                echo \"Ralph PID: \$!\"
            "
            ok "Ralph loop launched on Studio"
            ;;
        kxkm-ai)
            log "Starting Ralph loop on kxkm-ai..."
            ssh "$KXKM_SSH" "
                cd $KXKM_DIR
                git pull origin main
                nohup env MAX_ITERATIONS=${MAX_ITERATIONS:-10} \
                    ~/.local/bin/uv run .ralph/loop.py \
                    > outputs/ralph-loop.log 2>&1 &
                echo \"Ralph PID: \$!\"
            "
            ok "Ralph loop launched on kxkm-ai"
            ;;
        *)
            err "Unknown machine: $machine (use grosmac, studio, or kxkm-ai)"
            exit 1
            ;;
    esac
}

cmd_kill_zombie() {
    log "Killing stalled jobs on kxkm-ai..."
    ssh "$KXKM_SSH" "
        # Kill stalled devstral-v4 chain
        pkill -f 'chain-devstral-v4-sft.sh' 2>/dev/null && echo 'Killed devstral-v4 chain' || echo 'No devstral-v4 chain found'
        # Kill associated tee
        pkill -f 'devstral-v4-opus-sft/run-' 2>/dev/null && echo 'Killed tee logger' || echo 'No tee logger'
    "
    ok "Zombie cleanup done"
}

# Main dispatcher
case "${1:-}" in
    status)       cmd_status ;;
    sync)         cmd_sync ;;
    train)        cmd_train "${2:-}" ;;
    eval)         cmd_eval "${2:-}" ;;
    distill)      cmd_distill "${2:-}" ;;
    ralph)        cmd_ralph "${2:-}" ;;
    kill-zombie)  cmd_kill_zombie ;;
    *)
        echo "Usage: $0 <command> [args]"
        echo ""
        echo "Commands:"
        echo "  status          Show all machines status"
        echo "  sync            Pull main on all machines"
        echo "  train <stack>   Train stack on Studio"
        echo "  eval <stack>    Eval stack on kxkm-ai"
        echo "  distill <domain> Run distillation on Studio"
        echo "  ralph <machine> Start Ralph loop (grosmac|studio|kxkm-ai)"
        echo "  kill-zombie     Kill stalled jobs on kxkm-ai"
        exit 1
        ;;
esac

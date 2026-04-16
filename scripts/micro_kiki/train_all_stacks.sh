#!/usr/bin/env bash
# ==============================================================================
# Brainstacks — Train all 32 domain stacks sequentially
#
# Each stack is trained in curriculum order. After each stack:
#   1. The stack is frozen and saved to disk
#   2. All previous domains are evaluated for forgetting
#   3. If any domain's loss degrades > 0.03, training pauses
#
# Usage:
#   ./scripts/micro_kiki/train_all_stacks.sh
#   ./scripts/micro_kiki/train_all_stacks.sh --resume-from 5
#   ./scripts/micro_kiki/train_all_stacks.sh --dry-run
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="$PROJECT_ROOT/configs/micro_kiki/brainstacks.yaml"
LOG_DIR="$PROJECT_ROOT/output/micro-kiki/logs"

# --------------------------------------------------------------------------
# Parse arguments
# --------------------------------------------------------------------------
RESUME_FROM=1
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume-from)
            RESUME_FROM="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--resume-from N] [--dry-run] [--config PATH]"
            echo ""
            echo "Options:"
            echo "  --resume-from N   Skip the first N-1 stacks (start at stack N)"
            echo "  --dry-run         Print commands without executing"
            echo "  --config PATH     Override brainstacks.yaml path"
            echo "  -h, --help        Show this help"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Run $0 --help for usage."
            exit 1
            ;;
    esac
done

# --------------------------------------------------------------------------
# Curriculum order (must match brainstacks.yaml)
# --------------------------------------------------------------------------
DOMAINS=(
    # Phase 1 — Foundations
    chat-fr
    reasoning
    # Phase 2 — Coding core
    python
    typescript
    cpp
    rust
    # Phase 3 — Coding secondary
    html-css
    shell
    sql
    yaml-json
    docker
    kicad-dsl
    spice
    lua-upy
    # Phase 4 — Technical domains
    embedded
    stm32
    iot
    freecad
    platformio
    power
    emc
    dsp
    spice-sim
    electronics
    kicad-pcb
    # Phase 5 — Applications
    web-frontend
    web-backend
    music-audio
    devops
    llm-orch
    # Phase 6 — Complements
    math
    security
)

TOTAL=${#DOMAINS[@]}

# --------------------------------------------------------------------------
# Banner
# --------------------------------------------------------------------------
echo "================================================================"
echo "Brainstacks — ${TOTAL} MoE-LoRA Stack Training"
echo "================================================================"
echo "Config:       $CONFIG"
echo "Total stacks: $TOTAL"
echo "Resume from:  $RESUME_FROM"
echo "Dry run:      $DRY_RUN"
echo "Log dir:      $LOG_DIR"
echo "================================================================"

# --------------------------------------------------------------------------
# Validate resume-from
# --------------------------------------------------------------------------
if [[ "$RESUME_FROM" -lt 1 || "$RESUME_FROM" -gt "$TOTAL" ]]; then
    echo "ERROR: --resume-from must be between 1 and $TOTAL (got $RESUME_FROM)"
    exit 1
fi

# --------------------------------------------------------------------------
# Pre-flight: verify all 32 train.jsonl files exist
# --------------------------------------------------------------------------
echo ""
echo "Pre-flight check: verifying datasets..."
MISSING=0
for domain in "${DOMAINS[@]}"; do
    train_file="$PROJECT_ROOT/data/micro-kiki/$domain/train.jsonl"
    if [[ ! -f "$train_file" ]]; then
        echo "  MISSING: $train_file"
        MISSING=$((MISSING + 1))
    fi
done

if [[ "$MISSING" -gt 0 ]]; then
    echo ""
    echo "ERROR: $MISSING domain dataset(s) missing."
    echo "Run Plan 1 (Data Pipeline) first to generate all datasets."
    exit 1
fi

echo "All $TOTAL domain datasets found."

# --------------------------------------------------------------------------
# Verify config file exists
# --------------------------------------------------------------------------
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi

mkdir -p "$LOG_DIR"

# --------------------------------------------------------------------------
# Timing arrays
# --------------------------------------------------------------------------
declare -a STACK_NAMES=()
declare -a STACK_DURATIONS=()
GLOBAL_START=$(date +%s)

# --------------------------------------------------------------------------
# Train each stack
# --------------------------------------------------------------------------
for i in $(seq 1 "$TOTAL"); do
    domain="${DOMAINS[$((i - 1))]}"

    if [[ "$i" -lt "$RESUME_FROM" ]]; then
        echo "[${i}/${TOTAL}] Skipping $domain (before resume point)"
        continue
    fi

    echo ""
    echo "================================================================"
    echo "[${i}/${TOTAL}] Training stack: $domain"
    echo "================================================================"

    LOG_FILE="$LOG_DIR/${i}-${domain}.log"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY RUN] python $SCRIPT_DIR/train_stack.py \\"
        echo "      --config $CONFIG \\"
        echo "      --domain $domain \\"
        echo "      --stack-index $i"
        STACK_NAMES+=("$domain")
        STACK_DURATIONS+=(0)
        continue
    fi

    START_TIME=$(date +%s)

    # Run training, tee to log, preserve exit code through pipe
    set +e
    python "$SCRIPT_DIR/train_stack.py" \
        --config "$CONFIG" \
        --domain "$domain" \
        --stack-index "$i" \
        2>&1 | tee "$LOG_FILE"
    EXIT_CODE=${PIPESTATUS[0]}
    set -e

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    STACK_NAMES+=("$domain")
    STACK_DURATIONS+=("$DURATION")

    if [[ "$EXIT_CODE" -ne 0 ]]; then
        echo ""
        echo "ERROR: Stack $domain failed (exit code $EXIT_CODE)"
        echo "Log: $LOG_FILE"
        echo ""
        echo "To resume from this point:"
        echo "  $0 --resume-from $i"
        exit 1
    fi

    # Verify adapters.safetensors was produced
    STACK_DIR="$PROJECT_ROOT/output/micro-kiki/stacks/$domain"
    if [[ ! -f "$STACK_DIR/adapters.safetensors" ]]; then
        echo ""
        echo "ERROR: Stack $domain completed but adapters.safetensors not found at:"
        echo "  $STACK_DIR/adapters.safetensors"
        echo ""
        echo "To resume from this point:"
        echo "  $0 --resume-from $i"
        exit 1
    fi

    STACK_SIZE=$(du -sh "$STACK_DIR" | cut -f1)
    MINUTES=$((DURATION / 60))
    SECONDS_REM=$((DURATION % 60))
    echo ""
    echo "[${i}/${TOTAL}] $domain complete in ${MINUTES}m${SECONDS_REM}s  (stack size: $STACK_SIZE)"
done

# --------------------------------------------------------------------------
# Timing summary
# --------------------------------------------------------------------------
GLOBAL_END=$(date +%s)
GLOBAL_DURATION=$((GLOBAL_END - GLOBAL_START))
GLOBAL_MINUTES=$((GLOBAL_DURATION / 60))
GLOBAL_SECONDS=$((GLOBAL_DURATION % 60))

echo ""
echo "================================================================"
echo "Timing Summary"
echo "================================================================"
printf "%-4s %-20s %s\n" "#" "Domain" "Duration"
printf "%-4s %-20s %s\n" "---" "--------------------" "--------"

for idx in "${!STACK_NAMES[@]}"; do
    d="${STACK_DURATIONS[$idx]}"
    m=$((d / 60))
    s=$((d % 60))
    n=$((idx + RESUME_FROM))
    if [[ "$DRY_RUN" == "true" ]]; then
        printf "%-4s %-20s %s\n" "$n" "${STACK_NAMES[$idx]}" "(dry run)"
    else
        printf "%-4s %-20s %dm%02ds\n" "$n" "${STACK_NAMES[$idx]}" "$m" "$s"
    fi
done

echo "---"
printf "Total: %dm%02ds\n" "$GLOBAL_MINUTES" "$GLOBAL_SECONDS"

if [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    echo "Dry run complete. No training was executed."
    exit 0
fi

# --------------------------------------------------------------------------
# Final forgetting evaluation
# --------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "ALL $TOTAL STACKS TRAINED SUCCESSFULLY"
echo "================================================================"
echo ""
echo "Output: $PROJECT_ROOT/output/micro-kiki/stacks/"
echo ""
echo "Running final forgetting evaluation..."
echo ""

python "$SCRIPT_DIR/eval_stack.py" --config "$CONFIG" --all

echo ""
echo "Next steps:"
echo "  1. Review forgetting matrix above"
echo "  2. Proceed to Plan 3 (Meta-Router Training)"

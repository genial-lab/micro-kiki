#!/bin/bash
# Micro_KIKI Data Pipeline — Full Orchestrator
#
# Runs all data preparation steps in sequence:
#   1. Download public datasets
#   2. Classify into 32 domains
#   3. Generate synthetic data for sparse domains
#   4. Deduplicate cross-domain
#   5. Split train/valid per domain
#
# Usage:
#   ./scripts/micro_kiki/pipeline_data.sh [--skip-download] [--skip-generate] [--dry-run]
#
# Options:
#   --skip-download    Skip dataset download (use existing data/raw/)
#   --skip-generate    Skip synthetic generation (use classified data only)
#   --dry-run          Show generation plan without running teachers
#   --teacher MODEL    Override teacher model path

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Defaults
SKIP_DOWNLOAD=false
SKIP_GENERATE=false
DRY_RUN=""
TEACHER_MODEL="models/Qwen3.5-35B-A3B-Opus-vlm"
CONFIG="configs/micro_kiki/domains.yaml"
MAX_PER_DOMAIN=3000
MAX_GENERATE=500

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-download)  SKIP_DOWNLOAD=true; shift ;;
        --skip-generate)  SKIP_GENERATE=true; shift ;;
        --dry-run)        DRY_RUN="--dry-run"; shift ;;
        --teacher)        TEACHER_MODEL="$2"; shift 2 ;;
        --config)         CONFIG="$2"; shift 2 ;;
        --max-generate)   MAX_GENERATE="$2"; shift 2 ;;
        *)                echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Activate venv
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

echo "=========================================="
echo "  Micro_KIKI Data Pipeline"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""
echo "Config:         $CONFIG"
echo "Teacher:        $TEACHER_MODEL"
echo "Skip download:  $SKIP_DOWNLOAD"
echo "Skip generate:  $SKIP_GENERATE"
echo "Dry run:        ${DRY_RUN:-no}"
echo ""

START_TIME=$(date +%s)

# Step 1: Download
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo "=== Step 1/5: Download datasets ==="
    bash "$SCRIPT_DIR/download_datasets.sh" all
    echo ""
else
    echo "=== Step 1/5: Download — SKIPPED ==="
    echo ""
fi

# Step 2: Classify
echo "=== Step 2/5: Classify into 32 domains ==="
python "$SCRIPT_DIR/classify_domains.py" \
    --config "$CONFIG" \
    --input-dir "data/raw" \
    --output-dir "data/micro-kiki/classified" \
    --max-per-domain "$MAX_PER_DOMAIN"
echo ""

# Step 3: Generate (optional)
if [ "$SKIP_GENERATE" = false ]; then
    echo "=== Step 3/5: Generate synthetic data for sparse domains ==="
    python "$SCRIPT_DIR/generate_missing.py" \
        --config "$CONFIG" \
        --classified-dir "data/micro-kiki/classified" \
        --output-dir "data/micro-kiki/generated" \
        --teacher-model "$TEACHER_MODEL" \
        --max-generate "$MAX_GENERATE" \
        $DRY_RUN
    echo ""
else
    echo "=== Step 3/5: Generate — SKIPPED ==="
    # Create empty generated dir so dedup doesn't fail
    mkdir -p "data/micro-kiki/generated"
    echo ""
fi

# Step 4: Deduplicate
echo "=== Step 4/5: Cross-domain deduplication ==="
python "$SCRIPT_DIR/deduplicate.py" \
    --config "$CONFIG" \
    --classified-dir "data/micro-kiki/classified" \
    --generated-dir "data/micro-kiki/generated" \
    --output-dir "data/micro-kiki/deduped"
echo ""

# Step 5: Split
echo "=== Step 5/5: Train/valid split ==="
python "$SCRIPT_DIR/split_domains.py" \
    --config "$CONFIG" \
    --input-dir "data/micro-kiki/deduped" \
    --output-dir "data/micro-kiki"
echo ""

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo "=========================================="
echo "  Pipeline complete in ${MINUTES}m${SECONDS}s"
echo "=========================================="
echo ""

# Final summary
echo "=== Final data structure ==="
for domain_dir in data/micro-kiki/*/; do
    if [ -f "${domain_dir}train.jsonl" ]; then
        domain="$(basename "$domain_dir")"
        train_count=$(wc -l < "${domain_dir}train.jsonl" 2>/dev/null || echo 0)
        valid_count=$(wc -l < "${domain_dir}valid.jsonl" 2>/dev/null || echo 0)
        printf "  %-16s train=%5s valid=%4s\n" "$domain" "$train_count" "$valid_count"
    fi
done

total_train=$(cat data/micro-kiki/*/train.jsonl 2>/dev/null | wc -l || echo 0)
total_valid=$(cat data/micro-kiki/*/valid.jsonl 2>/dev/null | wc -l || echo 0)
echo ""
echo "  TOTAL: train=$total_train valid=$total_valid"

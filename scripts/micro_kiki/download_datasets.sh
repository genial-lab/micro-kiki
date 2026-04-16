#!/bin/bash
# Download all public datasets needed for Micro_KIKI 32-domain classification.
# Usage: ./scripts/micro_kiki/download_datasets.sh [dataset|all]
#
# Downloads to data/raw/<dataset-name>/
# Uses `hf` CLI (not huggingface-cli which is deprecated).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"

RAW_DIR="$PROJECT_DIR/data/raw"
mkdir -p "$RAW_DIR"

download_codefeedback() {
    echo "=== CodeFeedback-Filtered-Instruction (156K) ==="
    local dest="$RAW_DIR/CodeFeedback-Filtered-Instruction"
    if [ -d "$dest" ] && [ "$(ls -A "$dest" 2>/dev/null)" ]; then
        echo "  Already downloaded at $dest"
        return
    fi
    hf download m-a-p/CodeFeedback-Filtered-Instruction \
        --repo-type dataset \
        --local-dir "$dest"
    echo "  Done: $(find "$dest" -name '*.parquet' -o -name '*.jsonl' | wc -l) files"
}

download_opencodereasoning() {
    echo "=== OpenCodeReasoning (735K) ==="
    local dest="$RAW_DIR/OpenCodeReasoning"
    if [ -d "$dest" ] && [ "$(ls -A "$dest" 2>/dev/null)" ]; then
        echo "  Already downloaded at $dest"
        return
    fi
    hf download nvidia/OpenCodeReasoning \
        --repo-type dataset \
        --local-dir "$dest"
    echo "  Done: $(find "$dest" -name '*.parquet' -o -name '*.jsonl' | wc -l) files"
}

download_magicoder() {
    echo "=== Magicoder-OSS-Instruct-75K ==="
    local dest="$RAW_DIR/Magicoder-OSS-Instruct-75K"
    if [ -d "$dest" ] && [ "$(ls -A "$dest" 2>/dev/null)" ]; then
        echo "  Already downloaded at $dest"
        return
    fi
    hf download ise-uiuc/Magicoder-OSS-Instruct-75K \
        --repo-type dataset \
        --local-dir "$dest"
    echo "  Done: $(find "$dest" -name '*.parquet' -o -name '*.jsonl' | wc -l) files"
}

download_openhermes() {
    echo "=== OpenHermes-2.5 (1M, general) ==="
    local dest="$RAW_DIR/OpenHermes-2.5"
    if [ -d "$dest" ] && [ "$(ls -A "$dest" 2>/dev/null)" ]; then
        echo "  Already downloaded at $dest"
        return
    fi
    hf download teknium/OpenHermes-2.5 \
        --repo-type dataset \
        --local-dir "$dest"
    echo "  Done: $(find "$dest" -name '*.parquet' -o -name '*.jsonl' -o -name '*.json' | wc -l) files"
}

download_kiki_datasets() {
    echo "=== kiki-* HuggingFace datasets (clemsail) ==="
    local kiki_datasets=(
        "clemsail/kiki-embedded"
        "clemsail/kiki-electronics"
        "clemsail/kiki-esp32"
        "clemsail/kiki-kicad"
        "clemsail/kiki-kicad-pcb"
        "clemsail/kiki-stm32"
        "clemsail/kiki-iot"
        "clemsail/kiki-platformio"
        "clemsail/kiki-power"
        "clemsail/kiki-spice"
        "clemsail/kiki-spice-sim"
    )
    for ds in "${kiki_datasets[@]}"; do
        local name="${ds#clemsail/}"
        local dest="$RAW_DIR/$name"
        if [ -d "$dest" ] && [ "$(ls -A "$dest" 2>/dev/null)" ]; then
            echo "  $name already downloaded"
            continue
        fi
        echo "  Downloading $ds..."
        hf download "$ds" \
            --repo-type dataset \
            --local-dir "$dest" 2>/dev/null || echo "  WARNING: $ds not found or inaccessible, skipping"
    done
    echo "  Done."
}

download_existing_local() {
    echo "=== Linking existing local datasets ==="
    # final-opus-v3-1 (reasoning, general)
    if [ -d "$PROJECT_DIR/data/final-opus-v3-1" ]; then
        ln -sfn "$PROJECT_DIR/data/final-opus-v3-1" "$RAW_DIR/final-opus-v3-1"
        echo "  Linked final-opus-v3-1 ($(wc -l < "$PROJECT_DIR/data/final-opus-v3-1/train.jsonl") train)"
    fi
    # Opus reasoning datasets
    for ds in Opus-4.6-Reasoning-3000x-filtered Opus-4.6-reasoning-sft-12k claude-opus-4.6-10000x; do
        if [ -d "$PROJECT_DIR/data/$ds" ]; then
            ln -sfn "$PROJECT_DIR/data/$ds" "$RAW_DIR/$ds"
            echo "  Linked $ds"
        fi
    done
    echo "  Done."
}

print_usage() {
    echo "Usage: $0 <dataset|all>"
    echo ""
    echo "Datasets:"
    echo "  codefeedback        CodeFeedback-Filtered-Instruction (156K examples)"
    echo "  opencodereasoning   OpenCodeReasoning NVIDIA (735K examples)"
    echo "  magicoder           Magicoder-OSS-Instruct-75K"
    echo "  openhermes          OpenHermes-2.5 (general instruction, 1M)"
    echo "  kiki                kiki-* datasets from clemsail HuggingFace"
    echo "  local               Link existing local datasets"
    echo "  all                 Download everything"
}

case "${1:-help}" in
    codefeedback)       download_codefeedback ;;
    opencodereasoning)  download_opencodereasoning ;;
    magicoder)          download_magicoder ;;
    openhermes)         download_openhermes ;;
    kiki)               download_kiki_datasets ;;
    local)              download_existing_local ;;
    all)
        download_existing_local
        download_kiki_datasets
        download_codefeedback
        download_opencodereasoning
        download_magicoder
        download_openhermes
        ;;
    *)  print_usage ;;
esac

echo ""
echo "=== Raw data summary ==="
for d in "$RAW_DIR"/*/; do
    if [ -d "$d" ]; then
        name="$(basename "$d")"
        count=$(find "$d" -name '*.jsonl' -o -name '*.parquet' -o -name '*.json' 2>/dev/null | wc -l)
        echo "  $name: $count data files"
    fi
done

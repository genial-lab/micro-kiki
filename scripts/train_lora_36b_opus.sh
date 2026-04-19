#!/usr/bin/env bash
# V-35B-Opus: retrain regressed domains with Opus-enriched data
# Only retrains: chat-fr, reasoning, freecad, html-css
# Uses higher iters for foundations (500) vs niches (200)
set -euo pipefail

MODEL="models/Qwen3.6-35B-A3B"
DATA="data/micro-kiki"
OUTPUT="output/micro-kiki/lora-qwen36-35b-opus"
PYTHON="/opt/homebrew/bin/python3.12"

# Domains to retrain (the 4 regressions)
# chat-fr and reasoning get 500 iters (foundation), others get 200
declare -A DOMAIN_ITERS=(
  [chat-fr]=500
  [reasoning]=500
  [freecad]=200
  [html-css]=200
)

mkdir -p "$OUTPUT"

echo "================================================================"
echo "V-35B-Opus Training — ${#DOMAIN_ITERS[@]} regressed domains"
echo "Model: $MODEL | LR: 1e-5 | 8 layers | BF16"
echo "================================================================"

for domain in "${!DOMAIN_ITERS[@]}"; do
  iters="${DOMAIN_ITERS[$domain]}"
  adapter="$OUTPUT/$domain"

  [ -f "$adapter/adapters.safetensors" ] && echo "SKIP $domain (done)" && continue
  [ ! -f "$DATA/$domain/train.jsonl" ] && echo "SKIP $domain (no data)" && continue

  n=$(wc -l < "$DATA/$domain/train.jsonl")
  echo ""
  echo "=== $domain ($n examples, $iters iters) ==="

  $PYTHON -m mlx_lm lora \
    --model "$MODEL" \
    --data "$DATA/$domain" \
    --train \
    --iters "$iters" \
    --batch-size 1 \
    --learning-rate 1e-5 \
    --adapter-path "$adapter" \
    --max-seq-length 512 \
    --num-layers 8 \
    --steps-per-report 25 \
    --steps-per-eval 50 \
    --grad-checkpoint \
    --clear-cache-threshold 0.2 \
    2>&1 | tee "$OUTPUT/log-$domain.txt"

  echo "$domain DONE"
  sleep 10
done

echo "================================================================"
echo "ALL COMPLETE"
echo "================================================================"

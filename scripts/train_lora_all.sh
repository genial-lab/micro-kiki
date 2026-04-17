#!/usr/bin/env bash
# Train standard LoRA adapters for all 35 domains using mlx_lm lora
# Each domain gets its own adapter — the domain router selects which to load
set -euo pipefail

MODEL="models/Qwen3.5-4B"
DATA_BASE="data/micro-kiki"
OUTPUT_BASE="output/micro-kiki/lora-standard"
ITERS=500
BATCH=1
LR="2e-5"
MAX_SEQ=2048
PYTHON="/opt/homebrew/bin/python3.12"

CURRICULUM=(
  chat-fr reasoning python typescript cpp rust
  html-css shell sql yaml-json docker kicad-dsl spice lua-upy
  embedded stm32 iot freecad platformio power emc dsp
  spice-sim electronics kicad-pcb
  web-frontend web-backend music-audio devops llm-orch
  math security
  components llm-ops ml-training
)

mkdir -p "$OUTPUT_BASE"

echo "================================================================"
echo "LoRA Standard Training — ${#CURRICULUM[@]} domains"
echo "Model: $MODEL | LR: $LR | Iters: $ITERS | Batch: $BATCH"
echo "================================================================"

for i in "${!CURRICULUM[@]}"; do
  domain="${CURRICULUM[$i]}"
  idx=$((i + 1))
  data_dir="$DATA_BASE/$domain"
  adapter_dir="$OUTPUT_BASE/$domain"

  # Skip if no training data
  if [ ! -f "$data_dir/train.jsonl" ]; then
    echo "[$idx/${#CURRICULUM[@]}] SKIP $domain (no train.jsonl)"
    continue
  fi

  # Skip if already trained
  if [ -f "$adapter_dir/adapters.safetensors" ]; then
    echo "[$idx/${#CURRICULUM[@]}] SKIP $domain (already done)"
    continue
  fi

  # Count examples for adaptive iters
  n_examples=$(wc -l < "$data_dir/train.jsonl")
  # Adaptive: min(500, max(100, n/20))
  adaptive_iters=$(python3 -c "print(min($ITERS, max(100, $n_examples // 20)))")

  echo ""
  echo "================================================================"
  echo "[$idx/${#CURRICULUM[@]}] Training: $domain ($n_examples examples, $adaptive_iters iters)"
  echo "================================================================"

  $PYTHON -m mlx_lm lora \
    --model "$MODEL" \
    --data "$data_dir" \
    --train \
    --iters "$adaptive_iters" \
    --batch-size $BATCH \
    --learning-rate $LR \
    --adapter-path "$adapter_dir" \
    --max-seq-length $MAX_SEQ \
    --steps-per-report 50 \
    --steps-per-eval 100 \
    --grad-checkpoint \
    2>&1 | tee "$OUTPUT_BASE/log-$domain.txt"

  echo "[$idx/${#CURRICULUM[@]}] $domain DONE"
done

echo ""
echo "================================================================"
echo "ALL DOMAINS COMPLETE"
echo "Adapters: $OUTPUT_BASE/"
echo "================================================================"
ls -la "$OUTPUT_BASE"/*/adapters.safetensors 2>/dev/null | wc -l
echo "adapters trained"

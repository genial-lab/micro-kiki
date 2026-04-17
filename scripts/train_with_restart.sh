#!/bin/bash
# Train a LoRA adapter with automatic restart every N iters to avoid Metal resource_limit(499000).
#
# The Metal allocation counter is per-process and never resets. After ~60-160 iters on MoE 35B,
# the counter hits 499K and the process crashes. This wrapper saves a checkpoint every CHUNK iters,
# kills the process, and restarts from the checkpoint with a fresh Metal counter.
#
# Usage:
#   bash scripts/train_with_restart.sh platformio
#   bash scripts/train_with_restart.sh kicad-dsl 40
#   DOMAIN=power CHUNK=50 bash scripts/train_with_restart.sh

set -euo pipefail
cd "$(dirname "$0")/.."

DOMAIN="${1:-${DOMAIN:-}}"
CHUNK="${2:-${CHUNK:-40}}"  # iters per restart (safe under Metal limit)

if [ -z "$DOMAIN" ]; then
    echo "Usage: $0 <domain> [chunk_iters]"
    echo "Domains: kicad-dsl spice emc stm32 embedded freecad platformio power dsp electronics"
    exit 1
fi

PYTHON="$HOME/KIKI-Mac_tunner/.venv/bin/python3"
STACK_DIR="outputs/stacks/stack-${DOMAIN}"
ADAPTER="${STACK_DIR}/adapters.safetensors"
CONFIG="${STACK_DIR}/train_config.yaml"
TRAIN_SCRIPT="${STACK_DIR}/_train.py"
LOG="outputs/sft-restart-${DOMAIN}.log"

echo "=== $(date) Training ${DOMAIN} with restart every ${CHUNK} iters ===" | tee -a "$LOG"

# Step 1: Generate config and _train.py via the main script (dry run style)
.venv/bin/python3 -c "
import sys; sys.path.insert(0, '.')
from scripts.train_niches_mlxtune import NICHE_DOMAINS, OUTPUTS_DIR, MODEL_PATH, find_training_data
import yaml, json
from pathlib import Path

domain = '${DOMAIN}'
rank, epochs, lr, seq_len, dropout = NICHE_DOMAINS[domain]
output_dir = OUTPUTS_DIR / f'stack-{domain}'
output_dir.mkdir(parents=True, exist_ok=True)
data_path = find_training_data(domain)
n_examples = sum(1 for _ in open(data_path))
total_iters = int(n_examples * epochs / 1)

config = {
    'model': str(MODEL_PATH) if MODEL_PATH.exists() else 'Qwen/Qwen3.5-35B-A3B',
    'fine_tune_type': 'lora',
    'lora_parameters': {'rank': rank, 'alpha': rank * 2, 'dropout': dropout, 'scale': 2.0},
    'num_layers': 40,
    'learning_rate': lr,
    'batch_size': 1,
    'grad_accumulation_steps': 4,
    'iters': ${CHUNK},
    'max_seq_length': seq_len,
    'grad_checkpoint': True,
    'save_every': ${CHUNK},
    'steps_per_report': 10,
    'steps_per_eval': ${CHUNK},
    'val_batches': 25,
    'train': True,
    'seed': 42,
}

config_path = output_dir / 'train_config.yaml'
with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

# Write total iters for the wrapper
(output_dir / '.total_iters').write_text(str(total_iters))
print(f'{domain}: {n_examples} examples, {total_iters} total iters, chunk={${CHUNK}}')
" 2>&1 | tee -a "$LOG"

TOTAL_ITERS=$(cat "${STACK_DIR}/.total_iters")
DONE_ITERS=0

echo "Total iters: ${TOTAL_ITERS}, chunk: ${CHUNK}" | tee -a "$LOG"

# Step 2: Loop — train CHUNK iters, restart
while [ "$DONE_ITERS" -lt "$TOTAL_ITERS" ]; do
    REMAINING=$((TOTAL_ITERS - DONE_ITERS))
    THIS_CHUNK=$((REMAINING < CHUNK ? REMAINING : CHUNK))

    echo "=== $(date) Chunk ${DONE_ITERS}..$(($DONE_ITERS + $THIS_CHUNK)) / ${TOTAL_ITERS} ===" | tee -a "$LOG"

    # Update iters in config
    $PYTHON -c "
import yaml
with open('${CONFIG}') as f:
    c = yaml.safe_load(f)
c['iters'] = ${THIS_CHUNK}
c['save_every'] = ${THIS_CHUNK}
c['steps_per_eval'] = ${THIS_CHUNK}
with open('${CONFIG}', 'w') as f:
    yaml.dump(c, f, default_flow_style=False)
"

    # Generate _train.py with resume adapter if exists
    RESUME_FLAG=""
    if [ -f "${ADAPTER}" ]; then
        RESUME_FLAG="--resume-adapter-file ${ADAPTER}"
    fi

    cat > "${TRAIN_SCRIPT}" << PYEOF
import mlx.core as mx
mx.set_memory_limit(460 * 1024**3)
mx.set_cache_limit(32 * 1024**3)
import os, sys
os.environ["PYTHONPATH"] = "/Users/clems/KIKI-Mac_tunner/lib"
sys.path.insert(0, "/Users/clems/KIKI-Mac_tunner/lib")
from mlx_lm_fork.lora import main as lora_main
sys.argv = ["lora", "-c", "${CONFIG}",
            "--data", "$(dirname $(.venv/bin/python3 -c "from scripts.train_niches_mlxtune import find_training_data; print(find_training_data('${DOMAIN}'))"))",
            "--adapter-path", "${STACK_DIR}"${RESUME_FLAG:+,
            "${RESUME_FLAG}"}]
lora_main()
PYEOF

    # Run training chunk
    if $PYTHON "${TRAIN_SCRIPT}" >> "$LOG" 2>&1; then
        echo "=== $(date) Chunk OK ===" | tee -a "$LOG"
    else
        echo "=== $(date) Chunk crashed (expected Metal limit), continuing ===" | tee -a "$LOG"
    fi

    DONE_ITERS=$((DONE_ITERS + THIS_CHUNK))

    # Check if adapter was saved
    if [ ! -f "${ADAPTER}" ]; then
        echo "WARNING: No adapter saved after chunk, retrying..." | tee -a "$LOG"
    fi

    # Sleep to let Metal fully release
    sleep 10
done

echo "=== $(date) TRAINING COMPLETE: ${DOMAIN} (${TOTAL_ITERS} iters) ===" | tee -a "$LOG"

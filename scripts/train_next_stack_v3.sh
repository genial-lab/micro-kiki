#!/usr/bin/env bash
# train_next_stack_v3.sh — operator one-shot: train the next stack in the v3 curriculum.
#
# Reads configs/micro_kiki/brainstacks.yaml to determine curriculum order, inspects
# /Users/clems/KIKI-Mac_tunner/output/micro-kiki/lora-qwen36-35b-v3/ for already-trained
# stacks, picks the next one, writes its config, and kicks off mlx_lm lora with the
# prior stack's adapters as resume_adapter_file.
#
# Runs ON Mac Studio (sources ~/KIKI-Mac_tunner/.venv). Intended invocation via
# SSH chain: ssh grosmac 'ssh studio bash -s' <<< "$(cat scripts/train_next_stack_v3.sh)"
#
# Exit codes:
#   0 — training completed successfully
#   1 — curriculum exhausted (no next stack)
#   2 — dataset missing for next stack
#   3 — adapter health check failed on new adapter
#   4 — training process crashed
set -euo pipefail

REPO=/Users/clems/KIKI-Mac_tunner
OUT="$REPO/output/micro-kiki/lora-qwen36-35b-v3"
MODEL="models/Qwen3.6-35B-A3B"
DATA_ROOT="data/micro-kiki"

cd "$REPO"

# Parse curriculum order from brainstacks.yaml (relies on local rsync of the file).
BRAIN=/tmp/brainstacks-v3.yaml
if [ ! -f "$BRAIN" ]; then
    echo "ERROR: rsync configs/micro_kiki/brainstacks.yaml to $BRAIN first." >&2
    exit 3
fi

CURR=$(.venv/bin/python -c "
import yaml
c = yaml.safe_load(open('$BRAIN'))
print('\n'.join(c['curriculum']))
")

DONE=()
for dir in "$OUT"/*/; do
    name=$(basename "$dir")
    [ -f "$dir/adapters.safetensors" ] && DONE+=("$name")
done

NEXT=""
PRIOR=""
while IFS= read -r stack; do
    already=0
    for d in "${DONE[@]:-}"; do [ "$d" = "$stack" ] && already=1 && break; done
    if [ "$already" -eq 1 ]; then
        PRIOR="$stack"
        continue
    fi
    NEXT="$stack"
    break
done <<< "$CURR"

if [ -z "$NEXT" ]; then
    echo "Curriculum complete. ${#DONE[@]} stacks trained." >&2
    exit 1
fi

echo "=== Next stack: $NEXT (prior: ${PRIOR:-<none>}) ==="
[ -d "$DATA_ROOT/$NEXT" ] && [ -f "$DATA_ROOT/$NEXT/train.jsonl" ] || {
    echo "ERROR: dataset missing at $DATA_ROOT/$NEXT/train.jsonl" >&2
    exit 2
}

CFG="$OUT/config-$NEXT.yaml"
ADAPTER_PATH="$OUT/$NEXT"
mkdir -p "$ADAPTER_PATH"

# Determine rank from configs/mlx-per-domain/<stack>.yaml (foundations=32, niches=4/8/12/16)
RANK=$(.venv/bin/python -c "
import yaml, os
p = os.path.expanduser('~/micro-kiki-configs/mlx-per-domain/${NEXT}.yaml')
if os.path.exists(p):
    c = yaml.safe_load(open(p))
    print(c.get('lora_parameters', {}).get('rank', 32))
else:
    print(32)
" 2>/dev/null || echo 32)
ALPHA=$((RANK * 2))

cat > "$CFG" <<YAML
model: "$MODEL"
data: "$DATA_ROOT/$NEXT"
train: true
iters: 1000
batch_size: 1
learning_rate: 2e-5
adapter_path: "output/micro-kiki/lora-qwen36-35b-v3/$NEXT"
# No resume_adapter_file — niches train independently from base.
max_seq_length: 1024
num_layers: 20
steps_per_report: 50
steps_per_eval: 100
grad_checkpoint: true
lora_parameters:
  rank: $RANK
  alpha: $ALPHA
  dropout: 0.01
  scale: 20.0
YAML

echo "=== Config written: $CFG (rank=$RANK alpha=$ALPHA) ==="
cat "$CFG"

echo "=== Launching mlx_lm lora ==="
LOG="$OUT/log-$NEXT.txt"
.venv/bin/python -c "
import mlx.core as mx
mx.set_memory_limit(460 * 1024**3)
mx.set_cache_limit(32 * 1024**3)
import sys
sys.argv = ['mlx_lm', 'lora', '-c', '$CFG']
from mlx_lm.cli import main
main()
" > "$LOG" 2>&1

rc=$?
if [ $rc -ne 0 ]; then
    echo "ERROR: training crashed (rc=$rc). Tail:"
    tail -30 "$LOG"
    exit 4
fi

echo "=== Training complete. Health-checking $NEXT adapter ==="
if [ -f /Users/clems/tmp/validate_adapter_health.py ]; then
    .venv/bin/python /Users/clems/tmp/validate_adapter_health.py "$ADAPTER_PATH/adapters.safetensors" || exit 3
fi

echo "=== $NEXT trained and healthy. Next operator step: run forgetting gate ==="
echo "  python scripts/measure_forgetting.py \\"
echo "    --prior-adapter $OUT/${PRIOR:-$NEXT}/adapters.safetensors \\"
echo "    --new-adapter   $ADAPTER_PATH/adapters.safetensors \\"
echo "    --output results/gate-$NEXT.json"

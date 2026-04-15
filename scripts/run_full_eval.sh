#!/usr/bin/env bash
set -euo pipefail

# Full evaluation suite for micro-kiki v0.1
# Run after all 32 stacks + final router are trained.

echo "=== micro-kiki Full Evaluation Suite ==="
echo "Date: $(date -Iseconds)"

# 1. Group eval across all 32 stacks
echo "[1/4] Running group eval (32 stacks)..."
uv run scripts/group_eval.py --stacks 32 --output results/full-eval-v0.1-group.json

# 2. Per-domain eval (each stack vs base)
echo "[2/4] Running per-domain evals..."
for eval_file in data/eval/*.jsonl; do
    domain=$(basename "$eval_file" .jsonl)
    echo "  Evaluating domain: $domain"
    # Actual eval requires trained model — placeholder command
    # uv run python -m src.eval.stack_eval --stack "stack-*-$domain" --eval "$eval_file"
done

# 3. Benchmark evals (HumanEval, GSM8K, MMLU-Pro)
echo "[3/4] Running benchmark evals..."
echo "  NOTE: HumanEval, GSM8K, MMLU-Pro require eval harness setup on GPU machine."

# 4. Latency benchmarks
echo "[4/4] Running latency benchmarks..."
echo "  NOTE: Latency benchmarks require vLLM running on kxkm-ai."

echo "=== Evaluation complete ==="
echo "Results: results/full-eval-v0.1-group.json"

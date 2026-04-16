#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/run_forgetting.sh <new-stack-id> [--all-previous] [--results-file <path>]
# Runs forgetting check for the specified new stack against all previous stacks.
#
# Exit code 0 = passed, 1 = rollback triggered.
#
# Examples:
#   ./scripts/run_forgetting.sh stack-05 --results-file results/measurements.json
#   ./scripts/run_forgetting.sh stack-05 --all-previous --results-file results/measurements.json

if [ $# -lt 1 ]; then
    echo "Usage: $0 <new-stack-id> [--all-previous] [--results-file <path>]" >&2
    exit 1
fi

NEW_STACK_ID="$1"
shift

echo "=== Forgetting Check: ${NEW_STACK_ID} ==="
echo "Date: $(date -Iseconds)"

exec uv run python -m src.eval.forgetting "${NEW_STACK_ID}" "$@"

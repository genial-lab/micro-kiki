#!/usr/bin/env bash
# train_router_v5.sh — Chain script: wait for coding retrain → rebuild router data → retrain router V5
#
# Usage:
#   bash scripts/train_router_v5.sh [--no-wait]
#
# By default the script polls /tmp/coding-v2-retrain.log until the 5 coding adapters
# are done. Pass --no-wait to skip the wait (e.g. adapters already finished).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RETRAIN_LOG="/tmp/coding-v2-retrain.log"
ROUTER_DATA_DIR="${REPO_ROOT}/data/router-v4"
PYTHON="${REPO_ROOT}/.venv/bin/python"
LOG_DIR="${REPO_ROOT}/output/router-v5"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
SCRIPT_LOG="/tmp/router-v5-train-${TIMESTAMP}.log"

NO_WAIT=0
for arg in "$@"; do
  [[ "$arg" == "--no-wait" ]] && NO_WAIT=1
done

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${SCRIPT_LOG}"; }

# ---------------------------------------------------------------------------
# 1. Wait for the coding retrain to finish
# ---------------------------------------------------------------------------
if [[ "${NO_WAIT}" -eq 0 ]]; then
  log "Waiting for coding retrain to finish (watching ${RETRAIN_LOG}) ..."
  until grep -q "All 5 domains retrained" "${RETRAIN_LOG}" 2>/dev/null; do
    sleep 60
    log "  ... still waiting (last line: $(tail -1 "${RETRAIN_LOG}" 2>/dev/null || echo 'log not found yet'))"
  done
  log "Coding retrain finished — proceeding."
else
  log "--no-wait passed, skipping wait."
fi

# ---------------------------------------------------------------------------
# 2. Rebuild router data from scratch
#    Delete existing split → re-build from classified data → augment from clean LoRA files
# ---------------------------------------------------------------------------
log "Deleting existing router-v4 train/valid splits ..."
rm -f "${ROUTER_DATA_DIR}/train.jsonl" "${ROUTER_DATA_DIR}/valid.jsonl"

log "Rebuilding router data from classified domain data ..."
"${PYTHON}" "${REPO_ROOT}/scripts/build_router_data.py" 2>&1 | tee -a "${SCRIPT_LOG}"

log "Augmenting router data with clean coding LoRA data ..."
"${PYTHON}" "${REPO_ROOT}/scripts/augment_router_sparse.py" 2>&1 | tee -a "${SCRIPT_LOG}"

# Sanity check
TRAIN_LINES=$(wc -l < "${ROUTER_DATA_DIR}/train.jsonl")
VALID_LINES=$(wc -l < "${ROUTER_DATA_DIR}/valid.jsonl")
log "Router data ready: ${TRAIN_LINES} train / ${VALID_LINES} valid examples."

if [[ "${TRAIN_LINES}" -lt 1000 ]]; then
  log "ERROR: train split looks suspiciously small (${TRAIN_LINES} lines). Aborting."
  exit 1
fi

# ---------------------------------------------------------------------------
# 3. Train the router (V5)
# ---------------------------------------------------------------------------
mkdir -p "${LOG_DIR}"
log "Training router V5 (30 epochs) — output to ${LOG_DIR} ..."
"${PYTHON}" "${REPO_ROOT}/scripts/train_router_v4.py" \
  --mode train \
  --epochs 30 \
  2>&1 | tee -a "${SCRIPT_LOG}"

# ---------------------------------------------------------------------------
# 4. Evaluate
# ---------------------------------------------------------------------------
log "Evaluating router V5 ..."
"${PYTHON}" "${REPO_ROOT}/scripts/train_router_v4.py" \
  --mode eval \
  2>&1 | tee -a "${SCRIPT_LOG}"

# ---------------------------------------------------------------------------
# 5. Restart pipeline server
# ---------------------------------------------------------------------------
log "Restarting full pipeline server on port 9200 ..."
pkill -f "full_pipeline_server" 2>/dev/null || true
sleep 2
nohup "${PYTHON}" -m uvicorn \
  src.serving.full_pipeline_server:make_default_app \
  --factory \
  --host 127.0.0.1 \
  --port 9200 \
  > /tmp/kiki-server.log 2>&1 &
SERVER_PID=$!
log "Pipeline server started (PID ${SERVER_PID}). Logs → /tmp/kiki-server.log"

# Brief smoke-check: give the server 10 s to bind
sleep 10
if kill -0 "${SERVER_PID}" 2>/dev/null; then
  log "Server is alive — router V5 retrain chain complete."
  log "Full log: ${SCRIPT_LOG}"
else
  log "WARNING: Server process exited within 10 s — check /tmp/kiki-server.log"
  exit 1
fi

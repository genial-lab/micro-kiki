#!/usr/bin/env bash
#
# Setup the Studio neuroscience environment for SpikingBrain-7B.
# Story 13 of v0.3 plan.
#
# Usage:
#   bash scripts/setup_neuro_env.sh                # real install, local
#   bash scripts/setup_neuro_env.sh --dry-run      # print plan, no action
#   bash scripts/setup_neuro_env.sh --studio       # run remotely on Studio
#   bash scripts/setup_neuro_env.sh --help         # show help
#
# Idempotent: re-running re-uses an existing venv and only installs
# missing packages. Safe to Ctrl-C at any step.

set -euo pipefail

STUDIO_HOST="${STUDIO_HOST:-studio}"
STUDIO_VENV="${STUDIO_VENV:-/Users/clems/KIKI-Mac_tunner/.venv-neuroscience}"
MIN_FREE_GB="${MIN_FREE_GB:-20}"

DRY_RUN=0
REMOTE_STUDIO=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --studio)  REMOTE_STUDIO=1; shift ;;
    --help|-h)
      grep '^# ' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

log()  { printf '[setup-neuro] %s\n' "$*"; }
run()  {
  if [[ $DRY_RUN -eq 1 ]]; then
    printf '[dry-run] %s\n' "$*"
  else
    log "exec: $*"
    eval "$@"
  fi
}

if [[ $REMOTE_STUDIO -eq 1 ]]; then
  log "re-launching on Studio via ssh ${STUDIO_HOST}"
  SCRIPT_LOCAL="$(cd "$(dirname "$0")/.." && pwd)"
  SCRIPT_NAME="$(basename "$0")"
  if [[ $DRY_RUN -eq 1 ]]; then
    printf '[dry-run] ssh %s "cd %s && bash scripts/%s"\n' \
      "$STUDIO_HOST" "/Users/clems/KIKI-Mac_tunner" "$SCRIPT_NAME"
    exit 0
  fi
  # Assumes the repo is already checked out at that path on Studio.
  ssh "$STUDIO_HOST" \
    "cd /Users/clems/KIKI-Mac_tunner && bash scripts/${SCRIPT_NAME}"
  exit $?
fi

log "target venv: ${STUDIO_VENV}"
log "min free disk required: ${MIN_FREE_GB} GB"

# 1. Disk headroom check
if [[ "$(uname)" == "Darwin" ]]; then
  FREE_GB=$(df -g "$HOME" | awk 'NR==2 {print $4}')
else
  FREE_GB=$(df -BG "$HOME" | awk 'NR==2 {gsub("G","",$4); print $4}')
fi
log "free disk in HOME partition: ${FREE_GB} GB"
if [[ "${FREE_GB:-0}" -lt "$MIN_FREE_GB" ]]; then
  log "WARNING: free disk (${FREE_GB} GB) below ${MIN_FREE_GB} GB threshold"
fi

# 2. Venv existence
if [[ -d "$STUDIO_VENV" ]]; then
  log "venv exists, reusing: ${STUDIO_VENV}"
else
  log "venv missing, will create: ${STUDIO_VENV}"
  run "mkdir -p \"$(dirname \"$STUDIO_VENV\")\""
  run "uv venv \"$STUDIO_VENV\" --python 3.12"
fi

# 3. Install / refresh the `neuro` extra
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
log "repo root: ${REPO_ROOT}"
run "cd \"$REPO_ROOT\" && VIRTUAL_ENV=\"$STUDIO_VENV\" \
     uv pip install -e .[neuro]"

# 4. Verify imports + MPS availability
VERIFY_PY='
import sys
try:
    import torch
    import spikingjelly  # noqa: F401
    import transformers  # noqa: F401
    import modelscope    # noqa: F401
except Exception as e:
    print(f"IMPORT_FAIL: {e}", file=sys.stderr)
    sys.exit(1)
mps = bool(getattr(torch.backends, "mps", None)
           and torch.backends.mps.is_available())
print(f"torch={torch.__version__} mps={mps}")
sys.exit(0 if mps or sys.platform != "darwin" else 1)
'

if [[ $DRY_RUN -eq 1 ]]; then
  printf '[dry-run] python -c <verify>\n'
else
  log "verifying imports + MPS"
  "$STUDIO_VENV/bin/python" -c "$VERIFY_PY"
fi

# 5. ModelScope token presence (non-fatal)
if [[ -f "$HOME/.cache/modelscope/token" ]]; then
  log "modelscope token present at ~/.cache/modelscope/token"
else
  log "no modelscope login token found (OK for public Apache-2.0 repo)"
  log "  public-download fallback:"
  log "  modelscope download --model Abel2076/SpikingBrain-7B-W8ASpike"
fi

log "setup complete"

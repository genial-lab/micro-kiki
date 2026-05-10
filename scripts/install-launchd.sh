#!/usr/bin/env bash
# Install the micro-kiki launchd service for the current user.
#
# N3 H6 fix (2026-05-10): consolidated from 2 divergent plists with
# hardcoded user paths. This script templates the user + repo path
# at install time so the same plist works on any machine.
#
# Usage:
#   bash scripts/install-launchd.sh
#   bash scripts/install-launchd.sh --uninstall

set -euo pipefail

LABEL="com.electron.full-pipeline"
USER_HOME="$HOME"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PLIST_TEMPLATE="$REPO_DIR/launchd/com.electron.full-pipeline.plist.template"
TARGET_PLIST="$USER_HOME/Library/LaunchAgents/${LABEL}.plist"

if [ "${1:-}" = "--uninstall" ]; then
    launchctl unload "$TARGET_PLIST" 2>/dev/null || true
    rm -f "$TARGET_PLIST"
    echo "Uninstalled $LABEL"
    exit 0
fi

if [ ! -f "$PLIST_TEMPLATE" ]; then
    echo "ERROR: template not found at $PLIST_TEMPLATE" >&2
    exit 1
fi

# Substitute placeholders into the template
mkdir -p "$USER_HOME/Library/LaunchAgents"
sed -e "s|\${USER_HOME}|$USER_HOME|g" \
    -e "s|\${REPO_DIR}|$REPO_DIR|g" \
    "$PLIST_TEMPLATE" > "$TARGET_PLIST"

launchctl unload "$TARGET_PLIST" 2>/dev/null || true
launchctl load "$TARGET_PLIST"
echo "Installed $LABEL → $TARGET_PLIST"
echo "Status:"
launchctl list | grep "$LABEL" || echo "  (not yet running, check logs)"

#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INPUT="$(python3 -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "${1:?Usage: $0 <input.md>}")"
NAME="$(basename "${INPUT%.md}")"
OUTPUT="${2:-$SCRIPT_DIR/pdf/${NAME}.pdf}"
TMPTYP="/tmp/mk-paper-${NAME}.typ"
mkdir -p "$(dirname "$OUTPUT")"
pandoc "$INPUT" -t typst -o "${TMPTYP}.body"
{ cat "$SCRIPT_DIR/template.typ"; echo; cat "${TMPTYP}.body"; } > "$TMPTYP"
typst compile "$TMPTYP" "$OUTPUT" 2>&1
echo "wrote $OUTPUT ($(du -h "$OUTPUT" | cut -f1))"
rm -f "${TMPTYP}.body" "$TMPTYP"

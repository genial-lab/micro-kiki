#!/usr/bin/env bash
set -euo pipefail
export PATH="/Library/TeX/texbin:/usr/local/texlive/2026/bin/universal-darwin:$PATH"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INPUT="$(python3 -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "${1:?Usage: $0 <input.md>}")"
NAME="$(basename "${INPUT%.md}")"
OUTPUT="${2:-$SCRIPT_DIR/pdf/${NAME}-latex.pdf}"
mkdir -p "$(dirname "$OUTPUT")"
# Use pandoc's default LaTeX template (robust) + inject custom header for styling
pandoc "$INPUT" \
  --pdf-engine=xelatex \
  -V geometry:margin=2.2cm \
  -V fontsize=10pt \
  -V documentclass=article \
  -V colorlinks=true \
  -V linkcolor=Blue \
  -V urlcolor=Blue \
  -H "$SCRIPT_DIR/latex-header.tex" \
  -o "$OUTPUT" 2>&1 | tail -8
echo "wrote $OUTPUT ($(du -h "$OUTPUT" | cut -f1))"

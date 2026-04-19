#!/bin/bash
# Build the Paper A v2 draft PDF.
# Requires: MacTeX or equivalent LaTeX distribution with pdflatex + bibtex.
set -euo pipefail
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode paper-a-v2 || true
bibtex paper-a-v2 || true
pdflatex -interaction=nonstopmode paper-a-v2 || true
pdflatex -interaction=nonstopmode paper-a-v2
echo "Built: docs/paper-a/paper-a-v2.pdf"

#!/usr/bin/env bash
# Import domain datasets from KIKI-models-tuning to micro-kiki
set -euo pipefail
SRC="${1:-$HOME/Documents/Projets/Factory 4 Life/KIKI-models-tuning/datasets/processed}"
DST="data/distilled"
mkdir -p "${DST}"
for f in embedded stm32 iot freecad platformio power emc dsp spice kicad; do
  [[ -f "${SRC}/${f}_train.jsonl" ]] && cp "${SRC}/${f}_train.jsonl" "${DST}/${f}.jsonl" && echo "ok ${f}"
done
# Derived domains
cp "${DST}/embedded.jsonl" "${DST}/electronics.jsonl" 2>/dev/null
cp "${DST}/spice.jsonl" "${DST}/spice-sim.jsonl" 2>/dev/null
cp "${DST}/kicad.jsonl" "${DST}/kicad-pcb.jsonl" 2>/dev/null
echo "$(wc -l ${DST}/*.jsonl | tail -1) total"

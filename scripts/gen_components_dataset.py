#!/usr/bin/env python3
"""Generate `components` domain training dataset via teacher distillation.

Scope: electronic components — passives (resistors, capacitors, inductors),
active devices (diodes, transistors, op-amps), MCUs & datasheets, pin
assignments, packaging (SMD/TH/BGA), derating, ESR/tolerance/temp-coeff,
sourcing (LCSC/Mouser/DigiKey), equivalents & cross-references.

This is a SCAFFOLD. Actual Q&A pairs are produced by calling the local
Qwen3-Coder-480B teacher; if the teacher is unreachable the script writes a
placeholder README next to the output and exits 0 so the operator can run it
later with the teacher up.

For a large, hand-curated structured-data generator see the sibling
`gen_component_dataset.py` (singular) — that script covers a mostly disjoint
slice (pinouts, BOM, cross-refs) and should remain the primary generator
until the teacher seed is expanded here.

Usage::

    uv run python scripts/gen_components_dataset.py \\
        --output data/micro-kiki/components/raw.jsonl \\
        --limit 10 \\
        --teacher-url http://kxkm-ai:8000

TODO(operator): expand TOPIC_SEEDS to ~50 entries and raise --limit to 2000
once the teacher server is confirmed reachable.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

DOMAIN = "components"
DEFAULT_OUTPUT = f"data/micro-kiki/{DOMAIN}/raw.jsonl"
DEFAULT_TEACHER_URL = "http://localhost:8000"
DEFAULT_TEACHER_MODEL = "Qwen3-Coder-480B-A35B-Instruct"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Topic seed — expand to ~50 before the production run
# ---------------------------------------------------------------------------
TOPIC_SEEDS: list[str] = [
    "How do I read the tolerance band on a 4-band resistor?",
    "What is ESR and why does it matter for electrolytic capacitors?",
    "Compare MLCC X7R vs C0G dielectrics for a precision filter.",
    "How do I pick a Schottky diode for a low-voltage buck converter?",
    "What is the difference between a BJT saturation region and cutoff?",
    "How do I decouple a modern ESP32-S3 power rail properly?",
    "Recommend an op-amp for a 100kHz transimpedance amplifier photodiode input.",
    "What does 'derating' mean for a 0603 resistor at 85 C?",
    "How do I read a JEDEC package code like SOT-23-5 vs SOT-353?",
    "Cross-reference the LM358 with pin-compatible modern alternatives.",
    "What is the difference between electrolytic, tantalum and polymer caps?",
    "Pick a MOSFET for 5V gate drive switching a 12V, 3A solenoid.",
    "Explain the pinout of a 3-pin LDO regulator (LM1117-style).",
    "How do I select a crystal load capacitance for an MCU HSE oscillator?",
    "What is the typical Iq of modern low-power LDOs for battery designs?",
    "Compare common-cathode vs common-anode 7-segment displays.",
    "How do I pick a pull-up resistor for an I2C bus at 400 kHz?",
    "What does the 'Rds(on)' spec mean for a logic-level MOSFET?",
    "Cross-reference a 2N3904 NPN transistor with SMD equivalents.",
    "How do I choose a ferrite bead vs an inductor for power filtering?",
    # TODO(operator): extend to ~50 seeds covering inductors, op-amps,
    # voltage references, shunt regulators, optos, crystals, packaging.
]


PROMPT_TEMPLATE = """You are an expert electronics engineer writing training
data for an AI assistant that helps designers pick and use real components.

Answer the question below with:
- Direct, correct technical detail
- A short worked example or rule of thumb where appropriate
- References to concrete part numbers (LCSC/DigiKey) when relevant
- Stay under ~350 words

Question: {prompt}

Answer:"""


# ---------------------------------------------------------------------------
# Teacher probe + minimal client
# ---------------------------------------------------------------------------
def _probe_teacher(url: str, timeout: float = 3.0) -> bool:
    """Return True if teacher appears reachable. Best-effort; any error → False."""
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return False
    # Try /v1/models first (OpenAI-compatible), fall back to /health.
    for path in ("/v1/models", "/health", "/"):
        try:
            req = Request(url.rstrip("/") + path, method="GET")
            with urlopen(req, timeout=timeout) as resp:  # noqa: S310
                if resp.status < 500:
                    return True
        except Exception:  # noqa: BLE001
            continue
    return False


def _teacher_complete(url: str, model: str, prompt: str, timeout: float = 60.0) -> str:
    """Minimal OpenAI-compatible /v1/chat/completions call."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.3,
    }).encode("utf-8")
    req = Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"].strip()


def _write_placeholder(output: Path, teacher_url: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    readme = output.parent / "README.md"
    readme.write_text(
        f"# {DOMAIN} — dataset not yet generated\n\n"
        f"Teacher at `{teacher_url}` was unreachable when `gen_{DOMAIN}_dataset.py` "
        f"was invoked. Re-run with `--teacher-url <url>` pointing at the local "
        f"Qwen3-Coder-480B MLX server (typically `http://mac-studio:8000`) "
        f"once it is up:\n\n"
        f"```bash\n"
        f"uv run python scripts/gen_{DOMAIN}_dataset.py \\\n"
        f"    --output {output} \\\n"
        f"    --limit 2000 \\\n"
        f"    --teacher-url http://mac-studio:8000\n"
        f"```\n",
        encoding="utf-8",
    )
    # Create an empty jsonl so downstream scripts that `open()` it don't crash.
    if not output.exists():
        output.write_text("", encoding="utf-8")
    logger.warning("Teacher unreachable — wrote placeholder %s", readme)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"Generate {DOMAIN} training data via teacher distillation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path)
    parser.add_argument("--limit", type=int, default=10,
                        help="Number of seed prompts to distill (scaffold default is small).")
    parser.add_argument("--teacher-url", default=DEFAULT_TEACHER_URL)
    parser.add_argument("--teacher-model", default=DEFAULT_TEACHER_MODEL)
    args = parser.parse_args()

    output: Path = args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    if not _probe_teacher(args.teacher_url):
        _write_placeholder(output, args.teacher_url)
        return 0

    seeds = TOPIC_SEEDS[: args.limit]
    logger.info("Distilling %d %s prompts via teacher %s", len(seeds), DOMAIN, args.teacher_url)

    written = 0
    with output.open("w", encoding="utf-8") as fh:
        for idx, seed in enumerate(seeds, start=1):
            try:
                answer = _teacher_complete(
                    args.teacher_url, args.teacher_model,
                    PROMPT_TEMPLATE.format(prompt=seed),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("[%d/%d] teacher call failed: %s", idx, len(seeds), exc)
                continue
            record = {"messages": [
                {"role": "user", "content": seed},
                {"role": "assistant", "content": answer},
            ], "domain": DOMAIN}
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            logger.info("[%d/%d] OK (%d chars)", idx, len(seeds), len(answer))

    logger.info("Wrote %d examples to %s", written, output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

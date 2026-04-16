#!/usr/bin/env python3
"""Distill python dataset (2K examples) from a teacher LLM.

Reads seed prompts from ``data/prompts/python.jsonl``, sends each to the
teacher via the OpenAI-compatible HTTP endpoint, and writes completed
examples to ``data/distilled/python.jsonl``.  Supports checkpoint resume
(handled by ``generate_examples``).

Usage::

    uv run python scripts/distill_python.py \\
        --teacher-url http://localhost:8000 \\
        --teacher-model qwen3.5-35b-opus

The script multiplies seed prompts via ``n_per_prompt`` to reach the
target example count.  For 300 seeds targeting 2000 examples, each prompt
produces ~7 completions (with distinct ``sample_idx`` hashes).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Allow direct invocation (uv run python scripts/distill_python.py) in
# addition to module invocation (uv run python -m scripts.distill_python).
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.distill.generator import GeneratorConfig, generate_examples
from src.distill.teacher_client import GenerateParams, TeacherClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_TEACHER_URL = "http://localhost:8000"
DEFAULT_TEACHER_MODEL = "qwen3.5-35b-opus"
DEFAULT_PROMPTS = Path("data/prompts/python.jsonl")
DEFAULT_OUTPUT = Path("data/distilled/python.jsonl")
DEFAULT_MAX_EXAMPLES = 2000


# ---------------------------------------------------------------------------
# Sync adapter — bridges async TeacherClient to sync TeacherProtocol
# ---------------------------------------------------------------------------


class SyncTeacherAdapter:
    """Wrap :class:`TeacherClient` into the synchronous :class:`TeacherProtocol`.

    ``generate_examples`` calls ``teacher.complete(prompt, **params)``
    synchronously.  Uses a persistent event loop to avoid the
    ``Event loop is closed`` error from ``asyncio.run()`` re-creating
    loops on each call (httpx loses its connection pool).
    """

    def __init__(self, client: TeacherClient, model: str, params: GenerateParams) -> None:
        self._client = client
        self.model = model
        self._params = params
        self._loop = asyncio.new_event_loop()

    def complete(self, prompt: str, **_params: Any) -> str:
        """Synchronous completion via the teacher HTTP endpoint."""
        return self._loop.run_until_complete(
            self._client.generate(prompt, self.model, params=self._params)
        )


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


def load_seed_prompts(path: Path) -> list[str]:
    """Load prompt strings from a JSONL file.

    Each line must have a ``"prompt"`` key.  Blank / malformed lines are
    skipped with a warning.
    """
    if not path.exists():
        logger.error("Seed prompts file not found: %s", path)
        sys.exit(1)

    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON at line %d in %s", lineno, path)
                continue
            prompt = entry.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                logger.warning("Missing or empty 'prompt' at line %d in %s", lineno, path)
                continue
            prompts.append(prompt)

    logger.info("Loaded %d seed prompts from %s", len(prompts), path)
    return prompts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distill python dataset from a teacher LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--teacher-url",
        default=DEFAULT_TEACHER_URL,
        help="Base URL of the teacher's OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--teacher-model",
        default=DEFAULT_TEACHER_MODEL,
        help="Model name to pass in the API request.",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=DEFAULT_PROMPTS,
        help="Path to the seed prompts JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSONL path for distilled examples.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=DEFAULT_MAX_EXAMPLES,
        help="Target number of completed examples.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for the teacher.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max tokens per teacher completion.",
    )
    args = parser.parse_args()

    # -- Load seed prompts ---------------------------------------------------
    prompts = load_seed_prompts(args.prompts)
    if not prompts:
        logger.error("No valid prompts found.  Aborting.")
        sys.exit(1)

    # -- Compute n_per_prompt to reach target --------------------------------
    n_per_prompt = max(1, args.max_examples // len(prompts))
    total_target = n_per_prompt * len(prompts)
    logger.info(
        "Targeting %d examples: %d seeds x %d completions each = %d",
        args.max_examples,
        len(prompts),
        n_per_prompt,
        total_target,
    )

    # -- Build teacher adapter -----------------------------------------------
    client = TeacherClient(
        endpoints={args.teacher_model: args.teacher_url},
    )
    gen_params = GenerateParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    teacher = SyncTeacherAdapter(client, args.teacher_model, gen_params)

    # -- Configure generator -------------------------------------------------
    config = GeneratorConfig(
        n_per_prompt=n_per_prompt,
        max_retries=3,
        retry_backoff_s=1.0,
        domain="python",
        params=gen_params.to_dict(),
    )

    # -- Run distillation ----------------------------------------------------
    logger.info("Starting distillation -> %s", args.output)
    stats = generate_examples(
        prompts=prompts,
        teacher=teacher,
        output_path=args.output,
        config=config,
    )
    logger.info(
        "Done.  generated=%d  skipped=%d  failed=%d",
        stats["generated"],
        stats["skipped"],
        stats["failed"],
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()

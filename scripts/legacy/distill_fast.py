#!/usr/bin/env python3
"""Fast distillation script for reasoning and python domains.

Uses the teacher on localhost:8000 (Qwen3.5-35B llama.cpp).
Handles Qwen3 thinking tags in responses.
Designed for kxkm-ai execution with existing venv.

Usage:
    cd /home/kxkm/micro-kiki
    UNSLOTH_COMPILE_DISABLE=1 /home/kxkm/KIKI-models-tuning/.venv/bin/python \
        scripts/distill_fast.py --domain reasoning --max-examples 1200
    UNSLOTH_COMPILE_DISABLE=1 /home/kxkm/KIKI-models-tuning/.venv/bin/python \
        scripts/distill_fast.py --domain python --max-examples 1200
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TEACHER_URL = "http://localhost:8000/v1/chat/completions"
TEACHER_MODEL = "Qwen3.5-35B-A3B-UD-Q3_K_XL.gguf"


def call_teacher(prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> str:
    """Call teacher and return completion text. Disables thinking for direct answers."""
    payload = {
        "model": TEACHER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    resp = httpx.post(TEACHER_URL, json=payload, timeout=180.0)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    if not content:
        # Fallback: try reasoning_content if content is empty
        content = resp.json()["choices"][0]["message"].get("reasoning_content", "")
    # Strip any remaining thinking tags
    if "<think>" in content:
        content = content.split("</think>")[-1].strip()
    return content


def load_existing(path: Path) -> tuple[list[dict], set[str]]:
    """Load existing distilled data, return non-empty records and their prompt set."""
    records = []
    prompts = set()
    if path.exists():
        for line in path.read_text().strip().split("\n"):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if rec.get("completion", "").strip():
                    records.append(rec)
                    prompts.add(rec["prompt"])
            except json.JSONDecodeError:
                continue
    return records, prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, choices=["reasoning", "python"])
    parser.add_argument("--max-examples", type=int, default=1200)
    parser.add_argument("--n-per-prompt", type=int, default=0, help="0=auto")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    prompts_path = Path(f"data/prompts/{args.domain}.jsonl")
    output_path = Path(f"data/distilled/{args.domain}.jsonl")

    # Load seed prompts
    seeds = []
    for line in prompts_path.read_text().strip().split("\n"):
        if line.strip():
            seeds.append(json.loads(line)["prompt"])
    logger.info("Loaded %d seed prompts from %s", len(seeds), prompts_path)

    # Load existing valid completions
    existing_records, existing_prompts = load_existing(output_path)
    logger.info("Existing valid completions: %d", len(existing_records))

    # Calculate how many more we need
    needed = args.max_examples - len(existing_records)
    if needed <= 0:
        logger.info("Already have %d examples (target %d), nothing to do",
                     len(existing_records), args.max_examples)
        return

    n_per_prompt = args.n_per_prompt if args.n_per_prompt > 0 else max(1, needed // len(seeds) + 1)
    logger.info("Need %d more examples, %d per prompt, %d seeds", needed, n_per_prompt, len(seeds))

    # Rewrite output with only valid records, then append new ones
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rec in existing_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    generated = 0
    failed = 0
    with open(output_path, "a") as f:
        for si, seed in enumerate(seeds):
            if generated >= needed:
                break
            for sample_idx in range(n_per_prompt):
                if generated >= needed:
                    break
                # Skip if we already have this exact prompt+idx
                prompt_key = f"{seed}::{sample_idx}"
                try:
                    t0 = time.time()
                    # Add variation for multi-sample
                    actual_prompt = seed
                    if sample_idx > 0:
                        actual_prompt = f"{seed}\n\nProvide a different approach or perspective."
                    completion = call_teacher(actual_prompt, args.temperature, args.max_tokens)
                    elapsed = time.time() - t0

                    if not completion.strip():
                        logger.warning("Empty completion for: %s", seed[:60])
                        failed += 1
                        continue

                    rec = {
                        "prompt": seed,
                        "completion": completion,
                        "teacher_model": TEACHER_MODEL,
                        "domain": args.domain,
                        "sample_idx": sample_idx,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    generated += 1

                    if generated % 10 == 0:
                        logger.info("Generated %d/%d (failed=%d) [%.1fs/call] seed %d/%d",
                                    generated, needed, failed, elapsed, si + 1, len(seeds))
                except Exception as e:
                    logger.warning("Failed on prompt '%s': %s", seed[:60], e)
                    failed += 1
                    time.sleep(2)

    # Count final
    with open(output_path) as f:
        total = sum(1 for l in f if l.strip())
    logger.info("=" * 60)
    logger.info("DISTILL DONE: generated=%d failed=%d total=%d", generated, failed, total)
    logger.info("Output: %s", output_path)


if __name__ == "__main__":
    main()

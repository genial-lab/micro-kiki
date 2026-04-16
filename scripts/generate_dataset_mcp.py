#!/usr/bin/env python3
"""Generate training datasets using Claude CLI — v2 with rate limiting.

Handles:
- Rate limiting with adaptive delay between calls
- Timeout with retry (3 attempts, increasing timeout)
- Exponential backoff on failures
- Progress checkpoint (resume from where it stopped)
- Token budget tracking (estimate ~2000 tokens/call)
- Batch size control

Usage:
    python3 scripts/generate_dataset_mcp.py --domain kicad-dsl --max 100
    python3 scripts/generate_dataset_mcp.py --all --max 50 --delay 5
    python3 scripts/generate_dataset_mcp.py --all --budget 500000
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# os.chdir handled by caller
OUTPUT_ROOT = Path("data/mcp-generated")
EXPANDED_ROOT = Path("data/prompts-expanded")
CLAUDE = "claude"

# Rate limiting config
DEFAULT_DELAY = 3.0        # seconds between calls (conservative)
MAX_RETRIES = 3            # retries per prompt
BASE_TIMEOUT = 300          # seconds, increases per retry
BACKOFF_MULTIPLIER = 2.0   # exponential backoff on failure
TOKENS_PER_CALL = 2000     # rough estimate for budget tracking
COOLDOWN_AFTER_ERROR = 30  # seconds after rate limit / server error


def load_prompts(domain: str, max_n: int) -> list[str]:
    """Load prompts from expanded templates + MCP extras."""
    prompts: list[str] = []
    seen: set[str] = set()

    # Expanded prompts (parametric templates)
    expanded = EXPANDED_ROOT / f"{domain}.jsonl"
    if expanded.exists():
        with open(expanded) as f:
            for line in f:
                d = json.loads(line.strip())
                p = d.get("prompt", "")
                if p and p not in seen:
                    prompts.append(p)
                    seen.add(p)

    # MCP extra prompts
    mcp_extra = Path("data/mcp_extra_prompts.json")
    if mcp_extra.exists():
        data = json.load(open(mcp_extra))
        for p in data.get(domain, []):
            if p not in seen:
                prompts.append(p)
                seen.add(p)

    return prompts[:max_n]


def generate_one(prompt: str, retries: int = MAX_RETRIES) -> str | None:
    """Call Claude CLI with retry and adaptive timeout."""
    for attempt in range(retries):
        timeout = BASE_TIMEOUT * (attempt + 1)  # 300, 600, 900  # 90, 180, 270
        try:
            result = subprocess.run(
                [CLAUDE, "--print", "-p", prompt],
                capture_output=True, text=True, timeout=timeout,
                cwd=".",
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()

            stderr = result.stderr.lower()
            if "rate" in stderr or "limit" in stderr or "429" in stderr:
                logger.warning("Rate limited, cooling down %ds...", COOLDOWN_AFTER_ERROR)
                time.sleep(COOLDOWN_AFTER_ERROR)
                continue
            if "503" in stderr or "overloaded" in stderr:
                logger.warning("Server overloaded, cooling down %ds...", COOLDOWN_AFTER_ERROR)
                time.sleep(COOLDOWN_AFTER_ERROR)
                continue

            logger.warning("Attempt %d failed (exit %d)", attempt + 1, result.returncode)

        except subprocess.TimeoutExpired:
            logger.warning("Timeout (%ds) attempt %d/%d", timeout, attempt + 1, retries)
        except Exception as e:
            logger.warning("Error attempt %d: %s", attempt + 1, e)

        # Backoff between retries
        backoff = DEFAULT_DELAY * (BACKOFF_MULTIPLIER ** attempt)
        time.sleep(backoff)

    return None


def count_existing(out_file: Path) -> int:
    if out_file.exists():
        return sum(1 for _ in open(out_file))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate training data via Claude CLI with rate limiting.",
    )
    parser.add_argument("--domain", help="Single domain")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--max", type=int, default=100, help="Max prompts per domain")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY,
                        help="Seconds between API calls (default 3)")
    parser.add_argument("--budget", type=int, default=0,
                        help="Max estimated tokens (0 = unlimited)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    all_domains = ["kicad-dsl", "spice", "emc", "stm32", "embedded",
                   "power", "dsp", "electronics", "freecad", "platformio"]
    domains = [args.domain] if args.domain else (all_domains if args.all else [])
    if not domains:
        parser.print_help()
        return

    tokens_used = 0
    total_generated = 0
    total_failed = 0

    for domain in domains:
        prompts = load_prompts(domain, args.max)
        if not prompts:
            logger.warning("No prompts for %s", domain)
            continue

        out_dir = OUTPUT_ROOT / domain
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "train.jsonl"
        existing = count_existing(out_file)

        if args.dry_run:
            logger.info("%s: %d prompts, %d existing, %d to generate",
                        domain, len(prompts), existing, max(0, len(prompts) - existing))
            continue

        logger.info("=== %s: %d prompts (skip %d existing) ===", domain, len(prompts), existing)

        domain_generated = 0
        domain_failed = 0

        with open(out_file, "a") as f:
            for i, prompt in enumerate(prompts):
                if i < existing:
                    continue

                # Budget check
                if args.budget > 0 and tokens_used >= args.budget:
                    logger.info("Token budget exhausted (%d/%d)", tokens_used, args.budget)
                    break

                response = generate_one(prompt)

                if response:
                    example = {
                        "messages": [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": response},
                        ],
                        "domain": domain,
                        "source": "claude-mcp-v2",
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    }
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
                    f.flush()
                    domain_generated += 1
                    tokens_used += TOKENS_PER_CALL
                else:
                    domain_failed += 1

                # Progress report
                if (i + 1) % 10 == 0 or (i + 1) == len(prompts):
                    rate = domain_generated / max(1, (time.time() - t0)) * 60 if 't0' in dir() else 0
                    logger.info("%s: %d/%d done, %d failed, ~%d tokens used",
                                domain, i + 1, len(prompts), domain_failed, tokens_used)

                # Rate limiting delay
                time.sleep(args.delay)

        total_generated += domain_generated
        total_failed += domain_failed
        total = count_existing(out_file)
        logger.info("DONE %s: %d new + %d existing = %d total (%d failed)",
                    domain, domain_generated, existing, total, domain_failed)

    logger.info("=== SUMMARY: %d generated, %d failed, ~%d tokens ===",
                total_generated, total_failed, tokens_used)


if __name__ == "__main__":
    t0 = time.time()
    main()

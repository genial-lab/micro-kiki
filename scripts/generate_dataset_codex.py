#!/usr/bin/env python3
"""Generate training datasets using Codex CLI — with rate limiting.

Handles:
- Rate limiting with adaptive delay between calls
- Timeout with retry (3 attempts, increasing timeout)
- Exponential backoff on failures
- Progress checkpoint (resume from where it stopped)

Usage:
    python3 scripts/generate_dataset_codex.py --domain kicad-dsl --max 100
    python3 scripts/generate_dataset_codex.py --all --max 50 --delay 3
    python3 scripts/generate_dataset_codex.py --domain stm32 --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = REPO_ROOT / "data" / "codex-generated"
EXPANDED_ROOT = REPO_ROOT / "data" / "prompts-expanded"

# Rate limiting config
DEFAULT_DELAY = 2.0         # seconds between calls
MAX_RETRIES = 3             # retries per prompt
BASE_TIMEOUT = 300          # seconds; increases per retry
BACKOFF_MULTIPLIER = 2.0    # exponential backoff on failure
COOLDOWN_AFTER_ERROR = 30   # seconds after rate-limit / server error

ALL_DOMAINS = [
    "kicad-dsl", "spice", "emc", "stm32", "embedded",
    "power", "dsp", "electronics", "freecad", "platformio",
]


def load_prompts(domain: str, max_n: int) -> list[str]:
    """Load prompts from expanded templates."""
    prompts: list[str] = []
    seen: set[str] = set()

    expanded = EXPANDED_ROOT / f"{domain}.jsonl"
    if expanded.exists():
        with open(expanded) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                p = d.get("prompt", "")
                if p and p not in seen:
                    prompts.append(p)
                    seen.add(p)
    else:
        logger.warning("No expanded prompts found for domain '%s' at %s", domain, expanded)

    return prompts[:max_n]


def generate_one(prompt: str, retries: int = MAX_RETRIES) -> str | None:
    """Call Codex CLI with retry and adaptive timeout."""
    for attempt in range(retries):
        timeout = BASE_TIMEOUT * (attempt + 1)  # 300, 600, 900
        try:
            out_tmp = REPO_ROOT / "_codex_out.txt"
            out_tmp.unlink(missing_ok=True)
            result = subprocess.run(
                ["codex", "exec",
                 "-o", str(out_tmp), prompt],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                cwd=str(REPO_ROOT),
            )
            if out_tmp.exists():
                text = out_tmp.read_text().strip()
                out_tmp.unlink(missing_ok=True)
                if text:
                    return text
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

            logger.warning(
                "Attempt %d failed (exit %d): %s",
                attempt + 1,
                result.returncode,
                result.stderr[:200],
            )

        except subprocess.TimeoutExpired:
            logger.warning("Timeout (%ds) attempt %d/%d", timeout, attempt + 1, retries)
        except FileNotFoundError:
            logger.error(
                "codex CLI not found. Install it with: npm install -g @openai/codex"
            )
            return None
        except Exception as e:
            logger.warning("Error attempt %d: %s", attempt + 1, e)

        backoff = DEFAULT_DELAY * (BACKOFF_MULTIPLIER**attempt)
        time.sleep(backoff)

    return None


def count_existing(out_file: Path) -> int:
    if out_file.exists():
        with open(out_file) as f:
            return sum(1 for _ in f)
    return 0


def process_domain(domain: str, args: argparse.Namespace) -> tuple[int, int]:
    """Process a single domain; returns (generated, failed)."""
    prompts = load_prompts(domain, args.max)
    if not prompts:
        logger.warning("No prompts for %s", domain)
        return 0, 0

    out_dir = OUTPUT_ROOT / domain
    out_file = out_dir / "train.jsonl"
    existing = count_existing(out_file)

    if args.dry_run:
        logger.info(
            "%s: %d prompts, %d existing, %d to generate",
            domain,
            len(prompts),
            existing,
            max(0, len(prompts) - existing),
        )
        return 0, 0

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=== %s: %d prompts (skip %d existing) ===", domain, len(prompts), existing)

    generated = 0
    failed = 0
    t0 = time.time()

    with open(out_file, "a") as f:
        for i, prompt in enumerate(prompts):
            if i < existing:
                continue

            response = generate_one(prompt)

            if response:
                example = {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response},
                    ],
                    "domain": domain,
                    "source": "codex-generated",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                f.flush()
                generated += 1
            else:
                failed += 1

            if (i + 1) % 10 == 0 or (i + 1) == len(prompts):
                elapsed = time.time() - t0
                rate = generated / elapsed * 60 if elapsed > 0 else 0
                logger.info(
                    "%s: %d/%d done, %d failed, %.1f/min",
                    domain,
                    i + 1,
                    len(prompts),
                    failed,
                    rate,
                )

            time.sleep(args.delay)

    total = count_existing(out_file)
    logger.info(
        "DONE %s: %d new + %d existing = %d total (%d failed)",
        domain,
        generated,
        existing,
        total,
        failed,
    )
    return generated, failed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate training data via Codex CLI (o3-mini) with rate limiting.",
    )
    parser.add_argument("--domain", help="Single domain to process")
    parser.add_argument("--all", action="store_true", help="Process all domains")
    parser.add_argument("--max", type=int, default=100, help="Max prompts per domain")
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help="Seconds between API calls (default 2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without calling codex",
    )
    args = parser.parse_args()

    domains: list[str] = []
    if args.domain:
        domains = [args.domain]
    elif args.all:
        domains = ALL_DOMAINS
    else:
        parser.print_help()
        return

    total_generated = 0
    total_failed = 0

    for domain in domains:
        gen, fail = process_domain(domain, args)
        total_generated += gen
        total_failed += fail

    if not args.dry_run:
        logger.info(
            "=== SUMMARY: %d generated, %d failed ===", total_generated, total_failed
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate DPO preference pairs for niche domain fine-tuning.

For each niche domain, loads eval prompts from data/merged/<domain>/train.jsonl,
generates responses from the SFT model and the 480B judge, then uses the judge
to score and produce chosen/rejected pairs.

Usage:
    uv run scripts/generate_dpo_pairs.py --domain kicad-dsl --dry-run
    uv run scripts/generate_dpo_pairs.py --all --max-pairs 500
    uv run scripts/generate_dpo_pairs.py --domain spice \\
        --judge-url http://localhost:8481 \\
        --sft-url http://localhost:8200
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.distill.teacher_client import GenerateParams, TeacherClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MERGED_DATA = PROJECT_ROOT / "data" / "merged"
DPO_DATA = PROJECT_ROOT / "data" / "dpo"

NICHE_DOMAINS: list[str] = sorted([
    "kicad-dsl", "spice", "emc", "stm32", "embedded",
    "freecad", "platformio", "power", "dsp", "electronics",
])

DEFAULT_JUDGE_URL = "http://localhost:8481"
DEFAULT_SFT_URL = "http://localhost:8200"
DEFAULT_JUDGE_MODEL = "qwen3-coder-480b"
DEFAULT_SFT_MODEL = "sft-niche"
DEFAULT_MAX_PAIRS = 500

# System prompt for judging which response is better
JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator for technical AI responses.
Given a prompt and two responses (A and B), decide which response is better.
Reply with exactly one word: "A" or "B".
Provide no explanation."""

JUDGE_PROMPT_TEMPLATE = """\
Prompt:
{prompt}

Response A:
{response_a}

Response B:
{response_b}

Which response is better? Reply with "A" or "B" only."""

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_prompts(domain: str, max_pairs: int) -> list[str]:
    """Load prompts from data/merged/<domain>/train.jsonl."""
    data_path = MERGED_DATA / domain / "train.jsonl"
    if not data_path.exists():
        logger.warning("No merged data for %s at %s", domain, data_path)
        return []

    prompts: list[str] = []
    with data_path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Bad JSON line %d in %s", lineno, data_path)
                continue
            prompt = entry.get("prompt") or entry.get("instruction") or entry.get("input", "")
            if isinstance(prompt, str) and prompt.strip():
                prompts.append(prompt.strip())
            if len(prompts) >= max_pairs:
                break

    logger.info("Loaded %d prompts for %s from %s", len(prompts), domain, data_path)
    return prompts


# ---------------------------------------------------------------------------
# Dry-run synthetic pair generation
# ---------------------------------------------------------------------------


def _synthetic_pair(prompt: str, domain: str, idx: int) -> dict[str, Any]:
    """Generate a deterministic placeholder pair for dry-run testing."""
    return {
        "prompt": prompt,
        "chosen": (
            f"[DRY-RUN chosen] Detailed technical answer for {domain} "
            f"example {idx}: {prompt[:80]}"
        ),
        "rejected": (
            f"[DRY-RUN rejected] Incomplete answer for {domain} "
            f"example {idx}: {prompt[:40]}"
        ),
        "domain": domain,
    }


def dry_run_domain(domain: str, max_pairs: int) -> list[dict[str, Any]]:
    """Generate synthetic pairs without calling any model."""
    prompts = load_prompts(domain, max_pairs)
    if not prompts:
        # Fall back to generic placeholder prompts so the pipeline can be tested
        prompts = [f"Example {domain} question {i}" for i in range(min(5, max_pairs))]
        logger.info("Using %d placeholder prompts for dry-run of %s", len(prompts), domain)

    pairs = [_synthetic_pair(p, domain, i) for i, p in enumerate(prompts)]
    logger.info("DRY RUN — %d synthetic pairs for %s", len(pairs), domain)
    return pairs


# ---------------------------------------------------------------------------
# Live pair generation
# ---------------------------------------------------------------------------


async def generate_response(
    client: TeacherClient,
    model: str,
    prompt: str,
    params: GenerateParams,
) -> str | None:
    """Call model and return text, or None on error."""
    try:
        return await client.generate(prompt, model, params=params, use_cache=True)
    except Exception as exc:
        logger.warning("generate_response failed for model=%s: %s", model, exc)
        return None


async def judge_responses(
    judge_client: TeacherClient,
    judge_model: str,
    prompt: str,
    response_a: str,
    response_b: str,
) -> str | None:
    """Ask the 480B judge which response is better. Returns 'A', 'B', or None."""
    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
        prompt=prompt,
        response_a=response_a,
        response_b=response_b,
    )
    params = GenerateParams(temperature=0.0, max_tokens=4, thinking=False)
    try:
        verdict = await judge_client.generate(
            judge_prompt, judge_model, params=params, use_cache=True
        )
        verdict = verdict.strip().upper()
        if verdict in ("A", "B"):
            return verdict
        # Tolerate "A." or "B." punctuation
        if verdict.startswith("A"):
            return "A"
        if verdict.startswith("B"):
            return "B"
        logger.warning("Unexpected judge verdict: %r", verdict)
        return None
    except Exception as exc:
        logger.warning("judge_responses failed: %s", exc)
        return None


async def generate_pairs_for_domain(
    domain: str,
    prompts: list[str],
    judge_client: TeacherClient,
    sft_client: TeacherClient,
    judge_model: str,
    sft_model: str,
) -> list[dict[str, Any]]:
    """Generate DPO pairs for a single domain using both models + judge."""
    sft_params = GenerateParams(temperature=0.7, max_tokens=1024)
    judge_gen_params = GenerateParams(temperature=0.3, max_tokens=1024, thinking=False)

    pairs: list[dict[str, Any]] = []

    for idx, prompt in enumerate(prompts):
        logger.debug("Processing prompt %d/%d for %s", idx + 1, len(prompts), domain)

        # Generate SFT response and judge response concurrently
        sft_resp, judge_resp = await asyncio.gather(
            generate_response(sft_client, sft_model, prompt, sft_params),
            generate_response(judge_client, judge_model, prompt, judge_gen_params),
        )

        if sft_resp is None or judge_resp is None:
            logger.warning("Skipping prompt %d — missing response(s)", idx + 1)
            continue

        # Randomise A/B assignment to avoid position bias
        if random.random() < 0.5:
            response_a, response_b = sft_resp, judge_resp
            a_is_sft = True
        else:
            response_a, response_b = judge_resp, sft_resp
            a_is_sft = False

        verdict = await judge_responses(
            judge_client, judge_model, prompt, response_a, response_b
        )
        if verdict is None:
            logger.warning("Skipping prompt %d — judge gave no verdict", idx + 1)
            continue

        # Map verdict back to chosen/rejected
        # chosen = judge model response (higher quality)
        # rejected = SFT model response (what we want to improve upon)
        judge_won = (verdict == "A" and not a_is_sft) or (verdict == "B" and a_is_sft)
        if judge_won:
            chosen, rejected = judge_resp, sft_resp
        else:
            # SFT already won — still keep, chosen = sft (rare but valid)
            chosen, rejected = sft_resp, judge_resp

        pairs.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "domain": domain,
        })

        if (idx + 1) % 50 == 0:
            logger.info("  %s: %d/%d pairs so far", domain, len(pairs), idx + 1)

    logger.info("Domain %s: %d pairs from %d prompts", domain, len(pairs), len(prompts))
    return pairs


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_pairs(domain: str, pairs: list[dict[str, Any]]) -> Path:
    """Write pairs to data/dpo/<domain>/train.jsonl."""
    out_dir = DPO_DATA / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train.jsonl"

    with out_path.open("w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")

    logger.info("Wrote %d pairs to %s", len(pairs), out_path)
    return out_path


def dpo_data_exists(domain: str) -> bool:
    """Return True if DPO data already generated for this domain."""
    path = DPO_DATA / domain / "train.jsonl"
    return path.exists() and path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


async def run(
    domains: list[str],
    *,
    dry_run: bool,
    max_pairs: int,
    judge_url: str,
    sft_url: str,
) -> None:
    if dry_run:
        for domain in domains:
            pairs = dry_run_domain(domain, max_pairs)
            write_pairs(domain, pairs)
        return

    judge_client = TeacherClient(
        endpoints={DEFAULT_JUDGE_MODEL: judge_url},
        cache_dir=str(PROJECT_ROOT / "data" / "teacher_cache"),
    )
    sft_client = TeacherClient(
        endpoints={DEFAULT_SFT_MODEL: sft_url},
        cache_dir=str(PROJECT_ROOT / "data" / "teacher_cache"),
    )

    async with judge_client, sft_client:
        for domain in domains:
            if dpo_data_exists(domain):
                logger.info("Skipping %s — DPO data already exists", domain)
                continue

            logger.info("=== Generating DPO pairs for %s ===", domain)
            prompts = load_prompts(domain, max_pairs)
            if not prompts:
                logger.warning("No prompts for %s — skipping", domain)
                continue

            pairs = await generate_pairs_for_domain(
                domain=domain,
                prompts=prompts,
                judge_client=judge_client,
                sft_client=sft_client,
                judge_model=DEFAULT_JUDGE_MODEL,
                sft_model=DEFAULT_SFT_MODEL,
            )

            if pairs:
                write_pairs(domain, pairs)
            else:
                logger.warning("No pairs generated for %s", domain)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_dpo_pairs",
        description="Generate DPO preference pairs for niche domain fine-tuning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available domains: {', '.join(NICHE_DOMAINS)}",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--domain",
        metavar="NAME",
        help="Single domain to process.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Process all 10 niche domains.",
    )

    parser.add_argument(
        "--max-pairs",
        type=int,
        default=DEFAULT_MAX_PAIRS,
        metavar="N",
        help=f"Maximum pairs per domain (default: {DEFAULT_MAX_PAIRS}).",
    )
    parser.add_argument(
        "--judge-url",
        default=DEFAULT_JUDGE_URL,
        help=f"480B judge server URL (default: {DEFAULT_JUDGE_URL}).",
    )
    parser.add_argument(
        "--sft-url",
        default=DEFAULT_SFT_URL,
        help=f"SFT model server URL (default: {DEFAULT_SFT_URL}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate synthetic placeholder pairs for pipeline testing (no model calls).",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    parser = build_parser()
    args = parser.parse_args()

    if args.domain:
        if args.domain not in NICHE_DOMAINS:
            parser.error(
                f"Unknown domain {args.domain!r}. "
                f"Valid: {', '.join(NICHE_DOMAINS)}"
            )
        domains = [args.domain]
    else:
        domains = NICHE_DOMAINS

    asyncio.run(
        run(
            domains,
            dry_run=args.dry_run,
            max_pairs=args.max_pairs,
            judge_url=args.judge_url,
            sft_url=args.sft_url,
        )
    )


if __name__ == "__main__":
    main()

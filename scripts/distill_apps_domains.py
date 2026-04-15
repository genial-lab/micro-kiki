"""Distill datasets for Phase VII application domains (26-32).

Usage: uv run scripts/distill_apps_domains.py
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from src.distill.teacher_client import TeacherClient
from src.distill.generator import generate_examples

logger = logging.getLogger(__name__)

DOMAIN_TEACHERS = {
    "web-frontend": "devstral",
    "web-backend": "devstral",
    "music-audio": "qwen35",
    "devops": "devstral",
    "llm-orch": "mistral-large",
    "math": "mistral-large",
    "security": "mistral-large",
}


async def main() -> None:
    client = TeacherClient()

    for domain, teacher in DOMAIN_TEACHERS.items():
        prompts_path = Path(f"data/prompts/{domain}.jsonl")
        output_path = Path(f"data/distilled/{domain}.jsonl")

        if not prompts_path.exists():
            logger.warning("No seed prompts for %s, skipping", domain)
            continue

        prompts = [json.loads(l)["prompt"] for l in prompts_path.read_text().strip().split("\n") if l]
        n_per_prompt = max(1, 1500 // len(prompts))

        logger.info("Distilling %s: %d seeds x %d = ~%d examples via %s",
                     domain, len(prompts), n_per_prompt, len(prompts) * n_per_prompt, teacher)

        generated = await generate_examples(
            prompts=prompts, teacher=client, model_name=teacher,
            domain=domain, output_path=output_path, n_per_prompt=n_per_prompt,
        )
        logger.info("Generated %d examples for %s", generated, domain)

    logger.info("Phase VII distillation complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())

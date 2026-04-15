"""Distill chat-fr dataset from teacher model."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from src.distill.teacher_client import TeacherClient
from src.distill.generator import generate_examples

logger = logging.getLogger(__name__)


async def main(teacher_model: str, n_examples: int, output_path: str) -> None:
    prompts_path = Path("data/prompts/chat-fr.jsonl")
    if not prompts_path.exists():
        raise FileNotFoundError(f"Seed prompts not found: {prompts_path}")

    prompts = [json.loads(line)["prompt"] for line in prompts_path.read_text().strip().split("\n") if line]
    n_per_prompt = max(1, n_examples // len(prompts))
    logger.info("Generating %d examples (%d per prompt) from %d seeds", n_examples, n_per_prompt, len(prompts))

    client = TeacherClient()
    generated = await generate_examples(
        prompts=prompts, teacher=client, model_name=teacher_model,
        domain="chat-fr", output_path=Path(output_path), n_per_prompt=n_per_prompt,
    )
    logger.info("Generated %d examples -> %s", generated, output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="mistral-large")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--out", default="data/distilled/chat-fr.jsonl")
    asyncio.run(main(parser.parse_args().teacher, parser.parse_args().n, parser.parse_args().out))

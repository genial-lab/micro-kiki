from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _make_hash(prompt: str, completion: str) -> str:
    return hashlib.sha256(f"{prompt}:{completion}".encode()).hexdigest()[:16]


def load_existing_hashes(output_path: Path) -> set[str]:
    hashes = set()
    if output_path.exists():
        for line in output_path.read_text().strip().split("\n"):
            if line:
                hashes.add(json.loads(line).get("hash", ""))
    return hashes


async def generate_examples(
    prompts: list[str],
    teacher,
    model_name: str,
    domain: str,
    output_path: Path | str,
    n_per_prompt: int = 1,
) -> int:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_hashes = load_existing_hashes(output_path) if output_path.exists() else set()
    generated = 0

    with open(output_path, "a") as f:
        for prompt in prompts:
            for _ in range(n_per_prompt):
                try:
                    completion = await teacher.generate(prompt=prompt, model=model_name)
                except Exception as e:
                    logger.warning("Failed for prompt %.50s: %s", prompt, e)
                    continue
                h = _make_hash(prompt, completion)
                if h in existing_hashes:
                    continue
                f.write(json.dumps({"prompt": prompt, "completion": completion,
                                     "teacher_model": model_name, "domain": domain, "hash": h},
                                    ensure_ascii=False) + "\n")
                existing_hashes.add(h)
                generated += 1

    logger.info("Generated %d examples for domain %s", generated, domain)
    return generated

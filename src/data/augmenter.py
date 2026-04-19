"""Teacher-based synthetic completion for under-represented domains.

When C3's real corpus has fewer than target_n samples in a domain, we seed the
teacher LLM with actual real fragments and ask it to generate similar ones.
The seeds keep the synthetic output domain-coherent; the teacher's coverage
provides diversity.
"""
from __future__ import annotations

import random
from typing import Callable

_PROMPT = """You are helping expand a training corpus in the domain of {domain}.

Given these real example questions from the domain:

{seeds}

Generate ONE more question in the same domain, style, and technical depth. Do not explain, do not preface. Output only the question text itself."""


def augment_domain_via_teacher(
    domain: str,
    seeds: list[str],
    n_to_generate: int,
    teacher_fn: Callable[[str], str],
    seeds_per_prompt: int = 3,
    random_state: int = 0,
) -> list[str]:
    """Use the teacher LLM to generate n_to_generate new questions in the domain.

    Each generation prompt includes a random subset of `seeds` for grounding.
    """
    if n_to_generate <= 0:
        return []
    rng = random.Random(random_state)
    out: list[str] = []
    for _ in range(n_to_generate):
        sample_seeds = rng.sample(seeds, min(seeds_per_prompt, len(seeds)))
        seed_block = "\n".join(f"- {s}" for s in sample_seeds)
        prompt = _PROMPT.format(domain=domain, seeds=seed_block)
        generated = teacher_fn(prompt).strip()
        out.append(generated)
    return out

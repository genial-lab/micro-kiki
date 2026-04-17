"""Aeon integration hook for the serving pipeline.

Prepends recalled memories to prompts and writes new memories post-inference.
Uses dynamic memory budget and structured format matching POC v2.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.memory.aeon import AeonPalace

logger = logging.getLogger(__name__)

MEMORY_BUDGET = 3000


class AeonServingHook:
    """Wraps AeonPalace for pre/post inference memory injection."""

    def __init__(self, palace: AeonPalace) -> None:
        self._palace = palace

    def pre_inference(self, prompt: str, top_k: int = 8) -> str:
        """Recall memories and prepend them with structured format.

        Uses dynamic budget: MEMORY_BUDGET chars split evenly across
        recalled episodes.
        """
        try:
            episodes = self._palace.recall(prompt, top_k=top_k)
        except Exception:
            logger.warning("Aeon recall failed, returning original prompt", exc_info=True)
            return prompt

        if not episodes:
            return prompt

        per_ep = max(200, MEMORY_BUDGET // len(episodes))
        lines = [ep.content[:per_ep] for ep in episodes]
        memory_block = (
            "### Previous conversation context:\n"
            + "\n---\n".join(lines)
            + "\n\n### Current question:\n"
        )
        return memory_block + prompt

    def post_inference(
        self,
        prompt: str,
        response: str,
        domain: str,
        turn_id: str,
    ) -> None:
        """Write the full interaction to Aeon memory."""
        content = f"User: {prompt}\nAssistant: {response}"
        try:
            self._palace.write(
                content=content,
                domain=domain,
                source=turn_id,
            )
        except Exception:
            logger.warning("Aeon write failed for turn %s", turn_id, exc_info=True)

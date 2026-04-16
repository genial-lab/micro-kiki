"""Aeon integration hook for the serving pipeline.

Prepends recalled memories to prompts and writes new memories post-inference.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.memory.aeon import AeonPalace

logger = logging.getLogger(__name__)


class AeonServingHook:
    """Wraps AeonPalace for pre/post inference memory injection."""

    def __init__(self, palace: AeonPalace) -> None:
        self._palace = palace

    def pre_inference(self, prompt: str, top_k: int = 8) -> str:
        """Recall memories and prepend them to the prompt.

        Returns the augmented prompt with ``[Memory]`` lines inserted
        before the original user text.  When no memories are found the
        original prompt is returned unchanged.
        """
        try:
            episodes = self._palace.recall(prompt, top_k=top_k)
        except Exception:
            logger.warning("Aeon recall failed, returning original prompt", exc_info=True)
            return prompt

        if not episodes:
            return prompt

        lines = [f"[Memory] {ep.content}" for ep in episodes]
        return "\n".join(lines) + "\n" + prompt

    def post_inference(
        self,
        prompt: str,
        response: str,
        domain: str,
        turn_id: str,
    ) -> None:
        """Write the interaction to Aeon memory (fire-and-forget async)."""
        content = f"Q: {prompt}\nA: {response}"
        try:
            self._palace.write(
                content=content,
                domain=domain,
                source=turn_id,
            )
        except Exception:
            logger.warning("Aeon write failed for turn %s", turn_id, exc_info=True)

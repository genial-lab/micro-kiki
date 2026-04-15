from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Callable, Awaitable

from src.critique.templates import SELF_REFINE_CRITIQUE, SELF_REFINE_CORRECTION

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CritiqueFeedback:
    factual_errors: list[str]
    missing_info: list[str]
    clarity_issues: list[str]
    confidence: float
    needs_correction: bool
    summary: str


@dataclass(frozen=True)
class CritiqueResult:
    final_response: str
    corrected: bool
    critique: CritiqueFeedback


class SelfRefine:
    """Level 2 auto-critique: structured critique + single correction pass."""

    def __init__(self, generate_fn: Callable[[str], Awaitable[str]]) -> None:
        self._generate = generate_fn

    async def _get_critique(self, query: str, response: str) -> CritiqueFeedback:
        prompt = SELF_REFINE_CRITIQUE.format(query=query, response=response)
        raw = await self._generate(prompt)
        data = json.loads(raw)
        return CritiqueFeedback(
            factual_errors=data.get("factual_errors", []),
            missing_info=data.get("missing_info", []),
            clarity_issues=data.get("clarity_issues", []),
            confidence=data.get("confidence", 0.0),
            needs_correction=data.get("needs_correction", False),
            summary=data.get("summary", ""),
        )

    async def _correct(self, query: str, response: str, critique: CritiqueFeedback) -> str:
        prompt = SELF_REFINE_CORRECTION.format(
            query=query,
            response=response,
            critique=json.dumps({
                "factual_errors": critique.factual_errors,
                "missing_info": critique.missing_info,
                "clarity_issues": critique.clarity_issues,
                "summary": critique.summary,
            }),
        )
        return await self._generate(prompt)

    async def run(self, query: str, response: str) -> CritiqueResult:
        critique = await self._get_critique(query, response)
        if not critique.needs_correction:
            return CritiqueResult(
                final_response=response,
                corrected=False,
                critique=critique,
            )
        corrected = await self._correct(query, response, critique)
        logger.info("Self-refine corrected response (confidence=%.2f)", critique.confidence)
        return CritiqueResult(
            final_response=corrected,
            corrected=True,
            critique=critique,
        )

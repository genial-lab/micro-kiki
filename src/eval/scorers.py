"""Scoring callables for win-rate eval.

Pluggable scorers for :func:`src.eval.forgetting.measure_forgetting_signal`.
All scorers share the signature ``(prompt, reference, response) -> float``
and return a value in ``[0.0, 1.0]`` where higher = better match.

Two built-ins:

- :func:`containment_score` — async wrapper around the existing heuristic
  (reference-token containment in response). Cheap, deterministic, no I/O.
- :class:`JudgeScorer` — async LLM-judge callable, reuses
  :data:`src.eval.stack_eval.JUDGE_PROMPT` so the judge sees the canonical
  prompt template. Expects a ``judge_client`` exposing
  ``async generate(prompt: str, model: str) -> str``.

The forgetting gate calls scorers from a sync path, so both forms must be
awaitable. See :mod:`src.eval.forgetting` for the sync/async bridge.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.eval.stack_eval import JUDGE_PROMPT

logger = logging.getLogger(__name__)


async def containment_score(prompt: str, reference: str, response: str) -> float:
    """Token-containment heuristic (backward-compatible default).

    - 1.0 if reference is non-empty and fully contained in response
    - fraction of reference tokens appearing in response otherwise
    - 0.0 if reference is empty

    ``prompt`` is ignored — kept in the signature for scorer uniformity.
    """
    del prompt  # unused; kept for scorer signature uniformity
    if not reference:
        return 0.0
    out = response.lower()
    ref = reference.lower()
    if ref in out:
        return 1.0
    ref_tokens = [t for t in re.split(r"\s+", ref) if t]
    if not ref_tokens:
        return 0.0
    hits = sum(1 for t in ref_tokens if t in out)
    return hits / len(ref_tokens)


class JudgeScorer:
    """LLM-judge scorer reusing :data:`src.eval.stack_eval.JUDGE_PROMPT`.

    The judge returns JSON of shape ``{"winner": ..., "score": float, ...}``;
    we extract ``score`` and clip to ``[0, 1]``. Unparseable responses return
    ``0.0`` and log a warning — the forgetting gate stays robust when the
    judge is flaky.

    Note: :data:`JUDGE_PROMPT` compares two responses (base vs stack). For
    the forgetting-gate per-prompt scoring, we re-use the same template by
    passing ``response_base = reference`` and ``response_stack = response``.
    The judge's ``score`` field then functions as a quality rating of
    ``response`` against the reference answer, which is what the gate needs.
    """

    def __init__(self, judge_client: Any, judge_model: str = "mistral-large") -> None:
        self._judge = judge_client
        self._model = judge_model

    async def __call__(self, prompt: str, reference: str, response: str) -> float:
        judge_raw = await self._judge.generate(
            prompt=JUDGE_PROMPT.format(
                prompt=prompt,
                response_base=reference,
                response_stack=response,
            ),
            model=self._model,
        )
        try:
            parsed = json.loads(judge_raw)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("JudgeScorer: bad JSON from judge (%s); returning 0.0", exc)
            return 0.0

        raw_score = parsed.get("score", 0.0)
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            logger.warning(
                "JudgeScorer: non-numeric score %r; returning 0.0", raw_score
            )
            return 0.0

        # Clip to [0, 1] — judges occasionally emit out-of-range values.
        if score < 0.0:
            return 0.0
        if score > 1.0:
            return 1.0
        return score

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Awaitable

from src.ralph.research import StoryResearcher
from src.ralph.self_review import CodeReview, ReviewResult
from src.ralph.forgetting_auto import ForgettingChecker

logger = logging.getLogger(__name__)

_STACK_RE = re.compile(r"stack-(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class LoopConfig:
    max_consecutive_failures: int = 3
    max_review_passes: int = 3
    dry_run: bool = False
    progress_file: str = ".ralph/progress.txt"


@dataclass
class StoryOutcome:
    story_id: str
    success: bool
    research_path: Path | None
    review_passes: int
    forgetting_check: bool | None
    error: str | None
    forgetting_angle: float | None = None
    win_rate_drop: float | None = None


class AutonomousLoop:
    """Complete autonomous ralph loop: research -> implement -> critique -> test -> commit."""

    def __init__(
        self,
        researcher: StoryResearcher,
        code_review: CodeReview,
        forgetting_checker: ForgettingChecker,
        implement_fn: Callable[[dict, Path | None], Awaitable[str]],
        test_fn: Callable[[], Awaitable[bool]],
        commit_fn: Callable[[str], Awaitable[None]],
        config: LoopConfig | None = None,
        review_fn: Callable[[str], Awaitable[str]] | None = None,
        eval_fn: Callable[[str], Awaitable[tuple[float, float, float]]] | None = None,
    ) -> None:
        self._researcher = researcher
        self._review = code_review
        self._forgetting = forgetting_checker
        self._implement = implement_fn
        self._test = test_fn
        self._commit = commit_fn
        self._config = config or LoopConfig()
        self._review_fn = review_fn
        self._eval_fn = eval_fn
        self._consecutive_failures = 0

    def _is_training_story(self, story: dict) -> bool:
        title = story.get("title", "").lower()
        desc = story.get("description", "").lower()
        return "train stack" in title or "train stack" in desc

    def _extract_stack_id(self, story: dict) -> str | None:
        text = f"{story.get('title', '')} {story.get('description', '')}"
        m = _STACK_RE.search(text)
        return f"stack-{int(m.group(1)):02d}" if m else None

    async def _review_and_retry(
        self, story: dict, research_path: Path | None, code: str,
    ) -> tuple[int, str]:
        if self._review_fn is None:
            return 0, code

        passes = 1
        while passes <= self._config.max_review_passes:
            prompt = self._review.format_prompt(
                code=code, context=story.get("title", ""),
            )
            raw = await self._review_fn(prompt)
            try:
                result: ReviewResult = self._review.parse_review(raw)
            except Exception as e:
                logger.warning("Review parse failed pass %d: %s", passes, e)
                break

            if result.approved or result.total_issues == 0:
                logger.info("Review approved on pass %d", passes)
                break

            if passes >= self._config.max_review_passes:
                logger.warning(
                    "Review rejecting after %d passes: %s",
                    passes, result.summary,
                )
                break

            logger.info(
                "Review pass %d: %d issues, re-implementing",
                passes, result.total_issues,
            )
            code = await self._implement(story, research_path)
            passes += 1

        return passes, code

    async def run_story(self, story: dict) -> StoryOutcome:
        story_id = story["id"]
        logger.info("=== Starting %s: %s ===", story_id, story.get("title", ""))

        try:
            research_path = await self._researcher.research_story(story)
        except Exception as e:
            logger.warning("Research failed for %s: %s", story_id, e)
            research_path = None

        try:
            code = await self._implement(story, research_path)
        except Exception as e:
            return StoryOutcome(
                story_id=story_id, success=False, research_path=research_path,
                review_passes=0, forgetting_check=None,
                error=f"Implementation failed: {e}",
            )

        review_passes, _ = await self._review_and_retry(
            story, research_path, code or "",
        )

        tests_pass = await self._test()
        if not tests_pass:
            return StoryOutcome(
                story_id=story_id, success=False, research_path=research_path,
                review_passes=review_passes, forgetting_check=None,
                error="Tests failed",
            )

        forgetting_ok: bool | None = None
        forgetting_angle: float | None = None
        win_rate_drop: float | None = None

        if self._is_training_story(story):
            if self._eval_fn is None:
                logger.warning(
                    "Training story %s but no eval_fn — skipping",
                    story_id,
                )
                forgetting_ok = True
            else:
                stack_id = self._extract_stack_id(story) or story_id
                angle, wr_base, wr_adapted = await self._eval_fn(stack_id)
                forg = self._forgetting.evaluate(
                    angle=angle,
                    winrate_base=wr_base,
                    winrate_adapted=wr_adapted,
                )
                self._forgetting.save_result(stack_id, forg)
                forgetting_angle = forg.angle
                win_rate_drop = forg.winrate_drop
                forgetting_ok = not forg.should_rollback
                if forg.should_rollback:
                    return StoryOutcome(
                        story_id=story_id, success=False,
                        research_path=research_path,
                        review_passes=review_passes,
                        forgetting_check=False,
                        error="forgetting rollback triggered",
                        forgetting_angle=forgetting_angle,
                        win_rate_drop=win_rate_drop,
                    )

        if not self._config.dry_run:
            await self._commit(f"feat: {story.get('title', story_id)}")

        return StoryOutcome(
            story_id=story_id, success=True, research_path=research_path,
            review_passes=review_passes, forgetting_check=forgetting_ok,
            error=None,
            forgetting_angle=forgetting_angle, win_rate_drop=win_rate_drop,
        )

    async def run(self, stories: list[dict]) -> list[StoryOutcome]:
        outcomes: list[StoryOutcome] = []
        for story in stories:
            outcome = await self.run_story(story)
            outcomes.append(outcome)

            if outcome.success:
                self._consecutive_failures = 0
            else:
                self._consecutive_failures += 1
                logger.warning(
                    "Failure %d/%d on %s: %s",
                    self._consecutive_failures,
                    self._config.max_consecutive_failures,
                    outcome.story_id,
                    outcome.error,
                )
                if self._consecutive_failures >= self._config.max_consecutive_failures:
                    logger.error(
                        "Hard stop: %d consecutive failures",
                        self._consecutive_failures,
                    )
                    break

        return outcomes

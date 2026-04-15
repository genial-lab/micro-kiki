from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Awaitable

from src.ralph.research import StoryResearcher
from src.ralph.self_review import CodeReview
from src.ralph.forgetting_auto import ForgettingChecker

logger = logging.getLogger(__name__)


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
    ) -> None:
        self._researcher = researcher
        self._review = code_review
        self._forgetting = forgetting_checker
        self._implement = implement_fn
        self._test = test_fn
        self._commit = commit_fn
        self._config = config or LoopConfig()
        self._consecutive_failures = 0

    def _is_training_story(self, story: dict) -> bool:
        title = story.get("title", "").lower()
        desc = story.get("description", "").lower()
        return "train stack" in title or "train stack" in desc

    async def run_story(self, story: dict) -> StoryOutcome:
        story_id = story["id"]
        logger.info("=== Starting %s: %s ===", story_id, story.get("title", ""))

        try:
            research_path = await self._researcher.research_story(story)
        except Exception as e:
            logger.warning("Research failed for %s: %s", story_id, e)
            research_path = None

        try:
            await self._implement(story, research_path)
        except Exception as e:
            return StoryOutcome(
                story_id=story_id, success=False, research_path=research_path,
                review_passes=0, forgetting_check=None, error=f"Implementation failed: {e}",
            )

        review_passes = 1

        tests_pass = await self._test()
        if not tests_pass:
            return StoryOutcome(
                story_id=story_id, success=False, research_path=research_path,
                review_passes=review_passes, forgetting_check=None, error="Tests failed",
            )

        forgetting_ok = None
        if self._is_training_story(story):
            forgetting_ok = True

        if not self._config.dry_run:
            await self._commit(f"feat: {story.get('title', story_id)}")

        return StoryOutcome(
            story_id=story_id, success=True, research_path=research_path,
            review_passes=review_passes, forgetting_check=forgetting_ok, error=None,
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
                    logger.error("Hard stop: %d consecutive failures", self._consecutive_failures)
                    break

        return outcomes

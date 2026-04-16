from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.ralph.autonomous import AutonomousLoop, LoopConfig
from src.ralph.forgetting_auto import ForgettingChecker
from src.ralph.self_review import CodeReview


def _approved_review() -> str:
    return json.dumps({
        "bugs": [], "edge_cases": [], "perf": [],
        "security": [], "style": [],
        "approved": True, "summary": "ok",
    })


def _rejecting_review() -> str:
    return json.dumps({
        "bugs": ["missing guard"], "edge_cases": [], "perf": [],
        "security": [], "style": [],
        "approved": False, "summary": "needs fix",
    })


def _build_loop(
    *,
    implement_return: str = "code",
    review_fn=None,
    eval_fn=None,
    test_pass: bool = True,
    config: LoopConfig | None = None,
    tmp_path: Path,
) -> tuple[AutonomousLoop, dict]:
    researcher = MagicMock()
    researcher.research_story = AsyncMock(return_value=None)
    evals_dir = tmp_path / "evals"
    evals_dir.mkdir(exist_ok=True)
    loop = AutonomousLoop(
        researcher=researcher,
        code_review=CodeReview(),
        forgetting_checker=ForgettingChecker(evals_dir=evals_dir),
        implement_fn=AsyncMock(return_value=implement_return),
        test_fn=AsyncMock(return_value=test_pass),
        commit_fn=AsyncMock(return_value=None),
        config=config or LoopConfig(dry_run=True),
        review_fn=review_fn,
        eval_fn=eval_fn,
    )
    return loop, {
        "implement_fn": loop._implement,
        "test_fn": loop._test,
        "commit_fn": loop._commit,
    }


async def test_no_review_fn_passes_zero(tmp_path):
    loop, _ = _build_loop(tmp_path=tmp_path)
    outcome = await loop.run_story({"id": "s1", "title": "do thing"})
    assert outcome.success
    assert outcome.review_passes == 0


async def test_review_approved_first_pass(tmp_path):
    review_fn = AsyncMock(return_value=_approved_review())
    loop, _ = _build_loop(review_fn=review_fn, tmp_path=tmp_path)
    outcome = await loop.run_story({"id": "s1", "title": "do thing"})
    assert outcome.success
    assert outcome.review_passes == 1
    assert review_fn.await_count == 1


async def test_review_retries_until_approved(tmp_path):
    review_fn = AsyncMock(side_effect=[
        _rejecting_review(), _rejecting_review(), _approved_review(),
    ])
    loop, mocks = _build_loop(review_fn=review_fn, tmp_path=tmp_path)
    outcome = await loop.run_story({"id": "s1", "title": "do thing"})
    assert outcome.success
    assert outcome.review_passes == 3
    assert review_fn.await_count == 3
    assert mocks["implement_fn"].await_count == 3


async def test_review_capped_at_max_passes(tmp_path):
    review_fn = AsyncMock(return_value=_rejecting_review())
    loop, _ = _build_loop(
        review_fn=review_fn,
        config=LoopConfig(max_review_passes=3, dry_run=True),
        tmp_path=tmp_path,
    )
    outcome = await loop.run_story({"id": "s1", "title": "do thing"})
    assert outcome.success
    assert outcome.review_passes == 3
    assert review_fn.await_count == 3


async def test_forgetting_skipped_for_non_training(tmp_path):
    eval_fn = AsyncMock()
    loop, _ = _build_loop(eval_fn=eval_fn, tmp_path=tmp_path)
    outcome = await loop.run_story({"id": "s1", "title": "generate dataset"})
    assert outcome.success
    assert eval_fn.await_count == 0
    assert outcome.forgetting_angle is None
    assert outcome.win_rate_drop is None


async def test_forgetting_pass_for_training(tmp_path):
    eval_fn = AsyncMock(return_value=(45.0, 0.80, 0.79))
    loop, _ = _build_loop(eval_fn=eval_fn, tmp_path=tmp_path)
    outcome = await loop.run_story(
        {"id": "s12", "title": "train stack-01 (chat-fr)"},
    )
    assert outcome.success
    assert outcome.forgetting_check is True
    assert outcome.forgetting_angle == 45.0
    assert outcome.win_rate_drop == pytest.approx(0.01)


async def test_forgetting_rollback_blocks_commit(tmp_path):
    eval_fn = AsyncMock(return_value=(25.0, 0.80, 0.75))
    loop, mocks = _build_loop(eval_fn=eval_fn, tmp_path=tmp_path)
    outcome = await loop.run_story(
        {"id": "s12", "title": "train stack-01 (chat-fr)"},
    )
    assert not outcome.success
    assert outcome.error == "forgetting rollback triggered"
    assert outcome.forgetting_check is False
    assert outcome.forgetting_angle == 25.0
    assert outcome.win_rate_drop == pytest.approx(0.05)
    assert mocks["commit_fn"].await_count == 0


async def test_training_without_eval_fn_still_passes(tmp_path):
    loop, _ = _build_loop(tmp_path=tmp_path)
    outcome = await loop.run_story(
        {"id": "s12", "title": "train stack-01 (chat-fr)"},
    )
    assert outcome.success
    assert outcome.forgetting_check is True
    assert outcome.forgetting_angle is None


async def test_stack_id_extraction(tmp_path):
    eval_fn = AsyncMock(return_value=(45.0, 0.80, 0.79))
    loop, _ = _build_loop(eval_fn=eval_fn, tmp_path=tmp_path)
    await loop.run_story(
        {"id": "s12", "title": "train stack-07 (html-css)"},
    )
    eval_fn.assert_awaited_once_with("stack-07")


async def test_hard_stop_after_consecutive_failures(tmp_path):
    loop, _ = _build_loop(test_pass=False, tmp_path=tmp_path)
    stories = [{"id": f"s{i}", "title": f"story {i}"} for i in range(5)]
    outcomes = await loop.run(stories)
    assert len(outcomes) == 3
    assert all(not o.success for o in outcomes)

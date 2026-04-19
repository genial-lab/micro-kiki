"""Tests for src/routing/llm_judge.py — rubric prompt builder + score parser."""
from __future__ import annotations

import pytest


def test_build_rubric_prompt_contains_question_and_answer():
    from src.routing.llm_judge import build_rubric_prompt

    p = build_rubric_prompt(
        question="What is a Schmitt trigger?",
        answer="A comparator with hysteresis.",
        domain="electronics",
    )
    assert "Schmitt trigger" in p
    assert "comparator" in p
    assert "electronics" in p.lower()
    assert "0 to 5" in p or "0-5" in p


def test_parse_score_extracts_integer():
    from src.routing.llm_judge import parse_score

    assert parse_score("Score: 4") == 4
    assert parse_score("The answer is correct. Score: 5/5") == 5
    assert parse_score("I rate this 3 out of 5.") == 3
    assert parse_score("0") == 0


def test_parse_score_clamps_to_valid_range():
    from src.routing.llm_judge import parse_score

    assert parse_score("Score: 7") == 5, "should clamp above"
    assert parse_score("Score: -1") == 0, "should clamp below"


def test_parse_score_returns_none_on_unparseable():
    from src.routing.llm_judge import parse_score

    assert parse_score("") is None
    assert parse_score("I refuse to score this.") is None
    assert parse_score("garbage text with no numbers") is None


def test_parse_score_prefers_last_integer_on_ambiguity():
    """Judge may reason aloud with numbers before giving final score."""
    from src.routing.llm_judge import parse_score

    assert parse_score("mentions 2 components but misses 3. Final score: 4") == 4

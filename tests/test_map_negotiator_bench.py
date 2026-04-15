"""Tests for story-3 negotiator bench."""

from __future__ import annotations

from src.eval.map_negotiator_bench import V02NegotiatorAgent, run


def test_negotiator_picks_longer_candidate_mostly():
    agent = V02NegotiatorAgent(dissent_rate=0.0)
    assert agent.evaluate_pair("long evidence " * 10, "short") == "a"
    assert agent.evaluate_pair("short", "long evidence " * 10) == "b"


def test_negotiator_dissent_flips_sometimes():
    # With dissent_rate=1.0 every pick should be flipped.
    agent = V02NegotiatorAgent(dissent_rate=1.0)
    assert agent.evaluate_pair("long " * 10, "short") == "b"
    # With 0.0 never flipped.
    agent0 = V02NegotiatorAgent(dissent_rate=0.0)
    assert agent0.evaluate_pair("long " * 10, "short") == "a"


def test_negotiator_judge_confidence_bounds():
    agent = V02NegotiatorAgent()
    assert 0.0 <= agent.judge_confidence("a", "ab") <= 1.0
    assert agent.judge_confidence("x", "x") == 0.0
    assert agent.judge_confidence("x", "xxxxxxxxxx") > 0.5


def test_negotiator_escalation_cost_increases_on_close_pairs():
    agent = V02NegotiatorAgent(escalate_threshold=0.5)
    close_pairs = [("abcd", "abce")] * 5
    far_pairs = [("a", "z" * 50)] * 5
    close_cost = agent.escalation_cost(close_pairs)
    far_cost = agent.escalation_cost(far_pairs)
    assert close_cost > far_cost


def test_run_emits_full_report():
    result = run(n=30, seed=3)
    d = result.to_dict()
    assert d["schema_version"] == "map-negotiator-bench/1.0"
    assert d["agent"] == "V02NegotiatorAgent"
    assert d["n"] == 30
    # Must expose the 3 required scores (Spearman, escalation, cost).
    for key in (
        "spearman_rho",
        "escalation_rate",
        "judge_cost_mean",
        "dissent_rate",
    ):
        assert key in d
    assert isinstance(d["notes"], list)
    assert len(d["notes"]) >= 1


def test_run_deterministic():
    a = run(n=20, seed=11).to_dict()
    b = run(n=20, seed=11).to_dict()
    assert a == b

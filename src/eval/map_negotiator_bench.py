"""Benchmark v0.2 Negotiator vs MAP state-evaluator module (story-3).

The v0.2 Negotiator is specified in
``docs/specs/2026-04-15-cognitive-layer-design.md §3``:

* CAMP arbitration (arxiv 2604.00085): evidence-based winner pick.
* Catfish dissent (arxiv 2505.21503): structured contrarian pass to
  break groupthink.
* Adaptive judge: Qwen3.5-35B-A3B fast pass; escalate to
  Mistral-Large-Opus when confidence < 0.5 or dispatcher says
  ``reasoning`` / ``security``.

No Negotiator code exists on main yet. This bench builds a
**conceptual model**: a pairwise evaluator whose confidence gates
escalation, plus the Catfish dissent coin-flip, mapped onto MAP's
state-evaluator surface.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from src.eval.map_harness import (
    MAPHarness,
    MockAgent,
    spearman,
)


@dataclass
class V02NegotiatorAgent:
    """Conceptual model of v0.2 Negotiator as a MAP state evaluator.

    Scoring model:

    * CAMP picks winner = the candidate whose normalised length is
      higher (stand-in for "more evidence").
    * Catfish triggers a contrarian flip with probability
      ``dissent_rate`` (spec default 0.10). This mimics structured
      dissent surface area.
    * Judge confidence = abs(len(a) - len(b)) / max(len(a), len(b))
      (clamped to [0, 1]). If confidence < ``escalate_threshold``,
      the Negotiator would escalate to Mistral-Large-Opus; we log
      that as an escalation event but still emit the CAMP winner.
    """

    escalate_threshold: float = 0.50
    dissent_rate: float = 0.10
    dissent_seed: int = 17
    cost_fast: float = 1.0
    cost_deep: float = 7.0  # Mistral-Large-Opus is ~7x Qwen35B

    _dissent_counter: int = field(default=0, init=False, repr=False)

    def _dissent(self) -> bool:
        # Deterministic pseudo-random dissent pattern keyed off a
        # rolling counter so the bench is reproducible.
        self._dissent_counter += 1
        mixed = (self._dissent_counter * 2654435761 + self.dissent_seed) & 0xFFFFFFFF
        frac = (mixed % 10000) / 10000.0
        return frac < self.dissent_rate

    def evaluate_pair(self, a: str, b: str) -> str:
        la, lb = len(a), len(b)
        winner = "a" if la >= lb else "b"
        if self._dissent():
            winner = "b" if winner == "a" else "a"
        return winner

    def judge_confidence(self, a: str, b: str) -> float:
        la, lb = len(a), len(b)
        denom = max(la, lb, 1)
        return min(1.0, abs(la - lb) / denom)

    def escalation_cost(
        self, pairs: Sequence[tuple[str, str]]
    ) -> float:
        """Expected judge-call cost over a list of pairs."""
        total = 0.0
        for a, b in pairs:
            conf = self.judge_confidence(a, b)
            total += self.cost_fast
            if conf < self.escalate_threshold:
                total += self.cost_deep
        return total

    # --- MAP protocol stubs ------------------------------------------
    def conflict_monitor(
        self, intents: Sequence[tuple[str, float]]
    ) -> float:
        return 0.0

    def state_predict(self, history: Sequence[str]) -> str:
        return "unknown"

    def decompose(self, goal: str) -> list[str]:
        return []

    def coordinate(self, state: dict[str, Any]) -> str:
        return "act"


@dataclass
class BenchResult:
    """Structured result for story-3's JSON report."""

    agent: str
    n: int
    spearman_rho: float
    accuracy: float
    escalation_rate: float
    judge_cost_mean: float
    dissent_rate: float
    spearman_delta_vs_mock: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "map-negotiator-bench/1.0",
            "agent": self.agent,
            "n": self.n,
            "spearman_rho": round(self.spearman_rho, 4),
            "accuracy": round(self.accuracy, 4),
            "escalation_rate": round(self.escalation_rate, 4),
            "judge_cost_mean": round(self.judge_cost_mean, 4),
            "dissent_rate": round(self.dissent_rate, 4),
            "spearman_delta_vs_mock": round(
                self.spearman_delta_vs_mock, 4
            ),
            "notes": self.notes,
        }


def run(n: int = 50, seed: int = 42) -> BenchResult:
    """Run the Negotiator bench and return a BenchResult.

    Pairs come from the MAP harness's judge generator; we then compute
    Spearman correlation with reference scores plus Negotiator-specific
    metrics (escalation rate, judge cost, effective dissent rate).
    """
    from src.eval.map_harness import gen_judge_prompts

    items = gen_judge_prompts(n=n, seed=seed)
    agent = V02NegotiatorAgent()
    mock = MockAgent()

    preds_agent: list[float] = []
    preds_mock: list[float] = []
    refs: list[float] = []
    correct = 0
    escalations = 0
    total_cost = 0.0
    dissent_triggers = 0
    pre_counter = agent._dissent_counter

    for it in items:
        a, b = it.payload["pair"]
        # Track dissent: compare winner before/after dissent by
        # calling the non-dissent heuristic once.
        baseline_winner = "a" if len(a) >= len(b) else "b"
        winner = agent.evaluate_pair(a, b)
        if winner != baseline_winner:
            dissent_triggers += 1
        preds_agent.append(1.0 if winner == "a" else 0.0)
        preds_mock.append(1.0 if mock.evaluate_pair(a, b) == "a" else 0.0)
        refs.append(1.0 if it.reference == "a" else 0.0)
        if winner == it.reference:
            correct += 1
        conf = agent.judge_confidence(a, b)
        cost = agent.cost_fast
        if conf < agent.escalate_threshold:
            escalations += 1
            cost += agent.cost_deep
        total_cost += cost

    assert agent._dissent_counter - pre_counter == n

    rho_agent = spearman(preds_agent, refs)
    rho_mock = spearman(preds_mock, refs)

    notes = [
        "Conceptual model of v0.2 Negotiator (CAMP + Catfish + "
        "adaptive judge); no runnable code on main.",
        "CAMP winner = longer candidate (evidence proxy).",
        "Catfish dissent = deterministic 10% contrarian flip.",
        "Judge cost in Qwen35B-equivalent units; deep judge = 7x.",
        "Goal is structural mapping, not beating MAP reference.",
    ]

    return BenchResult(
        agent="V02NegotiatorAgent",
        n=len(items),
        spearman_rho=rho_agent,
        accuracy=correct / max(len(items), 1),
        escalation_rate=escalations / max(len(items), 1),
        judge_cost_mean=total_cost / max(len(items), 1),
        dissent_rate=dissent_triggers / max(len(items), 1),
        spearman_delta_vs_mock=rho_agent - rho_mock,
        notes=notes,
    )


def _main() -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="MAP state-evaluator bench vs v0.2 Negotiator"
    )
    parser.add_argument("--out", default="results/map-negotiator.json")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = run(n=args.n, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {out}")
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())

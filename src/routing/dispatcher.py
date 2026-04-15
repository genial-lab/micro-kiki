"""Training-free dispatcher: maps router sigmoid output to 7 meta-intents."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


class MetaIntent(str, Enum):
    QUICK_REPLY = "quick-reply"
    CODING = "coding"
    REASONING = "reasoning"
    CREATIVE = "creative"
    RESEARCH = "research"
    AGENTIC = "agentic"
    TOOL_USE = "tool-use"


@dataclass(frozen=True)
class DispatchResult:
    intent: MetaIntent
    confidence: float
    active_domains: list[int]


def load_intent_mapping(config_path: str | Path = "configs/meta_intents.yaml") -> dict[str, list[int]]:
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return data["intents"]


def validate_mapping(mapping: dict[str, list[int]], num_domains: int = 32) -> bool:
    """Each domain index 0..31 must appear in exactly one bucket."""
    seen: dict[int, str] = {}
    for intent, indices in mapping.items():
        for idx in indices:
            if idx in seen:
                raise ValueError(f"Domain {idx} in both '{seen[idx]}' and '{intent}'")
            seen[idx] = intent
    missing = set(range(num_domains)) - set(seen.keys())
    if missing:
        raise ValueError(f"Domains not assigned: {missing}")
    return True


def dispatch(router_logits, mapping: dict[str, list[int]]) -> DispatchResult:
    """Derive dominant meta-intent from 32-dim sigmoid output.

    Args:
        router_logits: tensor or list of 32 floats (sigmoid outputs).
        mapping: intent name -> list of domain indices.
    """
    if hasattr(router_logits, "tolist"):
        scores = router_logits.tolist()
        if isinstance(scores[0], list):
            scores = scores[0]
    else:
        scores = list(router_logits)

    intent_scores: dict[str, float] = {}
    for intent_name, indices in mapping.items():
        intent_scores[intent_name] = max(scores[i] for i in indices) if indices else 0.0

    best_intent = max(intent_scores, key=intent_scores.get)
    threshold = 0.12
    active = [i for i, s in enumerate(scores) if s > threshold]

    return DispatchResult(
        intent=MetaIntent(best_intent),
        confidence=intent_scores[best_intent],
        active_domains=active,
    )

"""OpenAI-compat model alias table: 7 meta + 35 niche = 42."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from src.routing.router import NICHE_DOMAINS  # authoritative niche frozenset

META_PREFIX = "kiki-meta-"
NICHE_PREFIX = "kiki-niche-"


@dataclass(frozen=True, slots=True)
class ModelAlias:
    model_id: str
    mode: Literal["meta", "niche"]
    target: str


def _meta_intents_path() -> Path:
    # Repo root layout: <repo>/configs/meta_intents.yaml
    # __file__ = <repo>/src/serving/model_aliases.py  -> parents[2] = <repo>
    return Path(__file__).resolve().parents[2] / "configs" / "meta_intents.yaml"


def _load_meta_intents() -> list[str]:
    """Read the meta intents from configs/meta_intents.yaml.

    Supports four shapes:
      - {"intents": {"code": [...], "chat": [...], ...}}  # nested dict (this repo)
      - {"intents": ["code", "chat", ...]}
      - {"code": [...], "chat": [...], ...}  # keys are intents
      - ["code", "chat", ...]
    """
    data = yaml.safe_load(_meta_intents_path().read_text())
    if isinstance(data, dict) and "intents" in data:
        inner = data["intents"]
        if isinstance(inner, dict):
            return sorted(inner.keys())
        if isinstance(inner, list):
            return sorted(inner)
        raise ValueError(
            f"unexpected meta_intents.yaml 'intents' shape: {type(inner).__name__}"
        )
    if isinstance(data, dict):
        return sorted(data.keys())
    if isinstance(data, list):
        return sorted(data)
    raise ValueError(f"unexpected meta_intents.yaml shape: {type(data).__name__}")


def build_aliases() -> list[ModelAlias]:
    aliases: list[ModelAlias] = []
    for intent in _load_meta_intents():
        aliases.append(ModelAlias(f"{META_PREFIX}{intent}", "meta", intent))
    for niche in sorted(NICHE_DOMAINS):
        aliases.append(ModelAlias(f"{NICHE_PREFIX}{niche}", "niche", niche))
    return aliases


def lookup(model_id: str) -> ModelAlias | None:
    for a in build_aliases():
        if a.model_id == model_id:
            return a
    return None

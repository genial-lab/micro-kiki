"""micro-kiki v0.3 memory package.

Contains the AeonSleep unified memory module and its building blocks:
Atlas (vector index), Trace (neuro-symbolic graph), and the AeonSleep
facade. Cognitive modules (sleep tagger, forgetting gate,
consolidation) live in :mod:`src.cognitive`.
"""

from __future__ import annotations

from src.memory.aeonsleep import (
    AeonSleep,
    Episode,
    RecallHit,
    SleepReport,
)
from src.memory.atlas import AtlasIndex, AtlasEntry, SearchHit
from src.memory.trace import Edge, Node, TraceGraph

__all__ = [
    "AeonSleep",
    "AtlasEntry",
    "AtlasIndex",
    "Edge",
    "Episode",
    "Node",
    "RecallHit",
    "SearchHit",
    "SleepReport",
    "TraceGraph",
]

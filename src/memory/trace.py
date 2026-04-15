"""Trace neuro-symbolic episodic graph (arxiv 2601.15311).

Graph store for episodes with causality edges and temporal validity.
Uses in-memory adjacency lists (networkx-free for minimal deps).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Episode:
    id: str
    content: str
    domain: str
    timestamp: datetime
    source: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CausalityEdge:
    from_id: str
    to_id: str
    weight: float
    relation: str = "causes"


class TraceGraph:
    """In-memory episodic graph with causality edges and temporal queries."""

    def __init__(self) -> None:
        self._episodes: dict[str, Episode] = {}
        self._edges: list[CausalityEdge] = []
        self._adjacency: dict[str, list[str]] = {}

    def add_episode(self, episode: Episode) -> None:
        self._episodes[episode.id] = episode
        if episode.id not in self._adjacency:
            self._adjacency[episode.id] = []

    def add_edge(self, edge: CausalityEdge) -> None:
        self._edges.append(edge)
        self._adjacency.setdefault(edge.from_id, []).append(edge.to_id)

    def get_episode(self, episode_id: str) -> Episode | None:
        return self._episodes.get(episode_id)

    def walk(self, from_id: str, max_depth: int = 3) -> list[Episode]:
        visited: set[str] = set()
        result: list[Episode] = []

        def _dfs(node_id: str, depth: int) -> None:
            if depth > max_depth or node_id in visited:
                return
            visited.add(node_id)
            ep = self._episodes.get(node_id)
            if ep:
                result.append(ep)
            for neighbor in self._adjacency.get(node_id, []):
                _dfs(neighbor, depth + 1)

        _dfs(from_id, 0)
        return result

    def query_by_time(self, start: datetime, end: datetime) -> list[Episode]:
        return [
            ep for ep in self._episodes.values()
            if start <= ep.timestamp <= end
        ]

    def query_by_rule(self, domain: str | None = None, min_causality: float = 0.0) -> list[Episode]:
        """Simple rule-based query: filter by domain and/or causality weight."""
        matching_ids: set[str] = set()

        if domain:
            for ep in self._episodes.values():
                if ep.domain == domain:
                    matching_ids.add(ep.id)
        else:
            matching_ids = set(self._episodes.keys())

        if min_causality > 0:
            causal_ids = set()
            for edge in self._edges:
                if edge.weight >= min_causality and edge.to_id in matching_ids:
                    causal_ids.add(edge.to_id)
            matching_ids &= causal_ids

        return [self._episodes[eid] for eid in matching_ids if eid in self._episodes]

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    @property
    def num_edges(self) -> int:
        return len(self._edges)

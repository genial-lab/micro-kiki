"""Tests for the consolidation module (story-10)."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest

from src.cognitive.consolidation import (
    Consolidator,
    RawEpisode,
    SummaryCluster,
    heuristic_summary,
    recall_via_summary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _embed(token_to_axis: dict[str, int], text: str, dim: int = 16) -> list[float]:
    """Toy embedding: each known token contributes a unit bump on its axis."""
    vec = [0.0] * dim
    words = text.lower().split()
    for w in words:
        if w in token_to_axis:
            vec[token_to_axis[w]] += 1.0
    # Normalise so cosine works cleanly.
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


def _make_dataset(seed: int = 0):
    """Build 100 episodes across 10 topics with planted facts."""
    topics = [
        ("alpha", "battery", "battery voltage reading is 12 volts"),
        ("beta", "motor", "motor torque curve peaks at 120 newton meters"),
        ("gamma", "heat", "thermal coupling increases when fan fails"),
        ("delta", "sensor", "lidar sensor scans at 30 hertz frame rate"),
        ("epsilon", "router", "router chooses stack three for math tasks"),
        ("zeta", "memory", "atlas index stores normalised vectors only"),
        ("eta", "sleep", "sleep cycle runs every twenty four hours"),
        ("theta", "energy", "energy draw drops under neuromorphic routing"),
        ("iota", "qa", "question answering escalates to opus rarely"),
        ("kappa", "cooling", "cooling pump activates above sixty degrees"),
    ]
    # Shared vocabulary for embeddings.
    vocab = sorted(
        {
            w
            for _topic, _key, text in topics
            for w in text.lower().split()
        }
    )
    axis = {w: i % 16 for i, w in enumerate(vocab)}

    base = datetime(2026, 4, 16, 9, 0, 0)
    episodes: list[RawEpisode] = []
    for t_idx, (topic, key, sentence) in enumerate(topics):
        for i in range(10):
            text = f"{sentence} (variant {i})"
            ts = base + timedelta(hours=t_idx * 2) + timedelta(minutes=i)
            episodes.append(
                RawEpisode(
                    id=f"{topic}-{i:02d}",
                    text=text,
                    embedding=_embed(axis, text),
                    ts=ts,
                    topic=topic,
                )
            )
    return episodes, topics


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_heuristic_summary_non_empty() -> None:
    texts = [
        "battery voltage reading is 12 volts.",
        "battery voltage at 12 volts measured twice.",
        "the battery holds charge steadily.",
    ]
    summary = heuristic_summary(texts, max_sentences=2)
    assert summary
    assert "battery" in summary.lower()


def test_consolidator_shrinks_to_cluster_count() -> None:
    episodes, topics = _make_dataset()
    consolidator = Consolidator(
        topic_threshold=0.5,
        temporal_window=timedelta(hours=6),
    )
    clusters = consolidator.consolidate(episodes)
    assert 1 <= len(clusters) <= 20, (
        f"expected ≤20 clusters, got {len(clusters)}"
    )
    stats = consolidator.last_stats()
    assert stats is not None
    assert stats.n_input == 100
    assert stats.compression_ratio <= 0.2


def test_backrefs_preserved() -> None:
    episodes, _ = _make_dataset()
    clusters = Consolidator(temporal_window=timedelta(hours=6)).consolidate(
        episodes
    )
    covered: set[str] = set()
    for cl in clusters:
        for m in cl.member_ids:
            covered.add(m)
    assert covered == {e.id for e in episodes}


def test_recall_via_summary_precision() -> None:
    episodes, topics = _make_dataset()
    clusters = Consolidator(
        topic_threshold=0.5,
        temporal_window=timedelta(hours=6),
    ).consolidate(episodes)
    # Held-out QA probes: ask for each planted fact, expect the cluster
    # that contains the matching episodes to rank first.
    hits = 0
    for topic, _key, sentence in topics:
        probe = sentence
        got = recall_via_summary(probe, clusters, top_k=1)
        if got and got[0].topic == topic:
            hits += 1
    recall = hits / len(topics)
    assert recall >= 0.9, f"recall {recall} below 0.9"


def test_temporal_window_splits_clusters() -> None:
    """Two episodes on the same topic but far apart in time should split."""
    axis = {"foo": 0, "bar": 1}
    base = datetime(2026, 4, 16, 9, 0, 0)
    e1 = RawEpisode(
        id="a",
        text="foo bar",
        embedding=_embed(axis, "foo bar"),
        ts=base,
        topic="alpha",
    )
    e2 = RawEpisode(
        id="b",
        text="foo bar",
        embedding=_embed(axis, "foo bar"),
        ts=base + timedelta(days=3),
        topic="alpha",
    )
    clusters = Consolidator(
        topic_threshold=0.5,
        temporal_window=timedelta(hours=24),
    ).consolidate([e1, e2])
    assert len(clusters) == 2


def test_topic_mismatch_forces_split() -> None:
    axis = {"foo": 0, "bar": 1}
    base = datetime(2026, 4, 16, 9, 0, 0)
    e1 = RawEpisode(
        id="a",
        text="foo",
        embedding=_embed(axis, "foo"),
        ts=base,
        topic="alpha",
    )
    e2 = RawEpisode(
        id="b",
        text="foo",
        embedding=_embed(axis, "foo"),
        ts=base + timedelta(minutes=1),
        topic="beta",
    )
    clusters = Consolidator(topic_threshold=0.5).consolidate([e1, e2])
    assert len(clusters) == 2


def test_empty_input_yields_no_clusters() -> None:
    clusters = Consolidator().consolidate([])
    assert clusters == []


def test_summarize_fn_injection() -> None:
    def stub(texts):
        return f"STUB:{len(texts)}"

    episodes, _ = _make_dataset()
    clusters = Consolidator(
        summarize_fn=stub,
        topic_threshold=0.5,
        temporal_window=timedelta(hours=6),
    ).consolidate(episodes)
    for cl in clusters:
        assert cl.text.startswith("STUB:")

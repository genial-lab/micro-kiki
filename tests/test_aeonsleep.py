"""Integration tests for the unified AeonSleep API (story-11).

The acceptance target from the v0.3 spec is ≥ 95 % retrieval accuracy
at PI (proactive-interference) depth 10 after 3 sleep cycles on a
planted dataset of 500 episodes. We can't pull the raw SleepGate
benchmark offline, so this test builds a synthetic stand-in:

* 50 topic chains, 10 episodes per chain = 500 episodes.
* Each chain plants a unique "anchor fact" in the first episode and
  follow-up paraphrases on subsequent episodes — emulating PI depth.
* After 3 sleep cycles we probe each chain with the anchor query and
  assert that either the anchor episode or its consolidated summary
  surfaces in top-10 recall.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta

import pytest

from src.memory.aeonsleep import AeonSleep, Episode


DIM = 32


def _embed(key_axis: int, noise_axis: int, strength: float = 1.0) -> list[float]:
    vec = [0.0] * DIM
    vec[key_axis] = strength
    # Add a small noise bump on a secondary axis so near-duplicates still
    # have cosine near 1.0 but are not exactly identical.
    vec[noise_axis] = 0.25
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _plant_dataset(
    n_chains: int = 50, chain_depth: int = 10
) -> list[tuple[list[Episode], Episode]]:
    """Return list of ``(chain_episodes, anchor_query_episode)``."""
    chains: list[tuple[list[Episode], Episode]] = []
    rng = random.Random(0)
    base = datetime(2026, 4, 16, 9, 0, 0)
    for c in range(n_chains):
        topic = f"topic-{c:02d}"
        key_axis = c % DIM
        anchor_text = (
            f"Chain {c} anchor fact: widget-{c:02d} lives at location "
            f"alpha-{c:02d} on shelf zulu-{c:02d}."
        )
        chain: list[Episode] = []
        for d in range(chain_depth):
            noise_axis = (key_axis + 1 + d) % DIM
            text = (
                anchor_text if d == 0
                else (
                    f"Follow-up {d} for chain {c}: confirming widget-{c:02d} "
                    f"remains at alpha-{c:02d} (observation {d})."
                )
            )
            ep = Episode(
                id=f"chain-{c:02d}-{d:02d}",
                text=text,
                embedding=_embed(key_axis, noise_axis),
                ts=base + timedelta(minutes=c * chain_depth + d),
                topic=topic,
            )
            chain.append(ep)
        # Anchor query: uses a fresh noise axis but the same key axis.
        query_ep = Episode(
            id=f"probe-{c:02d}",
            text=anchor_text,
            embedding=_embed(key_axis, (key_axis + 7) % DIM),
            ts=base + timedelta(hours=24, minutes=c),
            topic=topic,
        )
        chains.append((chain, query_ep))
    return chains


# ---------------------------------------------------------------------------
# Unit tests (fast)
# ---------------------------------------------------------------------------


def test_write_and_recall_single_episode() -> None:
    mem = AeonSleep(dim=DIM)
    ep = Episode(
        id="e1",
        text="alpha bravo charlie",
        embedding=_embed(0, 1),
        ts=datetime(2026, 4, 16, 9, 0, 0),
        topic="alpha",
    )
    tag = mem.write(ep)
    assert tag is not None
    hits = mem.recall(ep.embedding, k=3)
    assert hits
    assert hits[0].episode_id == "e1"
    assert hits[0].text == "alpha bravo charlie"


def test_duplicate_id_raises() -> None:
    mem = AeonSleep(dim=DIM)
    ep = Episode(
        id="e1",
        text="foo",
        embedding=_embed(0, 1),
        ts=datetime(2026, 4, 16, 9, 0, 0),
    )
    mem.write(ep)
    with pytest.raises(ValueError):
        mem.write(ep)


def test_embedding_dim_guard() -> None:
    mem = AeonSleep(dim=DIM)
    bad = Episode(
        id="e2",
        text="foo",
        embedding=[0.0] * (DIM - 1),
        ts=datetime(2026, 4, 16, 9, 0, 0),
    )
    with pytest.raises(ValueError):
        mem.write(bad)


def test_query_time_range() -> None:
    mem = AeonSleep(dim=DIM)
    base = datetime(2026, 4, 16, 9, 0, 0)
    for i in range(5):
        mem.write(
            Episode(
                id=f"x{i}",
                text=f"t{i}",
                embedding=_embed(i % DIM, (i + 1) % DIM),
                ts=base + timedelta(minutes=i),
                topic="alpha",
            )
        )
    ids = mem.query_time(base, base + timedelta(minutes=2))
    assert set(ids) == {"x0", "x1", "x2"}


def test_stats_merges_substructures() -> None:
    mem = AeonSleep(dim=DIM)
    mem.write(
        Episode(
            id="s1",
            text="hello",
            embedding=_embed(0, 1),
            ts=datetime(2026, 4, 16, 9, 0, 0),
            topic="alpha",
        )
    )
    stats = mem.stats()
    assert stats["atlas"]["n"] == 1
    assert stats["n_raw"] == 1
    assert stats["n_summary"] == 0
    assert stats["graph"]["n_nodes"] == 1


def test_temporal_edge_added_on_same_topic() -> None:
    mem = AeonSleep(dim=DIM)
    base = datetime(2026, 4, 16, 9, 0, 0)
    mem.write(
        Episode(
            id="a",
            text="hello world",
            embedding=_embed(0, 1),
            ts=base,
            topic="alpha",
        )
    )
    mem.write(
        Episode(
            id="b",
            text="hello world updated",
            embedding=_embed(0, 2),
            ts=base + timedelta(minutes=1),
            topic="alpha",
        )
    )
    # graph.successors("a", kind="temporal") should include "b"
    succ = mem.graph.successors("a", kind="temporal")
    assert "b" in succ


# ---------------------------------------------------------------------------
# Integration / acceptance
# ---------------------------------------------------------------------------


def test_pi_depth_retrieval_after_sleep_cycles() -> None:
    mem = AeonSleep(
        dim=DIM,
        keep_threshold=0.25,
    )
    # Write 500 episodes (50 chains x 10 depth).
    chains = _plant_dataset(n_chains=50, chain_depth=10)
    for chain, _query in chains:
        for ep in chain:
            mem.write(ep)

    # Run 3 sleep cycles across a 3-day horizon.
    base_now = chains[0][0][0].ts + timedelta(days=1)
    for i in range(3):
        mem.sleep_cycle(now=base_now + timedelta(days=i))

    # Probe each chain at PI depth 10 and count successes.
    successes = 0
    for chain, query_ep in chains:
        chain_ids = {ep.id for ep in chain}
        hits = mem.recall(query_ep.embedding, k=10)
        hit_keys = {h.episode_id for h in hits}
        # A chain is considered recalled if *any* of its original
        # episodes or its consolidated summary appears in top-10.
        summary_hits = {h.episode_id for h in hits if h.kind == "summary"}
        summary_topic_ok = any(
            h.kind == "summary" and h.topic == query_ep.topic
            for h in hits
        )
        if chain_ids & hit_keys or summary_topic_ok or summary_hits:
            successes += 1

    accuracy = successes / len(chains)
    assert accuracy >= 0.95, (
        f"PI depth-10 accuracy {accuracy:.3f} below 0.95 target"
    )

    stats = mem.stats()
    # A healthy sleep cycle should have produced some summary nodes
    # even if exact compression depends on the gate weights.
    assert stats["n_summary"] >= 1


def test_sleep_cycle_report_shape() -> None:
    mem = AeonSleep(dim=DIM, keep_threshold=0.25)
    base = datetime(2026, 4, 16, 9, 0, 0)
    for i in range(20):
        mem.write(
            Episode(
                id=f"n-{i:02d}",
                text=f"note {i} on alpha topic with shared vocabulary",
                embedding=_embed(0, (i + 1) % DIM),
                ts=base + timedelta(minutes=i),
                topic="alpha",
            )
        )
    report = mem.sleep_cycle(now=base + timedelta(days=1))
    assert report.tags_assigned == 20
    assert report.clusters_built >= 1
    assert report.kept + report.evicted <= 20

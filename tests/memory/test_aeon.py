"""Tests for Aeon Memory Palace (Phases VIII)."""
from __future__ import annotations

from datetime import datetime, timedelta
import numpy as np
import pytest
from src.memory.atlas import AtlasIndex
from src.memory.trace import TraceGraph, Episode, CausalityEdge
from src.memory.aeon import AeonPalace


class TestAtlasIndex:
    def test_insert_and_recall(self):
        idx = AtlasIndex(dim=64)
        vec = np.random.randn(64).astype(np.float32)
        idx.insert("v1", vec)
        results = idx.recall(vec, top_k=1)
        assert len(results) == 1
        assert results[0][0] == "v1"
        assert results[0][1] > 0.99

    def test_recall_empty(self):
        idx = AtlasIndex(dim=64)
        assert idx.recall(np.zeros(64), top_k=5) == []

    def test_top_k_ordering(self):
        idx = AtlasIndex(dim=32)
        target = np.ones(32, dtype=np.float32)
        close = target * 0.9 + np.random.randn(32).astype(np.float32) * 0.1
        far = np.random.randn(32).astype(np.float32)
        idx.insert("close", close)
        idx.insert("far", far)
        results = idx.recall(target, top_k=2)
        assert results[0][0] == "close"


class TestTraceGraph:
    def test_add_and_walk(self):
        g = TraceGraph()
        now = datetime.now()
        e1 = Episode(id="e1", content="first", domain="test", timestamp=now)
        e2 = Episode(id="e2", content="second", domain="test", timestamp=now)
        g.add_episode(e1)
        g.add_episode(e2)
        g.add_edge(CausalityEdge(from_id="e1", to_id="e2", weight=0.9))
        walked = g.walk("e1", max_depth=2)
        assert len(walked) == 2

    def test_query_by_time(self):
        g = TraceGraph()
        now = datetime.now()
        old = now - timedelta(days=30)
        g.add_episode(Episode(id="old", content="old", domain="x", timestamp=old))
        g.add_episode(Episode(id="new", content="new", domain="x", timestamp=now))
        results = g.query_by_time(now - timedelta(days=1), now + timedelta(hours=1))
        assert len(results) == 1
        assert results[0].id == "new"

    def test_query_by_rule(self):
        g = TraceGraph()
        now = datetime.now()
        g.add_episode(Episode(id="e1", content="a", domain="python", timestamp=now))
        g.add_episode(Episode(id="e2", content="b", domain="rust", timestamp=now))
        results = g.query_by_rule(domain="python")
        assert len(results) == 1


class TestAeonPalace:
    def test_write_and_recall(self):
        aeon = AeonPalace(dim=3072)
        eid = aeon.write("Test episode about MoE-LoRA", domain="ml")
        results = aeon.recall("MoE-LoRA", top_k=1)
        assert len(results) >= 1

    def test_write_with_links(self):
        aeon = AeonPalace(dim=3072)
        e1 = aeon.write("First event", domain="test")
        e2 = aeon.write("Second event", domain="test", links=[e1])
        walked = aeon.walk(e1, max_depth=2)
        assert len(walked) >= 2

    def test_compress(self):
        aeon = AeonPalace(dim=3072)
        old_ts = datetime.now() - timedelta(days=60)
        aeon.write("Old content that is very long " * 20, domain="test", timestamp=old_ts)
        compressed = aeon.compress(older_than=datetime.now() - timedelta(days=30))
        assert compressed == 1

    def test_stats(self):
        aeon = AeonPalace(dim=3072)
        aeon.write("Episode 1", domain="a")
        aeon.write("Episode 2", domain="b")
        stats = aeon.stats
        assert stats["vectors"] == 2
        assert stats["episodes"] == 2

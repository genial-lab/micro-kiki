from __future__ import annotations

import time
import pytest
from src.search.cache import SearchCache


@pytest.fixture
def cache(tmp_path):
    db_path = tmp_path / "test_cache.sqlite"
    return SearchCache(db_path=str(db_path))


class TestSearchCache:
    def test_store_and_retrieve(self, cache):
        cache.store(
            backend="exa",
            query="MoE-LoRA training",
            results=[{"title": "Paper A", "url": "https://a.com"}],
            ttl_seconds=3600,
        )
        hit = cache.lookup(backend="exa", query="MoE-LoRA training")
        assert hit is not None
        assert len(hit) == 1
        assert hit[0]["title"] == "Paper A"

    def test_miss_on_unknown_query(self, cache):
        hit = cache.lookup(backend="exa", query="nonexistent")
        assert hit is None

    def test_expired_entry_returns_none(self, cache):
        cache.store(
            backend="exa",
            query="old query",
            results=[{"title": "Old"}],
            ttl_seconds=0,
        )
        time.sleep(0.1)
        hit = cache.lookup(backend="exa", query="old query")
        assert hit is None

    def test_different_backends_independent(self, cache):
        cache.store(backend="exa", query="q", results=[{"src": "exa"}], ttl_seconds=3600)
        cache.store(backend="scholar", query="q", results=[{"src": "scholar"}], ttl_seconds=3600)
        assert cache.lookup(backend="exa", query="q")[0]["src"] == "exa"
        assert cache.lookup(backend="scholar", query="q")[0]["src"] == "scholar"

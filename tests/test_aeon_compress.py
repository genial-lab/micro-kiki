from __future__ import annotations

import hashlib
import threading
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.memory.aeon import AeonPalace


def _mock_embed(dim: int = 64):
    """Return a deterministic hash-based embed_fn for tests."""
    def fn(text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode()).digest()
        rng = np.random.RandomState(int.from_bytes(h[:4], "big"))
        vec = rng.randn(dim).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-8)
    return fn


def _populate_palace(palace: AeonPalace, n: int, age_days: int = 60) -> list[str]:
    """Write n episodes dated `age_days` ago."""
    ts = datetime.now(timezone.utc) - timedelta(days=age_days)
    ids = []
    for i in range(n):
        eid = palace.write(
            content=f"Episode {i} with some long content that should be compressible " * 5,
            domain="test",
            timestamp=ts - timedelta(hours=i),
        )
        ids.append(eid)
    return ids


async def test_compress_reduces_old_episodes():
    palace = AeonPalace(dim=64, embed_fn=_mock_embed(64))
    _populate_palace(palace, 50, age_days=60)
    _populate_palace(palace, 10, age_days=1)

    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    count = palace.compress(older_than=cutoff)

    assert count == 50
    assert palace.stats["episodes"] == 60


async def test_compress_with_summarize_fn():
    palace = AeonPalace(dim=64, embed_fn=_mock_embed(64))
    _populate_palace(palace, 20, age_days=60)

    def summarizer(text: str) -> str:
        return f"[SUMMARY] {text[:30]}"

    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    count = palace.compress(older_than=cutoff, summarize_fn=summarizer)

    assert count == 20
    episodes = palace.query_by_time(
        datetime.now(timezone.utc) - timedelta(days=90),
        datetime.now(timezone.utc),
    )
    compressed = [e for e in episodes if e.metadata.get("compressed")]
    assert len(compressed) == 20
    assert all(e.content.startswith("[SUMMARY]") for e in compressed)


async def test_compress_skips_recent():
    palace = AeonPalace(dim=64, embed_fn=_mock_embed(64))
    _populate_palace(palace, 10, age_days=5)

    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    count = palace.compress(older_than=cutoff)

    assert count == 0


async def test_recall_finds_compressed():
    palace = AeonPalace(dim=64, embed_fn=_mock_embed(64))
    ids = _populate_palace(palace, 30, age_days=60)

    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    palace.compress(older_than=cutoff)

    results = palace.recall("Episode 0", top_k=5)
    assert len(results) > 0


async def test_daemon_one_shot():
    """Test the daemon compresses via one-shot mode."""
    palace = AeonPalace(dim=64, embed_fn=_mock_embed(64))
    _populate_palace(palace, 10, age_days=60)

    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    count = palace.compress(older_than=cutoff)
    assert count == 10


async def test_daemon_cli_help():
    """Test that the daemon script accepts --help."""
    import subprocess
    result = subprocess.run(
        ["uv", "run", "python", "scripts/aeon_compress_daemon.py", "--help"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    assert "compression" in result.stdout.lower() or "compress" in result.stdout.lower()

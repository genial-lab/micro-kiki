"""Tests for AeonServingHook dynamic memory format."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from src.memory.aeon import AeonPalace
from src.serving.aeon_hook import AeonServingHook


def _mock_embed(dim: int = 64):
    def fn(text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode()).digest()
        rng = np.random.RandomState(int.from_bytes(h[:4], "big"))
        vec = rng.randn(dim).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-8)
    return fn


class TestAeonServingHookFormat:
    def test_pre_inference_uses_structured_format(self):
        palace = AeonPalace(dim=64, embed_fn=_mock_embed(64))
        palace.write("User: What is a buck converter?\nAssistant: A buck converter steps down voltage.",
                     domain="power")
        hook = AeonServingHook(palace)
        result = hook.pre_inference("Design a boost converter")
        assert "### Previous conversation context:" in result
        assert "### Current question:" in result
        assert "Design a boost converter" in result

    def test_pre_inference_no_memories_returns_original(self):
        palace = AeonPalace(dim=64, embed_fn=_mock_embed(64))
        hook = AeonServingHook(palace)
        result = hook.pre_inference("Hello world")
        assert result == "Hello world"

    def test_post_inference_stores_full_response(self):
        palace = AeonPalace(dim=64, embed_fn=_mock_embed(64))
        hook = AeonServingHook(palace)
        long_response = "SPICE is a circuit simulator. " * 100
        hook.post_inference(
            prompt="What is SPICE?",
            response=long_response,
            domain="spice",
            turn_id="t1",
        )
        episodes = palace.recall("SPICE simulator", top_k=1)
        assert len(episodes) == 1
        assert "User: What is SPICE?" in episodes[0].content
        assert len(episodes[0].content) > 500

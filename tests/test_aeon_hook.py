"""Tests for AeonServingHook — memory injection in serving pipeline."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.memory.trace import Episode
from src.serving.aeon_hook import AeonServingHook


def _make_episode(content: str, domain: str = "test") -> Episode:
    return Episode(
        id="ep-1",
        content=content,
        domain=domain,
        timestamp=datetime(2026, 1, 1),
    )


class TestPreInference:
    def test_injects_memories(self):
        palace = MagicMock()
        palace.recall.return_value = [
            _make_episode("fact A"),
            _make_episode("fact B"),
        ]
        hook = AeonServingHook(palace)
        result = hook.pre_inference("hello")
        assert "[Memory] fact A" in result
        assert "[Memory] fact B" in result
        assert result.endswith("hello")
        palace.recall.assert_called_once_with("hello", top_k=8)

    def test_custom_top_k(self):
        palace = MagicMock()
        palace.recall.return_value = []
        hook = AeonServingHook(palace)
        hook.pre_inference("hello", top_k=3)
        palace.recall.assert_called_once_with("hello", top_k=3)

    def test_no_memories_returns_original(self):
        palace = MagicMock()
        palace.recall.return_value = []
        hook = AeonServingHook(palace)
        result = hook.pre_inference("hello world")
        assert result == "hello world"

    def test_recall_exception_returns_original(self):
        palace = MagicMock()
        palace.recall.side_effect = RuntimeError("backend down")
        hook = AeonServingHook(palace)
        result = hook.pre_inference("hello")
        assert result == "hello"

    def test_empty_prompt(self):
        palace = MagicMock()
        palace.recall.return_value = []
        hook = AeonServingHook(palace)
        result = hook.pre_inference("")
        assert result == ""

    def test_none_recall_returns_original(self):
        palace = MagicMock()
        palace.recall.return_value = None
        hook = AeonServingHook(palace)
        # None is falsy, should return original
        result = hook.pre_inference("prompt")
        assert result == "prompt"


class TestPostInference:
    def test_writes_correctly(self):
        palace = MagicMock()
        hook = AeonServingHook(palace)
        hook.post_inference(
            prompt="what is x?",
            response="x is 42",
            domain="math",
            turn_id="turn-001",
        )
        palace.write.assert_called_once_with(
            content="Q: what is x?\nA: x is 42",
            domain="math",
            source="turn-001",
        )

    def test_write_exception_does_not_raise(self):
        palace = MagicMock()
        palace.write.side_effect = RuntimeError("write failed")
        hook = AeonServingHook(palace)
        # Should not raise
        hook.post_inference(
            prompt="q", response="a", domain="d", turn_id="t"
        )

    def test_empty_response(self):
        palace = MagicMock()
        hook = AeonServingHook(palace)
        hook.post_inference(prompt="q", response="", domain="d", turn_id="t")
        palace.write.assert_called_once()
        call_kwargs = palace.write.call_args
        assert "Q: q\nA: " in call_kwargs.kwargs["content"]

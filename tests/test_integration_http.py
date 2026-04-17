"""Integration test: router HTTP dispatch (Story-97).

Mock HTTP server, test router dispatch via HTTP, adapter loading, response generation.
Tests vllm_server.py and aeon_hook.py APIs.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.serving.vllm_server import VLLMServer, VLLMServerConfig
from src.serving.aeon_hook import AeonServingHook
from src.routing.dispatcher import dispatch, load_intent_mapping, MetaIntent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mapping():
    return load_intent_mapping("configs/meta_intents.yaml")


@pytest.fixture
def server():
    """VLLMServer instance (no real subprocess)."""
    config = VLLMServerConfig(model_path="mock/model", port=9999)
    return VLLMServer(config=config)


@pytest.fixture
def mock_aeon_palace():
    """Mock AeonPalace with recall/write methods."""
    palace = MagicMock()
    palace.recall.return_value = []
    palace.write.return_value = "eid-001"
    return palace


# ---------------------------------------------------------------------------
# Route + Serve
# ---------------------------------------------------------------------------


class TestRouteAndServe:
    """Test the full route -> serve pipeline via HTTP mock."""

    @patch("src.serving.vllm_server.httpx.post")
    def test_generate_returns_content(self, mock_post, server):
        mock_post.return_value = MagicMock(
            status_code=200,
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={
                "choices": [{"message": {"content": "Hello! How can I help?"}}]
            }),
        )
        result = server.generate("Bonjour", model="qwen3.5-35b")
        assert result == "Hello! How can I help?"
        mock_post.assert_called_once()

    @patch("src.serving.vllm_server.httpx.post")
    def test_generate_with_adapter(self, mock_post, server):
        mock_post.return_value = MagicMock(
            status_code=200,
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={
                "choices": [{"message": {"content": "Sorted list implementation"}}]
            }),
        )
        result = server.generate("Sort a list", model="qwen3.5-35b", adapter="stack-03-python")
        assert result == "Sorted list implementation"
        # Verify adapter is used as model name
        call_payload = mock_post.call_args[1]["json"]
        assert call_payload["model"] == "stack-03-python"

    @patch("src.serving.vllm_server.httpx.post")
    def test_generate_error_returns_empty(self, mock_post, server):
        import httpx
        mock_post.side_effect = httpx.HTTPError("Connection refused")
        result = server.generate("test", model="qwen3.5-35b")
        assert result == ""

    @patch("src.serving.vllm_server.httpx.post")
    def test_route_then_serve(self, mock_post, server, mapping):
        """Full pipeline: dispatch -> select adapter -> generate."""
        mock_post.return_value = MagicMock(
            status_code=200,
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={
                "choices": [{"message": {"content": "def sort_list(lst): ..."}}]
            }),
        )
        # Step 1: route
        logits = [0.05] * 35
        logits[2] = 0.90  # python
        result = dispatch(logits, mapping)
        assert result.intent == MetaIntent.CODING

        # Step 2: serve with adapter
        adapter = "stack-03-python" if 2 in result.active_domains else None
        response = server.generate("Sort a list", model="qwen3.5-35b", adapter=adapter)
        assert response == "def sort_list(lst): ..."

    @patch("src.serving.vllm_server.httpx.get")
    def test_health_check_ok(self, mock_get, server):
        mock_get.return_value = MagicMock(
            status_code=200,
            raise_for_status=MagicMock(),
        )
        health = server.health()
        assert health["status"] == "ok"

    @patch("src.serving.vllm_server.httpx.get")
    def test_health_check_error(self, mock_get, server):
        import httpx
        mock_get.side_effect = httpx.HTTPError("Timeout")
        health = server.health()
        assert health["status"] == "error"


# ---------------------------------------------------------------------------
# Adapter loading via HTTP
# ---------------------------------------------------------------------------


class TestAdapterLoading:
    """Test LoRA adapter load/unload via HTTP API."""

    @patch("src.serving.vllm_server.httpx.post")
    def test_load_adapter_success(self, mock_post, server):
        mock_post.return_value = MagicMock(
            status_code=200,
            raise_for_status=MagicMock(),
        )
        ok = server.load_adapter("stack-01-chat-fr", "/path/to/adapter")
        assert ok is True

    @patch("src.serving.vllm_server.httpx.post")
    def test_load_adapter_failure(self, mock_post, server):
        import httpx
        mock_post.side_effect = httpx.HTTPError("Not found")
        ok = server.load_adapter("stack-99", "/bad/path")
        assert ok is False

    @patch("src.serving.vllm_server.httpx.post")
    def test_unload_adapter_success(self, mock_post, server):
        mock_post.return_value = MagicMock(
            status_code=200,
            raise_for_status=MagicMock(),
        )
        ok = server.unload_adapter("stack-01-chat-fr")
        assert ok is True

    @patch("src.serving.vllm_server.httpx.post")
    def test_unload_adapter_failure(self, mock_post, server):
        import httpx
        mock_post.side_effect = httpx.HTTPError("Error")
        ok = server.unload_adapter("stack-99")
        assert ok is False

    @patch("src.serving.vllm_server.httpx.post")
    def test_load_then_generate(self, mock_post, server):
        """Load adapter, then generate with it."""
        # First call: load_adapter
        # Second call: generate
        mock_post.side_effect = [
            MagicMock(status_code=200, raise_for_status=MagicMock()),
            MagicMock(
                status_code=200,
                raise_for_status=MagicMock(),
                json=MagicMock(return_value={
                    "choices": [{"message": {"content": "Adapter response"}}]
                }),
            ),
        ]
        assert server.load_adapter("stack-03", "/path") is True
        result = server.generate("test", model="qwen3.5-35b", adapter="stack-03")
        assert result == "Adapter response"


# ---------------------------------------------------------------------------
# Aeon hook pre/post inference
# ---------------------------------------------------------------------------


class TestAeonHookPreInference:
    """Test Aeon memory injection before inference."""

    def test_pre_inference_no_memories(self, mock_aeon_palace):
        hook = AeonServingHook(palace=mock_aeon_palace)
        result = hook.pre_inference("Hello world")
        assert result == "Hello world"  # unchanged

    def test_pre_inference_with_memories(self, mock_aeon_palace):
        @dataclass
        class FakeEpisode:
            content: str

        mock_aeon_palace.recall.return_value = [
            FakeEpisode(content="Previous conversation about Python"),
            FakeEpisode(content="User prefers concise answers"),
        ]
        hook = AeonServingHook(palace=mock_aeon_palace)
        result = hook.pre_inference("Write a sort function")
        assert "[Memory] Previous conversation about Python" in result
        assert "[Memory] User prefers concise answers" in result
        assert result.endswith("Write a sort function")

    def test_pre_inference_recall_error(self, mock_aeon_palace):
        mock_aeon_palace.recall.side_effect = RuntimeError("Connection lost")
        hook = AeonServingHook(palace=mock_aeon_palace)
        result = hook.pre_inference("Hello")
        assert result == "Hello"  # graceful fallback


class TestAeonHookPostInference:
    """Test Aeon memory writing after inference."""

    def test_post_inference_writes_memory(self, mock_aeon_palace):
        hook = AeonServingHook(palace=mock_aeon_palace)
        hook.post_inference(
            prompt="How to configure I2C?",
            response="Use pull-up resistors...",
            domain="research",
            turn_id="turn-001",
        )
        mock_aeon_palace.write.assert_called_once()
        call_kwargs = mock_aeon_palace.write.call_args[1]
        assert "I2C" in call_kwargs["content"]
        assert call_kwargs["domain"] == "research"

    def test_post_inference_write_error(self, mock_aeon_palace):
        mock_aeon_palace.write.side_effect = RuntimeError("Write failed")
        hook = AeonServingHook(palace=mock_aeon_palace)
        # Should not raise
        hook.post_inference("q", "a", "domain", "turn-002")


# ---------------------------------------------------------------------------
# Multi-domain routing via HTTP
# ---------------------------------------------------------------------------


class TestMultiDomainRouting:
    """Test routing across multiple domains with HTTP serving."""

    @patch("src.serving.vllm_server.httpx.post")
    def test_route_coding_then_serve(self, mock_post, server, mapping):
        mock_post.return_value = MagicMock(
            status_code=200,
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={
                "choices": [{"message": {"content": "code result"}}]
            }),
        )
        logits = [0.05] * 35
        logits[5] = 0.88  # rust
        result = dispatch(logits, mapping)
        assert result.intent == MetaIntent.CODING
        resp = server.generate("Rust trait", model="qwen3.5-35b", adapter="stack-06-rust")
        assert resp == "code result"

    @patch("src.serving.vllm_server.httpx.post")
    def test_route_research_then_serve(self, mock_post, server, mapping):
        mock_post.return_value = MagicMock(
            status_code=200,
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={
                "choices": [{"message": {"content": "PCB answer"}}]
            }),
        )
        logits = [0.05] * 35
        logits[31] = 0.85  # kicad-pcb
        result = dispatch(logits, mapping)
        assert result.intent == MetaIntent.RESEARCH
        resp = server.generate("KiCad layers", model="qwen3.5-35b")
        assert resp == "PCB answer"

    def test_server_config_defaults(self):
        config = VLLMServerConfig()
        assert config.port == 8100
        assert config.max_loras == 4
        assert config.max_lora_rank == 16

    def test_server_config_custom(self):
        config = VLLMServerConfig(port=9000, max_loras=8)
        assert config.port == 9000
        assert config.max_loras == 8

    def test_server_build_args(self, server):
        args = server._build_args()
        assert "--enable-lora" in args
        assert "--port" in args
        assert "9999" in args

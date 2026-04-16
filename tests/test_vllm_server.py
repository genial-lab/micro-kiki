"""Tests for VLLMServer — dynamic LoRA serving wrapper."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.serving.vllm_server import VLLMServer, VLLMServerConfig


class TestVLLMServerConfig:
    def test_defaults(self):
        cfg = VLLMServerConfig()
        assert cfg.port == 8100
        assert cfg.max_loras == 4
        assert cfg.max_lora_rank == 16
        assert cfg.gpu_memory_utilization == 0.9

    def test_frozen(self):
        cfg = VLLMServerConfig()
        with pytest.raises(AttributeError):
            cfg.port = 9000  # type: ignore[misc]

    def test_custom_values(self):
        cfg = VLLMServerConfig(
            model_path="my/model",
            port=9000,
            max_loras=2,
            max_lora_rank=8,
            gpu_memory_utilization=0.8,
        )
        assert cfg.model_path == "my/model"
        assert cfg.port == 9000


class TestVLLMServer:
    def test_default_config(self):
        server = VLLMServer()
        assert server.config.port == 8100

    def test_custom_config(self):
        cfg = VLLMServerConfig(port=9999)
        server = VLLMServer(config=cfg)
        assert server.config.port == 9999

    @patch("src.serving.vllm_server.httpx.post")
    def test_load_adapter_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        server = VLLMServer()
        result = server.load_adapter("stack-01", "/path/to/adapter")

        assert result is True
        mock_post.assert_called_once_with(
            "http://localhost:8100/v1/load_lora_adapter",
            json={"lora_name": "stack-01", "lora_path": "/path/to/adapter"},
            timeout=30.0,
        )

    @patch("src.serving.vllm_server.httpx.post")
    def test_load_adapter_failure(self, mock_post):
        mock_post.side_effect = httpx.ConnectError("refused")
        server = VLLMServer()
        result = server.load_adapter("stack-01", "/path")
        assert result is False

    @patch("src.serving.vllm_server.httpx.post")
    def test_unload_adapter_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        server = VLLMServer()
        result = server.unload_adapter("stack-01")

        assert result is True
        mock_post.assert_called_once_with(
            "http://localhost:8100/v1/unload_lora_adapter",
            json={"lora_name": "stack-01"},
            timeout=30.0,
        )

    @patch("src.serving.vllm_server.httpx.post")
    def test_unload_adapter_failure(self, mock_post):
        mock_post.side_effect = httpx.ConnectError("refused")
        server = VLLMServer()
        result = server.unload_adapter("stack-01")
        assert result is False

    @patch("src.serving.vllm_server.httpx.get")
    def test_health_ok(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        server = VLLMServer()
        result = server.health()
        assert result == {"status": "ok"}

    @patch("src.serving.vllm_server.httpx.get")
    def test_health_error(self, mock_get):
        mock_get.side_effect = httpx.ConnectError("refused")
        server = VLLMServer()
        result = server.health()
        assert result == {"status": "error"}

    @patch("src.serving.vllm_server.httpx.post")
    def test_generate_without_adapter(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "42"}}],
        }
        mock_post.return_value = mock_resp

        server = VLLMServer()
        result = server.generate("what is x?", model="qwen3.5")
        assert result == "42"
        call_payload = mock_post.call_args.kwargs["json"]
        assert call_payload["model"] == "qwen3.5"

    @patch("src.serving.vllm_server.httpx.post")
    def test_generate_with_adapter(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "bonjour"}}],
        }
        mock_post.return_value = mock_resp

        server = VLLMServer()
        result = server.generate("hi", model="qwen3.5", adapter="chat-fr")
        assert result == "bonjour"
        call_payload = mock_post.call_args.kwargs["json"]
        assert call_payload["model"] == "chat-fr"

    @patch("src.serving.vllm_server.httpx.post")
    def test_generate_failure_returns_empty(self, mock_post):
        mock_post.side_effect = httpx.ConnectError("refused")
        server = VLLMServer()
        result = server.generate("hi", model="qwen3.5")
        assert result == ""

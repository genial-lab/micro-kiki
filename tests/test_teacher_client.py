from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.distill.teacher_client import TeacherClient


@pytest.fixture
def client(tmp_path):
    return TeacherClient(
        endpoints={"mistral-large": "http://localhost:8000/v1"},
        cache_dir=str(tmp_path / "cache"),
    )


class TestTeacherClient:
    @pytest.mark.asyncio
    async def test_generate_returns_completion(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"choices": [{"message": {"content": "Bonjour!"}}]}
        mock_resp.raise_for_status = MagicMock()
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await client.generate(prompt="Dis bonjour", model="mistral-large")
        assert result == "Bonjour!"

    @pytest.mark.asyncio
    async def test_cache_hit_skips_http(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"choices": [{"message": {"content": "Cached"}}]}
        mock_resp.raise_for_status = MagicMock()
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp) as mock_post:
            r1 = await client.generate(prompt="test", model="mistral-large")
            r2 = await client.generate(prompt="test", model="mistral-large")
        assert r1 == r2 == "Cached"
        assert mock_post.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_500(self, client):
        fail = MagicMock()
        fail.raise_for_status.side_effect = Exception("Server error")
        ok = MagicMock()
        ok.status_code = 200
        ok.json.return_value = {"choices": [{"message": {"content": "OK"}}]}
        ok.raise_for_status = MagicMock()
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=[fail, ok]):
            result = await client.generate(prompt="test", model="mistral-large")
        assert result == "OK"

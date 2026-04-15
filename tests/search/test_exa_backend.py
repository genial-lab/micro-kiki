from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.search.exa_backend import ExaBackend
from src.search.base import SearchResult


@pytest.fixture
def exa():
    return ExaBackend(api_key="test-key")


class TestExaBackend:
    def test_name(self, exa):
        assert exa.name == "exa"

    @pytest.mark.asyncio
    async def test_search_returns_results(self, exa):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"title": "MoE Tutorial", "url": "https://example.com/moe", "text": "A guide to MoE"}
            ]
        }
        mock_response.raise_for_status = MagicMock()
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
            results = await exa.search("MoE-LoRA training", max_results=5)
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].title == "MoE Tutorial"
        assert results[0].source == "exa"

    @pytest.mark.asyncio
    async def test_search_handles_empty_response(self, exa):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
            results = await exa.search("nonexistent topic")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_raises_on_api_error(self, exa):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("API error")
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
            with pytest.raises(Exception, match="API error"):
                await exa.search("query")

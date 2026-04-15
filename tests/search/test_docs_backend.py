from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.search.docs_backend import DocsBackend
from src.search.base import SearchResult


@pytest.fixture
def docs(tmp_path):
    return DocsBackend(index_path=str(tmp_path / "docs_index.sqlite"))


class TestDocsBackend:
    def test_name(self, docs):
        assert docs.name == "docs"

    @pytest.mark.asyncio
    async def test_scrape_and_index(self, docs):
        html = "<html><body><h1>ESP-IDF API</h1><p>GPIO functions for ESP32</p></body></html>"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = html
        mock_response.raise_for_status = MagicMock()
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response):
            await docs.index_url("https://docs.espressif.com/gpio.html")
        results = await docs.search("ESP32 GPIO")
        assert len(results) >= 1
        assert results[0].source == "docs"

    @pytest.mark.asyncio
    async def test_search_empty_index(self, docs):
        results = await docs.search("anything")
        assert results == []

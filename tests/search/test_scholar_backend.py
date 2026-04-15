from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.search.scholar_backend import ScholarBackend
from src.search.base import SearchResult


@pytest.fixture
def scholar():
    return ScholarBackend()


class TestScholarBackend:
    def test_name(self, scholar):
        assert scholar.name == "scholar"

    @pytest.mark.asyncio
    async def test_search_returns_papers(self, scholar):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {
                    "title": "MoLoRA: MoE-LoRA for LLMs",
                    "abstract": "We propose MoLoRA...",
                    "url": "https://arxiv.org/abs/2603.15965",
                    "year": 2026,
                    "citationCount": 42,
                    "authors": [{"name": "Author A"}],
                    "paperId": "abc123",
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response):
            results = await scholar.search("MoE-LoRA mixture of experts")
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].source == "scholar"
        assert results[0].metadata["year"] == 2026
        assert results[0].metadata["citations"] == 42

    @pytest.mark.asyncio
    async def test_search_handles_empty(self, scholar):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status = MagicMock()
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response):
            results = await scholar.search("nonexistent")
        assert results == []

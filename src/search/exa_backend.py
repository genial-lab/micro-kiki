from __future__ import annotations

import httpx
from src.search.base import SearchBackend, SearchResult


class ExaBackend(SearchBackend):
    """Exa API web search backend."""

    def __init__(self, api_key: str, base_url: str = "https://api.exa.ai") -> None:
        self._api_key = api_key
        self._base_url = base_url

    @property
    def name(self) -> str:
        return "exa"

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._base_url}/search",
                json={
                    "query": query,
                    "num_results": max_results,
                    "type": "auto",
                    "contents": {"text": True},
                },
                headers={"x-api-key": self._api_key},
                timeout=30.0,
            )
            response.raise_for_status()

        data = response.json()
        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("text", "")[:500],
                source="exa",
                metadata={"score": r.get("score")},
            )
            for r in data.get("results", [])
        ]

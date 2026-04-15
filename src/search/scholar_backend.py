from __future__ import annotations

import httpx
from src.search.base import SearchBackend, SearchResult


class ScholarBackend(SearchBackend):
    """Semantic Scholar API search backend."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    FIELDS = "title,abstract,url,year,citationCount,authors"

    def __init__(self, base_url: str | None = None) -> None:
        self._base_url = base_url or self.BASE_URL

    @property
    def name(self) -> str:
        return "scholar"

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self._base_url}/paper/search",
                params={
                    "query": query,
                    "limit": max_results,
                    "fields": self.FIELDS,
                },
                timeout=30.0,
            )
            response.raise_for_status()

        data = response.json()
        return [
            SearchResult(
                title=p.get("title", ""),
                url=p.get("url", ""),
                snippet=(p.get("abstract") or "")[:500],
                source="scholar",
                metadata={
                    "year": p.get("year"),
                    "citations": p.get("citationCount", 0),
                    "authors": [a["name"] for a in p.get("authors", [])],
                    "paper_id": p.get("paperId"),
                },
            )
            for p in data.get("data", [])
        ]

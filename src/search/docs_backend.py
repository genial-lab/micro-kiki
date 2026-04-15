from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import httpx
from src.search.base import SearchBackend, SearchResult


class DocsBackend(SearchBackend):
    """Targeted doc scraper with local SQLite FTS index."""

    def __init__(self, index_path: str = "data/docs_index.sqlite") -> None:
        self._index_path = index_path
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(index_path)
        self._conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS docs
            USING fts5(url, title, content, tokenize='porter')
            """
        )
        self._conn.commit()

    @property
    def name(self) -> str:
        return "docs"

    @staticmethod
    def _extract_text(html: str) -> tuple[str, str]:
        title_match = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.DOTALL)
        title = re.sub(r"<[^>]+>", "", title_match.group(1)).strip() if title_match else ""
        body = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
        body = re.sub(r"<style[^>]*>.*?</style>", "", body, flags=re.DOTALL)
        body = re.sub(r"<[^>]+>", " ", body)
        body = re.sub(r"\s+", " ", body).strip()
        return title, body

    async def index_url(self, url: str) -> None:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()

        title, content = self._extract_text(response.text)
        self._conn.execute("DELETE FROM docs WHERE url = ?", (url,))
        self._conn.execute(
            "INSERT INTO docs (url, title, content) VALUES (?, ?, ?)",
            (url, title, content),
        )
        self._conn.commit()

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        rows = self._conn.execute(
            "SELECT url, title, snippet(docs, 2, '<b>', '</b>', '...', 64) "
            "FROM docs WHERE docs MATCH ? LIMIT ?",
            (query, max_results),
        ).fetchall()
        return [
            SearchResult(
                title=row[1],
                url=row[0],
                snippet=re.sub(r"<[^>]+>", "", row[2]),
                source="docs",
                metadata={},
            )
            for row in rows
        ]

    def close(self) -> None:
        self._conn.close()

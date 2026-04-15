from __future__ import annotations

import logging
from src.search.base import SearchResult

logger = logging.getLogger(__name__)


class AeonIndexer:
    """Indexes search results into Aeon memory for long-term enrichment."""

    def __init__(self, aeon_client) -> None:
        self._aeon = aeon_client

    async def index_results(
        self,
        query: str,
        results: list[SearchResult],
        session_id: str,
    ) -> int:
        indexed = 0
        for result in results:
            trace = {
                "type": "search_result",
                "query": query,
                "title": result.title,
                "url": result.url,
                "snippet": result.snippet,
                "source": result.source,
                "session_id": session_id,
                "metadata": result.metadata,
            }
            await self._aeon.store_trace(trace)
            indexed += 1
        logger.info("Indexed %d search results into Aeon for query: %s", indexed, query[:50])
        return indexed

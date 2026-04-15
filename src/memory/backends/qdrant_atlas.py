"""Qdrant-backed Atlas index (remote backend)."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class QdrantAtlas:
    """Atlas backend using Qdrant for vector storage.

    Pluggable replacement for AtlasIndex when AEON_BACKEND=remote.
    Requires: pip install qdrant-client
    """

    def __init__(self, url: str = "http://localhost:6333", collection: str = "aeon") -> None:
        self._url = url
        self._collection = collection
        self._client = None

    def _connect(self):
        from qdrant_client import QdrantClient
        self._client = QdrantClient(url=self._url)

    def insert(self, vector_id: str, vector, metadata: dict | None = None) -> None:
        if self._client is None:
            self._connect()
        from qdrant_client.models import PointStruct
        self._client.upsert(
            collection_name=self._collection,
            points=[PointStruct(id=vector_id, vector=vector.tolist(), payload=metadata or {})],
        )

    def recall(self, query, top_k: int = 10) -> list[tuple[str, float]]:
        if self._client is None:
            self._connect()
        results = self._client.search(
            collection_name=self._collection,
            query_vector=query.tolist(),
            limit=top_k,
        )
        return [(r.id, r.score) for r in results]

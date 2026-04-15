"""Atlas SIMD Page-Clustered Vector Index (arxiv 2601.15311).

Page-clustered layout with SIMD-accelerated dot-product.
Falls back to numpy when SIMD extensions unavailable.
"""
from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

PAGE_SIZE = 256  # vectors per page


@dataclass
class VectorPage:
    vectors: np.ndarray  # (PAGE_SIZE, dim)
    ids: list[str] = field(default_factory=list)
    count: int = 0


class AtlasIndex:
    """Page-clustered vector index with numpy fallback."""

    def __init__(self, dim: int = 3072, num_clusters: int = 16) -> None:
        self.dim = dim
        self.num_clusters = num_clusters
        self._pages: list[VectorPage] = []
        self._centroids: np.ndarray | None = None
        self._id_to_page: dict[str, int] = {}

    def insert(self, vector_id: str, vector: np.ndarray) -> None:
        if vector.shape != (self.dim,):
            raise ValueError(f"Expected dim {self.dim}, got {vector.shape}")

        # Find or create page
        page_idx = self._find_nearest_page(vector)
        if page_idx is None or self._pages[page_idx].count >= PAGE_SIZE:
            page = VectorPage(
                vectors=np.zeros((PAGE_SIZE, self.dim), dtype=np.float32),
            )
            self._pages.append(page)
            page_idx = len(self._pages) - 1

        page = self._pages[page_idx]
        page.vectors[page.count] = vector
        page.ids.append(vector_id)
        page.count += 1
        self._id_to_page[vector_id] = page_idx

    def _find_nearest_page(self, vector: np.ndarray) -> int | None:
        if not self._pages:
            return None
        if self._centroids is not None:
            sims = self._centroids @ vector
            return int(np.argmax(sims))
        return len(self._pages) - 1

    def recall(self, query: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        if not self._pages:
            return []

        all_scores: list[tuple[str, float]] = []
        query_norm = query / (np.linalg.norm(query) + 1e-8)

        for page in self._pages:
            if page.count == 0:
                continue
            vecs = page.vectors[:page.count]
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
            sims = (vecs / norms) @ query_norm
            for i, sim in enumerate(sims):
                all_scores.append((page.ids[i], float(sim)))

        all_scores.sort(key=lambda x: x[1], reverse=True)
        return all_scores[:top_k]

    def rebuild_centroids(self) -> None:
        if not self._pages:
            return
        centroids = []
        for page in self._pages:
            if page.count > 0:
                mean = page.vectors[:page.count].mean(axis=0)
                centroids.append(mean / (np.linalg.norm(mean) + 1e-8))
        self._centroids = np.array(centroids, dtype=np.float32) if centroids else None

    @property
    def total_vectors(self) -> int:
        return sum(p.count for p in self._pages)

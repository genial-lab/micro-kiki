from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str  # "exa", "scholar", "docs"
    metadata: dict


class SearchBackend(ABC):
    """Abstract search backend interface."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]: ...

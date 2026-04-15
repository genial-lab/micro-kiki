from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path


class SearchCache:
    """SQLite-backed search result cache with per-entry TTL."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                results TEXT NOT NULL,
                expires_at REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    @staticmethod
    def _make_key(backend: str, query: str) -> str:
        raw = f"{backend}:{query}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def store(
        self,
        backend: str,
        query: str,
        results: list[dict],
        ttl_seconds: int,
    ) -> None:
        key = self._make_key(backend, query)
        expires_at = time.time() + ttl_seconds
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (key, results, expires_at) VALUES (?, ?, ?)",
            (key, json.dumps(results), expires_at),
        )
        self._conn.commit()

    def lookup(self, backend: str, query: str) -> list[dict] | None:
        key = self._make_key(backend, query)
        row = self._conn.execute(
            "SELECT results, expires_at FROM cache WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        results, expires_at = row
        if time.time() > expires_at:
            self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            self._conn.commit()
            return None
        return json.loads(results)

    def close(self) -> None:
        self._conn.close()

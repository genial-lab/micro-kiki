"""Neo4j-backed Trace graph (remote backend)."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class Neo4jTrace:
    """Trace backend using Neo4j for graph storage.

    Pluggable replacement for TraceGraph when AEON_BACKEND=remote.
    Requires: pip install neo4j
    """

    def __init__(self, uri: str = "bolt://localhost:7687", auth: tuple = ("neo4j", "password")) -> None:
        self._uri = uri
        self._auth = auth
        self._driver = None

    def _connect(self):
        from neo4j import GraphDatabase
        self._driver = GraphDatabase.driver(self._uri, auth=self._auth)

    def add_episode(self, episode_id: str, content: str, domain: str, timestamp: str) -> None:
        if self._driver is None:
            self._connect()
        with self._driver.session() as session:
            session.run(
                "MERGE (e:Episode {id: $id}) SET e.content = $content, e.domain = $domain, e.timestamp = $ts",
                id=episode_id, content=content, domain=domain, ts=timestamp,
            )

    def add_edge(self, from_id: str, to_id: str, weight: float, relation: str = "causes") -> None:
        if self._driver is None:
            self._connect()
        with self._driver.session() as session:
            session.run(
                "MATCH (a:Episode {id: $from_id}), (b:Episode {id: $to_id}) "
                "MERGE (a)-[r:CAUSES {weight: $weight}]->(b)",
                from_id=from_id, to_id=to_id, weight=weight,
            )

    def walk(self, from_id: str, max_depth: int = 3) -> list[dict]:
        if self._driver is None:
            self._connect()
        with self._driver.session() as session:
            result = session.run(
                f"MATCH path = (e:Episode {{id: $from_id}})-[:CAUSES*1..{max_depth}]->(n) "
                "RETURN n.id AS id, n.content AS content, n.domain AS domain",
                from_id=from_id,
            )
            return [dict(record) for record in result]

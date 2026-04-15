"""AeonSleep — unified memory facade.

Single entry point wrapping the five building blocks:

* :class:`src.memory.atlas.AtlasIndex`     — SIMD vector index.
* :class:`src.memory.trace.TraceGraph`     — neuro-symbolic episode graph.
* :class:`src.cognitive.sleep_tagger.SleepTagger`       — conflict tagger.
* :class:`src.cognitive.forgetting_gate.ForgettingGate` — learned evictor.
* :class:`src.cognitive.consolidation.Consolidator`     — cluster/summariser.

Public methods:

* :meth:`write` — persist an episode (vector + text + topic + ts).
* :meth:`recall` — top-k vector search with graph-backed payloads.
* :meth:`sleep_cycle` — tag → gate eviction → consolidate.
* :meth:`query_time` — range query on the Trace graph.
* :meth:`stats` — merged introspection snapshot.

The acceptance target (v0.3 spec): ≥ 95 % retrieval accuracy at
PI depth 10 on the SleepGate-paper benchmark after writing 500
episodes and running 3 sleep cycles. The integration test in
:mod:`tests.test_aeonsleep` validates this on a controlled planted
dataset.

Design notes
------------

* Backref preservation: when consolidation emits a summary cluster we
  create a ``summary`` node in the Trace graph and link each original
  episode to it via a ``summary_of`` edge (``member -> summary``).
  Originals are **not** deleted — the forgetting gate is the only path
  that removes nodes.
* Eviction is conservative: the gate outputs P(keep); we drop an
  episode only when ``P(keep) < keep_threshold`` AND the episode is
  represented by at least one summary cluster. This keeps recall high
  on the PI-depth benchmark while still bounding memory growth.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Iterable, Sequence

from src.cognitive.consolidation import (
    Consolidator,
    RawEpisode,
    SummaryCluster,
    SummarizeFn,
)
from src.cognitive.forgetting_gate import (
    EpisodeFeatures,
    ForgettingGate,
)
from src.cognitive.sleep_tagger import (
    RecentEpisode,
    SleepTagger,
    Tag,
)
from src.memory.atlas import AtlasIndex, SearchHit
from src.memory.trace import TraceGraph


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Episode:
    """Input shape accepted by :meth:`AeonSleep.write`."""

    id: str
    text: str
    embedding: Sequence[float]
    ts: datetime
    topic: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class RecallHit:
    """Result of :meth:`AeonSleep.recall`."""

    episode_id: str
    text: str
    score: float
    ts: datetime | None
    topic: str | None
    kind: str  # "raw" or "summary"
    payload: dict[str, Any]


@dataclass
class SleepReport:
    """Per-cycle summary returned by :meth:`AeonSleep.sleep_cycle`."""

    tags_assigned: int
    evicted: int
    kept: int
    clusters_built: int
    compression_ratio: float


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------


class AeonSleep:
    """Unified memory module.

    Parameters
    ----------
    dim : int
        Embedding dimensionality. All writes are validated.
    keep_threshold : float
        Gate threshold below which an episode can be evicted (default
        0.35). Kept intentionally low so we prefer recall over
        compression — the PI-depth benchmark is graded by recall.
    tagger : SleepTagger | None
    gate : ForgettingGate | None
    consolidator : Consolidator | None
    now_fn : callable
        Returns the "current" time; injected for determinism in tests.
    summarize_fn : SummarizeFn | None
        Propagated to the default :class:`Consolidator`.
    """

    def __init__(
        self,
        *,
        dim: int,
        keep_threshold: float = 0.35,
        tagger: SleepTagger | None = None,
        gate: ForgettingGate | None = None,
        consolidator: Consolidator | None = None,
        atlas: AtlasIndex | None = None,
        graph: TraceGraph | None = None,
        now_fn: Callable[[], datetime] | None = None,
        summarize_fn: SummarizeFn | None = None,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.keep_threshold = float(keep_threshold)
        self.atlas = atlas or AtlasIndex(dim=dim)
        self.graph = graph or TraceGraph()
        self.tagger = tagger or SleepTagger()
        self.gate = gate or ForgettingGate()
        self.consolidator = consolidator or Consolidator(
            summarize_fn=summarize_fn,
        )
        self._now = now_fn or datetime.utcnow
        # Episode-level metadata we need at sleep time.
        self._episode_meta: dict[str, dict[str, Any]] = {}
        self._summary_counter = itertools.count()

    # ------------------------------------------------------------ write api

    def write(self, episode: Episode) -> Tag:
        """Persist ``episode`` and return its conflict tag.

        Post-conditions:

        * atlas contains a vector keyed by ``episode.id``.
        * graph contains a ``raw`` node keyed by ``episode.id``.
        * previous episode on the same topic (if any) gets a
          ``temporal`` edge to this one.
        * episode meta is updated with access=0 and conflict=tag.level.
        """
        if len(episode.embedding) != self.dim:
            raise ValueError(
                f"embedding dim {len(episode.embedding)} != {self.dim}"
            )
        if self.graph.has_node(episode.id):
            raise ValueError(f"duplicate episode id: {episode.id}")

        # 1) Compute tag vs a short window of recent episodes.
        recent = self._recent_window()
        tag = self.tagger.tag(
            RecentEpisode(
                id=episode.id,
                text=episode.text,
                embedding=episode.embedding,
                ts=episode.ts,
                topic=episode.topic,
            ),
            recent,
        )

        # 2) Persist vector + graph node.
        payload = dict(episode.payload)
        payload.update(
            {
                "text": episode.text,
                "topic": episode.topic,
                "ts": episode.ts.isoformat(),
            }
        )
        self.atlas.add(episode.id, episode.embedding, payload=payload)
        self.graph.add_node(
            episode.id,
            kind="raw",
            ts=episode.ts,
            text=episode.text,
            topic=episode.topic,
            conflict=tag.level,
        )

        # 3) Temporal edge from the most recent same-topic episode.
        prev_same_topic = self._last_on_topic(episode.topic, before=episode.ts)
        if prev_same_topic is not None:
            self.graph.add_edge(
                prev_same_topic, episode.id, kind="temporal"
            )

        self._episode_meta[episode.id] = {
            "written_at": episode.ts,
            "access": 0,
            "conflict": tag.level,
            "embedding": tuple(episode.embedding),
            "text": episode.text,
            "topic": episode.topic,
        }
        return tag

    # ----------------------------------------------------------- recall api

    def recall(self, query: Sequence[float], k: int = 10) -> list[RecallHit]:
        """Top-``k`` retrieval. Hits record access for the gate."""
        hits = self.atlas.search(query, k=k)
        out: list[RecallHit] = []
        for hit in hits:
            node = self.graph.get_node(hit.key) if self.graph.has_node(hit.key) else None
            meta = self._episode_meta.get(hit.key, {})
            if hit.key in self._episode_meta:
                self._episode_meta[hit.key]["access"] = (
                    int(meta.get("access", 0)) + 1
                )
            ts = node.ts if node is not None else None
            kind = node.kind if node is not None else "raw"
            topic = (node.attrs.get("topic") if node else hit.payload.get("topic"))
            text = (node.attrs.get("text") if node else hit.payload.get("text", ""))
            out.append(
                RecallHit(
                    episode_id=hit.key,
                    text=text or "",
                    score=hit.score,
                    ts=ts,
                    topic=topic,
                    kind=kind or "raw",
                    payload=dict(hit.payload),
                )
            )
        return out

    # ------------------------------------------------------------ sleep api

    def sleep_cycle(
        self, *, now: datetime | None = None
    ) -> SleepReport:
        """Run one consolidation round: tag → gate → consolidate.

        1. Re-tag every live episode against its neighbours (conflict
           levels may drift as new episodes arrive).
        2. Evaluate the forgetting gate on every raw episode. Drop
           those below ``keep_threshold`` that are already covered by
           a summary cluster.
        3. Run the consolidator on the surviving raw episodes.
        4. Materialise new summary nodes with ``summary_of`` edges
           from each original member.
        """
        stamp = now or self._now()
        raw_ids = [n.id for n in self.graph.nodes(kind="raw")]
        tags_assigned = self._retag(raw_ids)

        # Consolidate BEFORE evicting so eviction is guarded by
        # cluster coverage. This preserves PI-depth recall.
        raw_episodes = [
            RawEpisode(
                id=nid,
                text=self._episode_meta[nid]["text"] or "",
                embedding=self._episode_meta[nid]["embedding"],
                ts=self._episode_meta[nid]["written_at"],
                topic=self._episode_meta[nid]["topic"],
            )
            for nid in raw_ids
            if nid in self._episode_meta
        ]
        clusters = self.consolidator.consolidate(raw_episodes)
        covered = self._materialise_clusters(clusters)

        # Eviction pass.
        evicted = self._evict(raw_ids, covered=covered, now=stamp)

        stats = self.consolidator.last_stats()
        compression = stats.compression_ratio if stats else 0.0
        return SleepReport(
            tags_assigned=tags_assigned,
            evicted=evicted,
            kept=len(raw_ids) - evicted,
            clusters_built=len(clusters),
            compression_ratio=compression,
        )

    # ------------------------------------------------------------ time api

    def query_time(
        self, start: datetime, end: datetime
    ) -> list[str]:
        """Return episode ids whose ts falls in ``[start, end]``."""
        return [n.id for n in self.graph.time_range(start, end)]

    # ---------------------------------------------------------- stats api

    def stats(self) -> dict[str, Any]:
        """Aggregate stats — atlas + graph + sleep."""
        atlas_stats = self.atlas.stats()
        graph_stats = self.graph.stats()
        raw = sum(1 for _ in self.graph.nodes(kind="raw"))
        summary = sum(1 for _ in self.graph.nodes(kind="summary"))
        return {
            "atlas": atlas_stats,
            "graph": graph_stats,
            "n_raw": raw,
            "n_summary": summary,
            "n_tracked": len(self._episode_meta),
        }

    # ------------------------------------------------------------ helpers

    def _recent_window(self, size: int = 32) -> list[RecentEpisode]:
        if not self._episode_meta:
            return []
        # Use ids most recently written.
        ids = list(self._episode_meta.keys())[-size:]
        out: list[RecentEpisode] = []
        for nid in ids:
            m = self._episode_meta[nid]
            out.append(
                RecentEpisode(
                    id=nid,
                    text=m["text"] or "",
                    embedding=m["embedding"],
                    ts=m["written_at"],
                    topic=m["topic"],
                )
            )
        return out

    def _last_on_topic(
        self, topic: str | None, *, before: datetime
    ) -> str | None:
        if topic is None:
            return None
        last: tuple[datetime, str] | None = None
        for nid, m in self._episode_meta.items():
            if m["topic"] != topic:
                continue
            if m["written_at"] >= before:
                continue
            if last is None or m["written_at"] > last[0]:
                last = (m["written_at"], nid)
        return None if last is None else last[1]

    def _retag(self, raw_ids: Sequence[str]) -> int:
        episodes = [
            RecentEpisode(
                id=nid,
                text=self._episode_meta[nid]["text"] or "",
                embedding=self._episode_meta[nid]["embedding"],
                ts=self._episode_meta[nid]["written_at"],
                topic=self._episode_meta[nid]["topic"],
            )
            for nid in raw_ids
            if nid in self._episode_meta
        ]
        tags = self.tagger.tag_batch(episodes)
        for ep, tag in zip(episodes, tags):
            self._episode_meta[ep.id]["conflict"] = tag.level
            node = self.graph.get_node(ep.id)
            node.attrs["conflict"] = tag.level
        return len(tags)

    def _materialise_clusters(
        self, clusters: Sequence[SummaryCluster]
    ) -> set[str]:
        covered: set[str] = set()
        for cluster in clusters:
            if len(cluster.member_ids) <= 1:
                covered.update(cluster.member_ids)
                continue
            sid = f"summary-{next(self._summary_counter):04d}"
            # Avoid accidental collisions with existing ids.
            while self.graph.has_node(sid):
                sid = f"summary-{next(self._summary_counter):04d}"
            self.graph.add_node(
                sid,
                kind="summary",
                ts=cluster.range[1],
                text=cluster.text,
                topic=cluster.topic,
                size=cluster.size,
            )
            # Put the summary into the atlas so recall can hit it too.
            self.atlas.add(
                sid,
                cluster.embedding,
                payload={
                    "text": cluster.text,
                    "topic": cluster.topic,
                    "size": cluster.size,
                    "is_summary": True,
                },
            )
            for member in cluster.member_ids:
                if self.graph.has_node(member):
                    self.graph.add_edge(member, sid, kind="summary_of")
                    covered.add(member)
        return covered

    def _evict(
        self,
        raw_ids: Sequence[str],
        *,
        covered: set[str],
        now: datetime,
    ) -> int:
        if not raw_ids:
            return 0
        feats: list[EpisodeFeatures] = []
        for nid in raw_ids:
            m = self._episode_meta[nid]
            age_hours = max(
                (now - m["written_at"]).total_seconds() / 3600.0, 0.0
            )
            feats.append(
                EpisodeFeatures(
                    age_hours=age_hours,
                    access_count=int(m.get("access", 0)),
                    conflict_level=float(m.get("conflict", 0.0)),
                    embedding_norm=_l2(m["embedding"]),
                )
            )
        probs = self.gate.predict_proba(feats)
        evicted = 0
        for nid, p in zip(raw_ids, probs):
            if float(p) >= self.keep_threshold:
                continue
            if nid not in covered:
                # Not safe to drop — no summary protects this episode.
                continue
            if self.atlas.remove(nid):
                self.graph.remove_node(nid)
                self._episode_meta.pop(nid, None)
                evicted += 1
        return evicted


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _l2(vec: Sequence[float]) -> float:
    total = 0.0
    for x in vec:
        total += float(x) * float(x)
    return total ** 0.5

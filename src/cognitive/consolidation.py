"""Consolidation — cluster & summarize Trace episodes.

Implements the "sleep consolidation" step of AeonSleep. Given a pool
of raw episodes (texts + embeddings + timestamps + topics), produce a
set of summary nodes with backrefs to the original episodes so the
originals remain reachable through the Trace graph.

Two clustering axes:

* **Topic** — cosine similarity on embeddings (single-link, threshold
  configurable).
* **Temporal proximity** — episodes within a rolling window must share
  the cluster before a new cluster is spawned.

The summary text is produced by a pluggable ``summarize_fn``. The
default is a naive top-k token-frequency extractor that returns the
most salient sentences joined back together — fast, deterministic, no
network. A real teacher-LLM backend (Qwen3.5-35B on kxkm-ai :8000) is
documented as a TODO and can be injected at runtime; we do not issue
network calls from this module.

API::

    consolidator = Consolidator(topic_threshold=0.75)
    summaries = consolidator.consolidate(episodes)
    # Each SummaryCluster has .summary_id, .text, .member_ids, .topic,
    # .range (ts_start, ts_end), .embedding.

Backref preservation is handled upstream by :class:`AeonSleep` which
adds ``summary_of`` edges in the Trace graph — this module only
produces the clusters.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Iterable, Sequence

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class RawEpisode:
    """Minimal input shape for consolidation."""

    id: str
    text: str
    embedding: Sequence[float]
    ts: datetime
    topic: str | None = None


@dataclass
class SummaryCluster:
    """Output of a single consolidated cluster."""

    summary_id: str
    text: str
    member_ids: list[str]
    topic: str | None
    range: tuple[datetime, datetime]
    embedding: tuple[float, ...]
    size: int = field(init=False)

    def __post_init__(self) -> None:
        self.size = len(self.member_ids)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD_RE = re.compile(r"[\w']+")
_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "but", "in", "on", "at",
    "to", "for", "is", "are", "was", "were", "be", "been", "being",
    "it", "this", "that", "with", "by", "as", "from", "has", "have",
    "had", "will", "would", "can", "could", "should", "may", "might",
    "le", "la", "les", "de", "du", "des", "et", "ou", "est", "une",
    "un", "qui", "que",
}


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _mean_vec(vectors: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if not vectors:
        return ()
    dim = len(vectors[0])
    out = [0.0] * dim
    for v in vectors:
        for i in range(dim):
            out[i] += float(v[i])
    n = float(len(vectors))
    for i in range(dim):
        out[i] /= n
    # renormalise (stable comparison to normalised embeddings)
    norm = math.sqrt(sum(x * x for x in out)) or 1.0
    return tuple(x / norm for x in out)


# ---------------------------------------------------------------------------
# Summarisation backends
# ---------------------------------------------------------------------------


def heuristic_summary(
    texts: Sequence[str],
    *,
    max_sentences: int = 2,
    max_chars: int = 400,
) -> str:
    """Naive extractive summary: pick sentences with highest token mass.

    Deterministic, zero dependencies, no network. Good enough to keep
    the recall-probe acceptance test green in offline dev. Mark as the
    default ``summarize_fn`` — callers can inject an LLM backend later.
    """
    if not texts:
        return ""

    # Global token frequency excluding stopwords.
    freq: Counter[str] = Counter()
    for text in texts:
        for tok in _WORD_RE.findall(text.lower()):
            if tok in _STOPWORDS or tok.isdigit():
                continue
            freq[tok] += 1

    # Flatten candidate sentences with their originating text index.
    candidates: list[tuple[float, int, str]] = []
    for text in texts:
        for sent in _SENT_SPLIT.split(text.strip()):
            sent = sent.strip()
            if not sent:
                continue
            tokens = _WORD_RE.findall(sent.lower())
            if not tokens:
                continue
            score = sum(freq[t] for t in tokens if t not in _STOPWORDS)
            # Penalise very long sentences so we keep summaries tight.
            score = score / max(len(tokens), 1)
            # Preserve insertion order for deterministic tiebreak.
            candidates.append((score, len(candidates), sent))

    if not candidates:
        # Fallback to joined prefix.
        joined = " | ".join(t.strip() for t in texts if t.strip())
        return joined[:max_chars]

    candidates.sort(key=lambda row: (-row[0], row[1]))
    chosen: list[tuple[int, str]] = []
    seen: set[str] = set()
    for _score, order, sent in candidates:
        norm = sent.lower().strip()
        if norm in seen:
            continue
        chosen.append((order, sent))
        seen.add(norm)
        if len(chosen) >= max_sentences:
            break
    chosen.sort(key=lambda row: row[0])
    out = " ".join(sent for _order, sent in chosen)
    if len(out) > max_chars:
        out = out[: max_chars - 1].rstrip() + "…"
    return out


SummarizeFn = Callable[[Sequence[str]], str]


# TODO(v0.3): wire Qwen3.5-35B-A3B at kxkm-ai:8000 via the ssh tunnel
# for higher-quality summaries. The callable contract is the same as
# :func:`heuristic_summary` — we keep heuristic as the offline default.


# ---------------------------------------------------------------------------
# Consolidator
# ---------------------------------------------------------------------------


@dataclass
class ConsolidationStats:
    """Reporting shape produced by :meth:`Consolidator.consolidate`."""

    n_input: int
    n_clusters: int
    compression_ratio: float
    topics_seen: int


class Consolidator:
    """Cluster episodes and build summary nodes.

    Parameters
    ----------
    topic_threshold : float
        Cosine similarity above which an episode joins an existing
        cluster centroid. Lower → fewer, bigger clusters.
    temporal_window : timedelta
        A candidate cluster is considered "active" only if its latest
        episode is within this window of the incoming episode. Older
        clusters are closed and will not receive new members.
    summarize_fn : SummarizeFn
        Pluggable summariser. Defaults to :func:`heuristic_summary`.
    summary_prefix : str
        Prefix stamped on generated summary IDs.
    """

    def __init__(
        self,
        *,
        topic_threshold: float = 0.7,
        temporal_window: timedelta = timedelta(hours=24),
        summarize_fn: SummarizeFn | None = None,
        summary_prefix: str = "sum",
    ) -> None:
        if not 0.0 <= topic_threshold <= 1.0:
            raise ValueError("topic_threshold must be in [0, 1]")
        self.topic_threshold = topic_threshold
        self.temporal_window = temporal_window
        self.summarize_fn = summarize_fn or heuristic_summary
        self.summary_prefix = summary_prefix
        self._last_stats: ConsolidationStats | None = None

    # -------------------------------------------------------------- clustering

    def _assign(
        self,
        ep: RawEpisode,
        centroids: list[dict],
    ) -> int | None:
        """Return index of best matching cluster or ``None``."""
        best: tuple[float, int] | None = None
        for idx, cent in enumerate(centroids):
            # Temporal gate: cluster is "closed" if its newest member
            # predates the window. Strict same-topic restriction too.
            if ep.topic is not None and cent["topic"] is not None:
                if ep.topic != cent["topic"]:
                    continue
            if ep.ts - cent["ts_end"] > self.temporal_window:
                continue
            sim = _cosine(ep.embedding, cent["centroid"])
            if sim < self.topic_threshold:
                continue
            if best is None or sim > best[0]:
                best = (sim, idx)
        return None if best is None else best[1]

    def consolidate(
        self, episodes: Sequence[RawEpisode]
    ) -> list[SummaryCluster]:
        """Build summary clusters for ``episodes``.

        Episodes are processed in chronological order. Each episode
        either joins an open centroid (same topic, inside the temporal
        window, similarity ≥ threshold) or opens a new centroid.
        """
        ordered = sorted(episodes, key=lambda e: e.ts)
        centroids: list[dict] = []
        for ep in ordered:
            idx = self._assign(ep, centroids)
            if idx is None:
                centroids.append(
                    {
                        "members": [ep],
                        "texts": [ep.text],
                        "vecs": [tuple(ep.embedding)],
                        "topic": ep.topic,
                        "centroid": tuple(ep.embedding),
                        "ts_start": ep.ts,
                        "ts_end": ep.ts,
                    }
                )
            else:
                cent = centroids[idx]
                cent["members"].append(ep)
                cent["texts"].append(ep.text)
                cent["vecs"].append(tuple(ep.embedding))
                cent["centroid"] = _mean_vec(cent["vecs"])
                cent["ts_end"] = ep.ts
                if cent["topic"] is None:
                    cent["topic"] = ep.topic

        # Materialise SummaryCluster objects.
        clusters: list[SummaryCluster] = []
        topics: set[str] = set()
        for i, cent in enumerate(centroids):
            text = self.summarize_fn(cent["texts"])
            if cent["topic"] is not None:
                topics.add(cent["topic"])
            clusters.append(
                SummaryCluster(
                    summary_id=f"{self.summary_prefix}-{i:04d}",
                    text=text,
                    member_ids=[m.id for m in cent["members"]],
                    topic=cent["topic"],
                    range=(cent["ts_start"], cent["ts_end"]),
                    embedding=cent["centroid"],
                )
            )

        n_in = len(ordered)
        self._last_stats = ConsolidationStats(
            n_input=n_in,
            n_clusters=len(clusters),
            compression_ratio=(
                len(clusters) / max(n_in, 1)
            ),
            topics_seen=len(topics),
        )
        return clusters

    def last_stats(self) -> ConsolidationStats | None:
        return self._last_stats


# ---------------------------------------------------------------------------
# Recall probe utility (used by tests and by AeonSleep's sleep cycle)
# ---------------------------------------------------------------------------


def recall_via_summary(
    query: str,
    clusters: Sequence[SummaryCluster],
    *,
    top_k: int = 3,
) -> list[SummaryCluster]:
    """Rank clusters by shared tokens with ``query``.

    Used by the acceptance test to verify that key facts survive the
    consolidation step. This is intentionally a tiny lexical matcher
    so the probe does not depend on an embedding model.
    """
    q_tokens = {t for t in _WORD_RE.findall(query.lower()) if t not in _STOPWORDS}
    scored: list[tuple[int, int, SummaryCluster]] = []
    for idx, cl in enumerate(clusters):
        tokens = {
            t for t in _WORD_RE.findall(cl.text.lower())
            if t not in _STOPWORDS
        }
        overlap = len(q_tokens & tokens)
        if overlap == 0:
            continue
        scored.append((overlap, -idx, cl))
    scored.sort(key=lambda row: (-row[0], -row[1]))
    return [row[2] for row in scored[:top_k]]

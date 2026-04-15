# AeonSleep — unified memory module architecture

**Status**: v0.3 neuroscience branch, Phase N-II story-5.
**Scope**: Fuses v0.2 Aeon memory palace (Atlas SIMD + Trace graph)
with the SleepGate conflict-aware tagger (arxiv 2603.14517),
adds a learned forgetting gate and a consolidation loop.

---

## 1. Goals

1. Replace the v0.2 split between `Aeon` (write/recall) and a
   separate `SleepGate` stage with a single entry-point class
   `AeonSleep` that exposes five methods: `write`, `recall`,
   `sleep_cycle`, `query_time`, `stats`.
2. Preserve classical v0.2 code paths (Atlas hash-based vector
   search, Trace neuro-symbolic NetworkX graph) — these transfer
   verbatim from the v0.2 spec.
3. Add conflict-aware temporal tagging so the sleep cycle can
   identify contradictions, topic drift, and stale entries.
4. Add a small learned gate (2-hidden-layer MLP, pure numpy)
   that scores `P(keep)` for each episode from cheap features.
5. Add a consolidation step that clusters related episodes and
   emits a summary node while keeping backrefs so originals
   remain reachable.
6. Hit ≥ 95 % retrieval accuracy at PI depth 10 on the
   SleepGate-paper synthetic benchmark.

## 2. Architecture

```
+----------------------+   write()    +-------------------+
|  caller              |------------->|  AeonSleep        |
+----------------------+              |                   |
                                      |  ┌─────────────┐  |
                                      |  | Atlas SIMD  |  |  spatial vec index
                                      |  └─────────────┘  |
                                      |  ┌─────────────┐  |
                                      |  | Trace graph |  |  episodic nodes+edges
                                      |  └─────────────┘  |
                                      |  ┌─────────────┐  |
                                      |  | SleepTagger |  |  conflict levels
                                      |  └─────────────┘  |
                                      |  ┌─────────────┐  |
                                      |  | ForgetGate  |  |  MLP P(keep)
                                      |  └─────────────┘  |
                                      |  ┌─────────────┐  |
                                      |  | Consolid.   |  |  summaries
                                      |  └─────────────┘  |
                                      +-------------------+
```

Write path: `write(episode)` inserts into Atlas (vector) and
Trace (graph), runs SleepTagger to annotate `conflict_level`
and `reason`, stores the tag on the Trace node metadata.

Read path: `recall(query, k=10)` embeds the query, fetches
candidate episodes from Atlas top-K, intersects with Trace to
filter by causal / topical edges, returns ranked episodes.

Sleep cycle: `sleep_cycle()` runs (a) re-tagging of recent
episodes, (b) ForgetGate scoring + eviction below threshold,
(c) Consolidation cluster pass, (d) metrics snapshot.

Time query: `query_time(start, end)` returns all episodes in
a temporal window using Trace's `temporal` edges and
timestamps.

Stats: `stats()` returns a dict with counts (nodes, edges,
evicted, summarized), last sleep-cycle duration, and average
recall latency.

## 3. Public API — pre/post conditions

### 3.1 `write(episode: Episode) -> EpisodeId`

Pre: `episode.embedding` is a list[float] length matching the
Atlas dim. `episode.text` is a non-empty string. `episode.ts`
is an ISO-8601 UTC timestamp. `episode.topic` is optional.

Post: episode is in Atlas (returns via top-K with its own
embedding), is a node in Trace with id matching the returned
EpisodeId, has `conflict_level` and `conflict_reason` set on
its node metadata.

### 3.2 `recall(query: str, k: int = 10) -> list[Episode]`

Pre: `query` is a non-empty string. `k` is a positive int.

Post: returns up to `k` episodes sorted by descending cosine
similarity against the query embedding, each with metadata
(node attrs) populated. No evicted episode is returned.

### 3.3 `sleep_cycle() -> SleepReport`

Pre: at least one episode written since last sleep (or first
run).

Post: returns `SleepReport(evicted=n, summarized=m,
took_seconds=t)`. Evicted episodes are removed from Atlas and
Trace. Summarized clusters produce a new summary node with
`kind="summary"` and typed `summary_of` edges back to the
original nodes (originals retained, not deleted).

### 3.4 `query_time(start: datetime, end: datetime) -> list[Episode]`

Pre: `start <= end`.

Post: returns all episodes with `start <= ep.ts <= end`,
sorted ascending by `ts`. Includes summary nodes whose
source window intersects the query.

### 3.5 `stats() -> dict`

Pre: none.

Post: returns a dict with keys `{n_episodes, n_summaries,
n_evicted_total, last_sleep_seconds, mean_recall_ms,
atlas_dim, trace_edges}`. Never raises.

## 4. Migration mapping from v0.2

| v0.2 component         | v0.3 AeonSleep location            | Status                |
|------------------------|------------------------------------|-----------------------|
| `aeon.atlas`           | `src/memory/atlas.py`              | ported verbatim       |
| `aeon.trace`           | `src/memory/trace.py`              | ported verbatim       |
| `aeon.sleep_gate`      | `src/cognitive/sleep_tagger.py`    | rewritten, tagger-only|
| n/a                    | `src/cognitive/forgetting_gate.py` | new (small MLP)       |
| n/a                    | `src/cognitive/consolidation.py`   | new (summary clusters)|
| `aeon.Aeon` facade     | `src/memory/aeonsleep.py`          | new API surface       |

Code transfer rules:
- **Transfers**: Atlas and Trace. Both are pure classical, zero
  dependency on v0.2 stack/router artifacts, so the cousin-fork
  rule (`BRANCH-neuroscience.md`) allows the code copy.
- **Replaced**: the old `SleepGate` was bundled with a teacher
  LLM call in the hot path; v0.3 splits it into a cheap in-
  process tagger plus an offline consolidation pass that can
  call a teacher LLM only during `sleep_cycle()`, not during
  `write()`.
- **New**: ForgettingGate and Consolidation did not exist in
  v0.2.

## 5. Data model

```python
@dataclass
class Episode:
    id: str              # uuid4 hex, assigned on write
    text: str
    embedding: list[float]
    ts: datetime         # UTC
    topic: str | None    # optional tag
    kind: str = "raw"    # raw | summary
    conflict_level: float = 0.0
    conflict_reason: str | None = None
    access_count: int = 0
```

Trace node kinds: `raw`, `summary`. Edge types: `temporal`,
`causal`, `topical`, `summary_of`.

## 6. Success criteria

- `tests/test_aeonsleep.py::test_pi_depth_10_accuracy` passes:
  write 500 synthetic memories, run 3 sleep cycles, recall at
  depth 10 ≥ 95 % accuracy on planted-probe queries.
- Individual component tests pass: Atlas roundtrip, Trace
  invariants, SleepTagger precision ≥ 0.8 / recall ≥ 0.7,
  ForgettingGate F1 ≥ 0.85, Consolidation cluster reduction
  ≥ 5× with recall preservation ≥ 0.9.

## 7. References

- Aeon: arxiv 2601.15311.
- SleepGate / conflict-aware temporal tagging: arxiv
  2603.14517 ("Memory under sleep: conflict-tagged
  consolidation in language agents").
- v0.2 cognitive-layer design: `docs/specs/2026-04-15-
  cognitive-layer-design.md`.
- Branch rules: `BRANCH-neuroscience.md`.

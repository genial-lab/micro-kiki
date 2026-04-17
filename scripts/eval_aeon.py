#!/usr/bin/env python3
"""Evaluate Aeon memory recall accuracy.

Writes synthetic test conversations to Aeon, then queries and measures recall@K.

Usage:
    uv run python scripts/eval_aeon.py --episodes 100 --queries 20
    uv run python scripts/eval_aeon.py --help
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.memory.aeon import AeonPalace


def _hash_embed(text: str) -> np.ndarray:
    h = hashlib.sha256(text.encode()).digest()
    rng = np.random.RandomState(int.from_bytes(h[:4], "big"))
    vec = rng.randn(384).astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-8)

logger = logging.getLogger(__name__)

DOMAINS = [
    "chat-fr", "python", "kicad-dsl", "electronics", "firmware",
    "reasoning", "safety", "math", "devops", "robotics",
]

SEED = 42


@dataclass(frozen=True)
class EvalConfig:
    episodes: int = 100
    queries: int = 20
    output: str = "results/aeon-eval.json"
    seed: int = SEED
    dim: int = 3072


@dataclass
class RecallMetrics:
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    total_queries: int = 0
    avg_latency_ms: float = 0.0
    hits: dict[str, int] = field(default_factory=dict)


def generate_test_episodes(count: int, rng: random.Random) -> list[dict]:
    """Generate synthetic conversation episodes for eval."""
    episodes = []
    for i in range(count):
        domain = rng.choice(DOMAINS)
        content = f"Episode {i}: {domain} conversation about topic-{rng.randint(0, 999)}"
        ts = datetime(2026, 1, 1) + timedelta(hours=i)
        episodes.append({
            "content": content,
            "domain": domain,
            "timestamp": ts,
            "source": f"eval-{i}",
            "metadata": {"eval_idx": i},
        })
    return episodes


def generate_test_queries(
    episodes: list[dict], count: int, rng: random.Random
) -> list[dict]:
    """Generate queries that should match specific episodes."""
    queries = []
    selected = rng.sample(episodes, min(count, len(episodes)))
    for ep in selected:
        queries.append({
            "query": ep["content"],
            "expected_id": None,  # filled after write
            "domain": ep["domain"],
        })
    return queries


def run_eval(config: EvalConfig) -> RecallMetrics:
    """Write episodes to Aeon, query, and measure recall."""
    rng = random.Random(config.seed)
    palace = AeonPalace(dim=384, embed_fn=_hash_embed)

    episodes = generate_test_episodes(config.episodes, rng)
    logger.info("Writing %d episodes to Aeon", len(episodes))

    episode_ids: list[str] = []
    for ep in episodes:
        eid = palace.write(
            content=ep["content"],
            domain=ep["domain"],
            timestamp=ep["timestamp"],
            source=ep["source"],
            metadata=ep["metadata"],
        )
        episode_ids.append(eid)

    logger.info("Aeon stats after write: %s", palace.stats)

    queries = generate_test_queries(episodes, config.queries, rng)
    for i, q in enumerate(queries):
        idx = next(
            j for j, ep in enumerate(episodes) if ep["content"] == q["query"]
        )
        q["expected_id"] = episode_ids[idx]

    metrics = RecallMetrics(total_queries=len(queries))
    hits_at = {1: 0, 5: 0, 10: 0}
    latencies: list[float] = []

    for q in queries:
        t0 = time.perf_counter()
        results = palace.recall(q["query"], top_k=10)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        result_ids = [r.id for r in results]
        expected = q["expected_id"]

        for k in (1, 5, 10):
            if expected in result_ids[:k]:
                hits_at[k] += 1

    total = len(queries)
    metrics.recall_at_1 = hits_at[1] / total if total else 0.0
    metrics.recall_at_5 = hits_at[5] / total if total else 0.0
    metrics.recall_at_10 = hits_at[10] / total if total else 0.0
    metrics.avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0
    metrics.hits = {f"at_{k}": v for k, v in hits_at.items()}

    logger.info(
        "Recall@1=%.3f  Recall@5=%.3f  Recall@10=%.3f  avg_latency=%.1fms",
        metrics.recall_at_1, metrics.recall_at_5,
        metrics.recall_at_10, metrics.avg_latency_ms,
    )
    return metrics


def save_results(metrics: RecallMetrics, output: str) -> Path:
    """Save eval results as JSON."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "timestamp": datetime.now().isoformat(),
        "recall@1": metrics.recall_at_1,
        "recall@5": metrics.recall_at_5,
        "recall@10": metrics.recall_at_10,
        "total_queries": metrics.total_queries,
        "avg_latency_ms": round(metrics.avg_latency_ms, 2),
        "hits": metrics.hits,
    }
    path.write_text(json.dumps(data, indent=2))
    logger.info("Results saved to %s", path)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Aeon memory recall accuracy",
    )
    parser.add_argument(
        "--episodes", type=int, default=100,
        help="Number of test episodes to write (default: 100)",
    )
    parser.add_argument(
        "--queries", type=int, default=20,
        help="Number of recall queries to run (default: 20)",
    )
    parser.add_argument(
        "--output", default="results/aeon-eval.json",
        help="Output JSON path (default: results/aeon-eval.json)",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = parse_args()
    config = EvalConfig(
        episodes=args.episodes,
        queries=args.queries,
        output=args.output,
        seed=args.seed,
    )
    metrics = run_eval(config)
    save_results(metrics, config.output)

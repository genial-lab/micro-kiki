#!/usr/bin/env python3
"""PoC Scenario B eval — AeonPredictor vs pure retrieval.

Generates a synthetic stream of (h_t, h_{t+1}) pairs (random walk on the
unit sphere), trains the predictor, then for N held-out queries compares:

    baseline:   palace.recall(h_q,                 k=5)
    predictive: palace.recall(predictor.predict_next(h_q), k=5)

Reports Recall@5, MRR, and the per-query win rate of the predictive path.

Usage:
    uv run python scripts/eval_aeon_predictor.py --dim 384 --n-turns 1000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.memory.aeon_predictor import AeonPredictor, PredictorConfig
from src.memory.aeonsleep import AeonSleep


@dataclass
class EvalResult:
    baseline_recall_at_5: float
    predictive_recall_at_5: float
    baseline_mrr: float
    predictive_mrr: float
    win_rate_predictive: float
    n_queries: int
    elapsed_seconds: float
    final_train_loss: float
    predictor_ready: bool


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-8)


def _build_stream(n_turns: int, dim: int, seed: int) -> list[np.ndarray]:
    """Random-walk on the unit sphere: h_{t+1} = normalize(h_t + 0.3 * noise)."""
    rng = np.random.default_rng(seed)
    out = [_unit(rng.standard_normal(dim).astype(np.float32))]
    for _ in range(n_turns - 1):
        step = 0.3 * rng.standard_normal(dim).astype(np.float32)
        out.append(_unit(out[-1] + step))
    return out


def _reciprocal_rank(hit_ids: list[str], gold: str) -> float:
    for rank, hid in enumerate(hit_ids, start=1):
        if hid == gold:
            return 1.0 / rank
    return 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--n-turns", type=int, default=1000)
    ap.add_argument("--n-queries", type=int, default=100)
    ap.add_argument("--cold-start", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    t_start = time.time()

    # 1. Build stream and ingest.
    palace = AeonSleep(dim=args.dim)
    cfg = PredictorConfig(
        dim=args.dim,
        hidden=min(256, args.dim),
        n_stacks=16,
        cold_start_threshold=args.cold_start,
        seed=args.seed,
    )
    pred = AeonPredictor(palace=palace, config=cfg)
    palace.attach_predictor(pred)

    stream = _build_stream(args.n_turns, args.dim, seed=args.seed)
    t0 = datetime(2026, 4, 17, 10, 0)
    for i, h in enumerate(stream):
        pred.ingest_latent(
            f"t{i}", h, ts=t0 + timedelta(seconds=i), stack_id=rng.integers(0, 16)
        )

    # 2. Train predictor.
    history = pred.fit_on_buffer(
        lr=args.lr, epochs=args.epochs, batch_size=args.batch_size
    )
    final_loss = history[-1] if history else float("nan")

    # 3. Build eval set: held-out indices (last 20% of stream).
    n_held = max(args.n_queries, 1)
    held_start = max(1, len(stream) - n_held - 1)
    queries = []
    for i in range(held_start, held_start + n_held):
        if i + 1 >= len(stream):
            break
        queries.append((stream[i], f"t{i + 1}"))  # (h_q, gold next-turn id)

    # 4. Compare baseline vs predictive.
    baseline_hits, pred_hits = [], []
    baseline_rr, pred_rr = [], []
    wins = 0
    for h_q, gold in queries:
        base = palace.recall(h_q.tolist(), k=5)
        base_ids = [h.episode_id for h in base]
        baseline_hits.append(gold in base_ids)
        baseline_rr.append(_reciprocal_rank(base_ids, gold))

        h_pred = pred.predict_next(h_q, horizon=1)
        pr = palace.recall(h_pred.tolist(), k=5)
        pr_ids = [h.episode_id for h in pr]
        pred_hits.append(gold in pr_ids)
        pred_rr.append(_reciprocal_rank(pr_ids, gold))

        if pred_rr[-1] >= baseline_rr[-1] and (
            pred_rr[-1] > baseline_rr[-1] or pred_hits[-1] > baseline_hits[-1]
        ):
            wins += 1

    result = EvalResult(
        baseline_recall_at_5=float(np.mean(baseline_hits)) if baseline_hits else 0.0,
        predictive_recall_at_5=float(np.mean(pred_hits)) if pred_hits else 0.0,
        baseline_mrr=float(np.mean(baseline_rr)) if baseline_rr else 0.0,
        predictive_mrr=float(np.mean(pred_rr)) if pred_rr else 0.0,
        win_rate_predictive=(wins / len(queries)) if queries else 0.0,
        n_queries=len(queries),
        elapsed_seconds=time.time() - t_start,
        final_train_loss=float(final_loss),
        predictor_ready=pred.ready,
    )

    payload = asdict(result)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

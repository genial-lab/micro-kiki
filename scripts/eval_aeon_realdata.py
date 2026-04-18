#!/usr/bin/env python3
"""Real-data Aeon latent predictor evaluation.

Replaces the synthetic random-walk / stack-structured streams of PoC B v2
with real embeddings from the 10-domain micro-kiki corpus (data/final/).

Pipeline:
  1. Load N samples per domain from data/final/<domain>/train.jsonl
  2. Embed each user message via sentence-transformers (MiniLM-L6-v2, 384-d)
  3. Interleave domains to create a "conversation with topic shifts" stream
  4. Ingest into AeonPredictor with stack_id = domain_idx
  5. Train predictor on first 80%, evaluate on last 20%
  6. Report Recall@5, MRR baseline vs predictive, win_rate_predictive, win_rate_stack_vs_null

Addresses the primary reviewer objection from PoC B v2: "synthetic data only".

Usage:
    uv run --python 3.13 python scripts/eval_aeon_realdata.py \\
        --data-dir data/final \\
        --out results/aeon-realdata.json
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
class RealDataEvalResult:
    baseline_recall_at_5: float
    predictive_recall_at_5: float
    null_stack_recall_at_5: float
    baseline_mrr: float
    predictive_mrr: float
    null_stack_mrr: float
    win_rate_predictive: float
    win_rate_stack_vs_null: float
    n_queries: int
    n_samples_ingested: int
    n_domains: int
    elapsed_seconds: float
    final_train_loss: float
    predictor_ready: bool
    use_centering: bool
    use_layernorm_delta: bool
    dim: int


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-8)


def _reciprocal_rank(hit_ids: list[str], gold: str) -> float:
    for rank, hid in enumerate(hit_ids, start=1):
        if hid == gold:
            return 1.0 / rank
    return 0.0


def _load_real_stream(
    data_dir: Path,
    domains: list[str],
    max_per_domain: int,
    backbone_dir: Path,
    seq_len: int,
    seed: int,
    stream_mode: str = "interleaved",
) -> tuple[list[np.ndarray], list[int]]:
    """Load N samples per domain, embed via MiniLM.

    stream_mode:
      - "interleaved": round-robin interleave domains (topic-switched stream)
      - "within-topic": concatenate per-domain streams (realistic per-topic conversation)
    """
    from sentence_transformers import SentenceTransformer

    print(f"loading backbone from {backbone_dir}...")
    model = SentenceTransformer(str(backbone_dir))

    per_domain_texts: dict[str, list[str]] = {}
    for dom in domains:
        path = data_dir / dom / "train.jsonl"
        if not path.exists():
            print(f"  missing {path}, skipping")
            per_domain_texts[dom] = []
            continue
        texts = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msgs = rec.get("messages", [])
                user = next(
                    (m["content"] for m in msgs if m.get("role") == "user"), None
                )
                if not user:
                    continue
                texts.append(user)
                if len(texts) >= max_per_domain:
                    break
        per_domain_texts[dom] = texts
        print(f"  {dom}: {len(texts)} samples")

    rng = np.random.default_rng(seed)
    stream_texts: list[str] = []
    stream_stacks: list[int] = []

    if stream_mode == "within-topic":
        # Concatenate per-domain streams in shuffled order (each domain runs contiguously)
        dom_order = list(domains)
        rng.shuffle(dom_order)
        for d in dom_order:
            pool = list(range(len(per_domain_texts[d])))
            rng.shuffle(pool)
            for idx in pool:
                stream_texts.append(per_domain_texts[d][idx])
                stream_stacks.append(domains.index(d))
        print(f"within-topic stream: {len(stream_texts)} turns, {len(dom_order)} contiguous blocks")
    else:
        # Default: round-robin interleave (topic-switched stream)
        pools = {d: list(range(len(per_domain_texts[d]))) for d in domains}
        for d in domains:
            rng.shuffle(pools[d])
        while any(pools.values()):
            dom_order = list(domains)
            rng.shuffle(dom_order)
            for d in dom_order:
                if pools[d]:
                    idx = pools[d].pop(0)
                    stream_texts.append(per_domain_texts[d][idx])
                    stream_stacks.append(domains.index(d))
        print(f"interleaved stream: {len(stream_texts)} turns across {len(domains)} domains")

    # Embed via MiniLM
    print("computing embeddings...")
    embeds = model.encode(
        stream_texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    print(f"embedded shape: {embeds.shape}")
    return list(embeds), stream_stacks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/final")
    ap.add_argument("--domains", default="dsp,electronics,emc,embedded,freecad,kicad-dsl,platformio,power,spice,stm32")
    ap.add_argument("--max-per-domain", type=int, default=100)
    ap.add_argument("--backbone", default="models/niche-embeddings")
    ap.add_argument("--n-queries", type=int, default=100)
    ap.add_argument("--cold-start", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--use-centering", action="store_true")
    ap.add_argument("--use-layernorm-delta", action="store_true")
    ap.add_argument("--stream-mode", choices=["interleaved", "within-topic"],
                    default="interleaved",
                    help="Stream topology: interleaved (topic-switched) or within-topic (contiguous blocks)")
    ap.add_argument("--eval-metric", choices=["exact", "soft-domain"],
                    default="exact",
                    help="exact: gold = exact t+1 turn id; soft-domain: gold = domain of t+1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    t_start = time.time()
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]

    stream, stream_stack_ids = _load_real_stream(
        data_dir=Path(args.data_dir),
        domains=domains,
        max_per_domain=args.max_per_domain,
        backbone_dir=Path(args.backbone),
        seq_len=32,
        seed=args.seed,
        stream_mode=args.stream_mode,
    )
    if not stream:
        print("no samples loaded", file=sys.stderr)
        return 2

    dim = stream[0].shape[0]
    print(f"stream dim: {dim}")

    # Build Aeon + predictor
    palace = AeonSleep(dim=dim)
    cfg = PredictorConfig(
        dim=dim,
        hidden=min(256, dim),
        n_stacks=max(16, len(domains) + 1),
        cold_start_threshold=args.cold_start,
        seed=args.seed,
        use_centering=args.use_centering,
        use_layernorm_delta=args.use_layernorm_delta,
    )
    pred = AeonPredictor(palace=palace, config=cfg)
    palace.attach_predictor(pred)

    # Ingest — also build turn_id → domain_idx map for soft-match metric
    t0 = datetime(2026, 4, 19, 10, 0)
    turn_to_domain: dict[str, int] = {}
    print("ingesting...")
    for i, (h, sid) in enumerate(zip(stream, stream_stack_ids)):
        tid = f"t{i}"
        turn_to_domain[tid] = sid
        pred.ingest_latent(
            tid, h.astype(np.float32), ts=t0 + timedelta(seconds=i), stack_id=sid
        )

    # Train
    print(f"training for {args.epochs} epochs...")
    history = pred.fit_on_buffer(lr=args.lr, epochs=args.epochs, batch_size=args.batch_size)
    final_loss = history[-1] if history else float("nan")
    print(f"final train loss: {final_loss:.4f}")

    # Build eval set: last N_queries held-out indices
    n_held = min(args.n_queries, len(stream) - 2)
    held_start = max(1, len(stream) - n_held - 1)
    queries = []
    for i in range(held_start, held_start + n_held):
        if i + 1 >= len(stream):
            break
        gold_id = f"t{i + 1}"
        gold_domain = stream_stack_ids[i + 1]
        queries.append((stream[i], gold_id, gold_domain, stream_stack_ids[i]))

    def _hit(ret_ids: list[str], gold_id: str, gold_dom: int) -> bool:
        if args.eval_metric == "soft-domain":
            return any(turn_to_domain.get(r, -1) == gold_dom for r in ret_ids)
        return gold_id in ret_ids

    def _rr(ret_ids: list[str], gold_id: str, gold_dom: int) -> float:
        if args.eval_metric == "soft-domain":
            for rank, r in enumerate(ret_ids, start=1):
                if turn_to_domain.get(r, -1) == gold_dom:
                    return 1.0 / rank
            return 0.0
        return _reciprocal_rank(ret_ids, gold_id)

    # 3-way comparison: baseline / predictive / null-stack
    baseline_hits, pred_hits, null_hits = [], [], []
    baseline_rr, pred_rr, null_rr = [], [], []
    wins_pred, wins_stack = 0, 0

    print(f"evaluating {len(queries)} queries (metric={args.eval_metric})...")
    for h_q, gold_id, gold_dom, cur_stack in queries:
        # Baseline: pure retrieval
        base = palace.recall(h_q.tolist(), k=5)
        base_ids = [h.episode_id for h in base]
        baseline_hits.append(_hit(base_ids, gold_id, gold_dom))
        baseline_rr.append(_rr(base_ids, gold_id, gold_dom))

        # Predictive (real stack)
        h_pred = pred.predict_next(h_q, horizon=1, stack_id=cur_stack)
        pr = palace.recall(h_pred.tolist(), k=5)
        pr_ids = [h.episode_id for h in pr]
        pred_hits.append(_hit(pr_ids, gold_id, gold_dom))
        pred_rr.append(_rr(pr_ids, gold_id, gold_dom))

        # Null-stack (stack_id=-1)
        h_null = pred.predict_next(h_q, horizon=1, stack_id=-1)
        nr = palace.recall(h_null.tolist(), k=5)
        nr_ids = [h.episode_id for h in nr]
        null_hits.append(_hit(nr_ids, gold_id, gold_dom))
        null_rr.append(_rr(nr_ids, gold_id, gold_dom))

        if pred_rr[-1] >= baseline_rr[-1] and (
            pred_rr[-1] > baseline_rr[-1] or pred_hits[-1] > baseline_hits[-1]
        ):
            wins_pred += 1
        if pred_rr[-1] > null_rr[-1]:
            wins_stack += 1

    result = RealDataEvalResult(
        baseline_recall_at_5=float(np.mean(baseline_hits)) if baseline_hits else 0.0,
        predictive_recall_at_5=float(np.mean(pred_hits)) if pred_hits else 0.0,
        null_stack_recall_at_5=float(np.mean(null_hits)) if null_hits else 0.0,
        baseline_mrr=float(np.mean(baseline_rr)) if baseline_rr else 0.0,
        predictive_mrr=float(np.mean(pred_rr)) if pred_rr else 0.0,
        null_stack_mrr=float(np.mean(null_rr)) if null_rr else 0.0,
        win_rate_predictive=wins_pred / len(queries) if queries else 0.0,
        win_rate_stack_vs_null=wins_stack / len(queries) if queries else 0.0,
        n_queries=len(queries),
        n_samples_ingested=len(stream),
        n_domains=len(domains),
        elapsed_seconds=time.time() - t_start,
        final_train_loss=float(final_loss),
        predictor_ready=pred.ready,
        use_centering=args.use_centering,
        use_layernorm_delta=args.use_layernorm_delta,
        dim=dim,
    )

    payload = asdict(result)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

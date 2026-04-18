#!/usr/bin/env python3
"""Task D: Aeon predictor as a next-domain anticipator.

Instead of exact-message retrieval (eval_aeon_realdata.py), ask: given the
current turn, can the predictor anticipate WHICH domain the next turn will
belong to? This is the relevant task for a routing-system memory module:
pre-warm the correct LoRA stack before the user's next turn arrives.

Pipeline:
  1. Load per-domain corpora, embed via MiniLM
  2. Build interleaved stream (topic-switched — hard case for this task)
  3. Ingest into AeonPredictor with stack_id
  4. Train predictor
  5. For each held-out turn t, compute:
     - h_pred = predictor.predict_next(h_t, stack_id=cur_stack)
     - baseline guess: cur_stack (assume stay on topic)
     - predictive guess: argmax cosine_sim(h_pred, domain_centroids)
  6. Measure next-domain accuracy for baseline vs predictive.

Metric: how often does the predictor correctly anticipate a topic shift.
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
class NextDomainResult:
    baseline_acc: float            # always guess stay-on-topic
    predictive_acc: float          # argmax cos(h_pred, centroids)
    top3_predictive_acc: float     # domain in top-3 predictions
    switch_detection_rate: float   # fraction of actual switches the predictor correctly flagged
    false_switch_rate: float       # fraction of non-switches the predictor flagged as switches
    n_queries: int
    n_switches_in_eval: int
    n_domains: int
    elapsed_seconds: float
    final_train_loss: float
    use_layernorm_delta: bool
    dim: int


def _cosine_sim_matrix(query: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """query: (dim,), centroids: (n_domains, dim). Returns (n_domains,) cosines."""
    q = query / (np.linalg.norm(query) + 1e-8)
    c = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
    return c @ q


def _load_stream(data_dir: Path, domains: list[str], max_per_domain: int,
                  backbone_dir: Path, seed: int) -> tuple[list[np.ndarray], list[int]]:
    """Interleaved stream (round-robin) — the hard case for next-domain prediction."""
    from sentence_transformers import SentenceTransformer

    print(f"loading backbone from {backbone_dir}...")
    model = SentenceTransformer(str(backbone_dir))

    per_domain_texts: dict[str, list[str]] = {}
    for dom in domains:
        path = data_dir / dom / "train.jsonl"
        if not path.exists():
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

    # Round-robin interleave
    rng = np.random.default_rng(seed)
    stream_texts: list[str] = []
    stream_stacks: list[int] = []
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

    print("computing embeddings...")
    embeds = model.encode(
        stream_texts, batch_size=32, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)
    return list(embeds), stream_stacks


def _compute_centroids(stream: list[np.ndarray], stacks: list[int],
                        n_domains: int, dim: int) -> np.ndarray:
    """Compute per-domain centroid from training-portion embeddings."""
    centroids = np.zeros((n_domains, dim), dtype=np.float32)
    counts = np.zeros(n_domains, dtype=np.int64)
    for h, s in zip(stream, stacks):
        centroids[s] += h
        counts[s] += 1
    # Average
    for d in range(n_domains):
        if counts[d] > 0:
            centroids[d] /= counts[d]
    return centroids


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
    ap.add_argument("--use-layernorm-delta", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    t_start = time.time()
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    n_domains = len(domains)

    stream, stream_stacks = _load_stream(
        Path(args.data_dir), domains, args.max_per_domain,
        Path(args.backbone), args.seed,
    )
    if not stream:
        return 2
    dim = stream[0].shape[0]

    # Train/held-out split: last n_queries for eval
    n_held = min(args.n_queries, len(stream) - 2)
    split_idx = len(stream) - n_held - 1
    train_stream = stream[:split_idx]
    train_stacks = stream_stacks[:split_idx]
    eval_stream = stream[split_idx:]
    eval_stacks = stream_stacks[split_idx:]

    # Centroids from training portion only
    centroids = _compute_centroids(train_stream, train_stacks, n_domains, dim)

    # Build predictor
    palace = AeonSleep(dim=dim)
    cfg = PredictorConfig(
        dim=dim, hidden=min(256, dim), n_stacks=max(16, n_domains + 1),
        cold_start_threshold=args.cold_start, seed=args.seed,
        use_layernorm_delta=args.use_layernorm_delta,
    )
    pred = AeonPredictor(palace=palace, config=cfg)
    palace.attach_predictor(pred)

    print(f"ingesting {len(train_stream)} training turns...")
    t0 = datetime(2026, 4, 19, 10, 0)
    for i, (h, s) in enumerate(zip(train_stream, train_stacks)):
        pred.ingest_latent(f"t{i}", h, ts=t0 + timedelta(seconds=i), stack_id=s)

    print(f"training {args.epochs} epochs...")
    history = pred.fit_on_buffer(lr=args.lr, epochs=args.epochs, batch_size=args.batch_size)
    final_loss = history[-1] if history else float("nan")
    print(f"final loss: {final_loss:.4f}")

    # Evaluate next-domain prediction on held-out
    baseline_correct = 0
    pred_correct = 0
    top3_correct = 0
    switch_tp = 0   # actual switch, correctly flagged
    switch_fn = 0   # actual switch, missed
    switch_fp = 0   # not a switch, but flagged
    n_switches = 0
    n_queries = 0

    for i in range(len(eval_stream) - 1):
        h_t = eval_stream[i]
        s_cur = eval_stacks[i]
        s_next = eval_stacks[i + 1]
        is_switch = s_next != s_cur

        # Baseline: always predict stay-on-topic
        baseline_guess = s_cur
        if baseline_guess == s_next:
            baseline_correct += 1

        # Predictive: compute h_pred, classify via centroids
        h_pred = pred.predict_next(h_t, horizon=1, stack_id=s_cur)
        sims = _cosine_sim_matrix(h_pred, centroids)
        pred_guess = int(np.argmax(sims))
        top3 = set(np.argsort(-sims)[:3].tolist())
        if pred_guess == s_next:
            pred_correct += 1
        if s_next in top3:
            top3_correct += 1

        # Switch detection
        if is_switch:
            n_switches += 1
            if pred_guess != s_cur:  # flagged as switch
                switch_tp += 1
            else:
                switch_fn += 1
        else:
            if pred_guess != s_cur:  # flagged as switch falsely
                switch_fp += 1
        n_queries += 1

    baseline_acc = baseline_correct / n_queries if n_queries else 0
    pred_acc = pred_correct / n_queries if n_queries else 0
    top3_acc = top3_correct / n_queries if n_queries else 0
    sw_det = switch_tp / n_switches if n_switches else 0
    fp_rate = switch_fp / max(n_queries - n_switches, 1)

    result = NextDomainResult(
        baseline_acc=baseline_acc,
        predictive_acc=pred_acc,
        top3_predictive_acc=top3_acc,
        switch_detection_rate=sw_det,
        false_switch_rate=fp_rate,
        n_queries=n_queries,
        n_switches_in_eval=n_switches,
        n_domains=n_domains,
        elapsed_seconds=time.time() - t_start,
        final_train_loss=float(final_loss),
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

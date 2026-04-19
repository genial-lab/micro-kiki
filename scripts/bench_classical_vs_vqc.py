#!/usr/bin/env python3
"""Phase C1 benchmark: classical baselines vs TorchVQCRouter on routing data.

Loads pre-computed embeddings (.npz with 'embeddings' + 'labels') or, if a
corpus+domains CLI is provided, embeds on the fly via SentenceTransformer.
Runs 5 baselines x N seeds with identical 80/20 splits, aggregates mean+-std.

Usage (cached embeddings - fast for iteration):
    uv run python scripts/bench_classical_vs_vqc.py \\
        --embeddings-npz results/.c1-cache.npz \\
        --output results/c1-classical-vs-vqc.json \\
        --seeds 0,1,2,3,4

Usage (full pipeline):
    uv run python scripts/bench_classical_vs_vqc.py \\
        --data-dir data/final \\
        --domains dsp,electronics,emc,embedded,freecad,kicad-dsl,platformio,power,spice,stm32 \\
        --max-per-domain 50 \\
        --backbone models/niche-embeddings \\
        --embeddings-npz results/.c1-cache.npz \\
        --output results/c1-classical-vs-vqc.json
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.routing.classical_baselines import run_classical_baseline

logger = logging.getLogger(__name__)

_BASELINES = ["stratified", "logreg", "logreg_pca", "mlp", "torch_vqc"]


def _embed_corpus(data_dir: Path, domains: list[str], max_per_domain: int,
                  backbone: str, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from sentence_transformers import SentenceTransformer

    from src.routing.text_jepa.dataset import load_domain_corpus

    samples = load_domain_corpus(data_dir, domains=domains, max_per_domain=max_per_domain)
    dom_to_idx = {d: i for i, d in enumerate(domains)}
    labels = np.array([dom_to_idx[s.domain] for s in samples], dtype=np.int64)

    st = SentenceTransformer(str(backbone), device="cpu")
    tok = st.tokenizer
    transformer = st[0].auto_model.to("cpu")

    embs = []
    for s in samples:
        enc = tok(s.text, return_tensors="pt", truncation=True, max_length=seq_len, padding="max_length")
        with torch.no_grad():
            out = transformer(**enc).last_hidden_state
        embs.append(out.squeeze(0).mean(dim=0).cpu().numpy())
    return np.stack(embs).astype(np.float64), labels


def _split(X, y, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    n_tr = int(0.8 * len(idx))
    return idx[:n_tr], idx[n_tr:]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embeddings-npz", type=Path, default=None,
                   help="NPZ cache with 'embeddings' + 'labels'. If exists, loaded; "
                        "if not and --data-dir provided, computed and saved.")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--domains", default="")
    p.add_argument("--max-per-domain", type=int, default=50)
    p.add_argument("--backbone", default="models/niche-embeddings")
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    # Load or compute embeddings
    if args.embeddings_npz and args.embeddings_npz.exists():
        logger.info("loading embeddings from %s", args.embeddings_npz)
        cache = np.load(args.embeddings_npz)
        X, y = cache["embeddings"], cache["labels"]
    elif args.data_dir is not None:
        domains = [d.strip() for d in args.domains.split(",") if d.strip()]
        if not domains:
            logger.error("--domains required when computing embeddings")
            return 2
        logger.info("computing embeddings for %d domains x %d samples",
                    len(domains), args.max_per_domain)
        X, y = _embed_corpus(args.data_dir, domains, args.max_per_domain,
                             args.backbone, args.seq_len)
        if args.embeddings_npz:
            args.embeddings_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.embeddings_npz, embeddings=X, labels=y)
            logger.info("cached embeddings to %s", args.embeddings_npz)
    else:
        logger.error("either --embeddings-npz (existing) or --data-dir is required")
        return 2

    logger.info("X shape=%s  y classes=%d", X.shape, int(y.max()) + 1)

    runs = []
    logger.info("running %d baselines x %d seeds = %d total",
                len(_BASELINES), len(seeds), len(_BASELINES) * len(seeds))
    for name in _BASELINES:
        for seed in seeds:
            tr, te = _split(X, y, seed)
            out = run_classical_baseline(
                name, X[tr], y[tr], X[te], y[te],
                seed=seed, epochs=args.epochs,
            )
            out["seed"] = seed
            logger.info("  %s seed=%d  acc=%.3f  f1=%.3f  t=%.2fs  p=%d",
                        name, seed, out["accuracy"], out["macro_f1"],
                        out["train_time_s"], out["n_params"])
            runs.append(out)

    # Aggregate
    aggregated = {}
    for name in _BASELINES:
        accs = [r["accuracy"] for r in runs if r["name"] == name]
        f1s = [r["macro_f1"] for r in runs if r["name"] == name]
        times = [r["train_time_s"] for r in runs if r["name"] == name]
        params = next(r["n_params"] for r in runs if r["name"] == name)
        aggregated[name] = {
            "accuracy_mean": float(statistics.mean(accs)),
            "accuracy_std": float(statistics.pstdev(accs)) if len(accs) > 1 else 0.0,
            "macro_f1_mean": float(statistics.mean(f1s)),
            "train_time_s_mean": float(statistics.mean(times)),
            "n_params": params,
        }

    out = {
        "runs": runs,
        "aggregated": aggregated,
        "config": {
            "n_samples": int(len(X)),
            "n_classes": int(y.max()) + 1,
            "input_dim": int(X.shape[1]),
            "seeds": seeds,
            "epochs": args.epochs,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s", args.output)

    # Also print a human-readable table
    print("\n=== C1 Results (mean over %d seeds) ===" % len(seeds))
    print(f"{'baseline':<12} {'acc':>8} {'f1':>8} {'time':>7} {'params':>8}")
    for name in _BASELINES:
        a = aggregated[name]
        print(f"{name:<12} {a['accuracy_mean']:>6.3f}+-{a['accuracy_std']:.3f} "
              f"{a['macro_f1_mean']:>8.3f} {a['train_time_s_mean']:>6.1f}s {a['n_params']:>8d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

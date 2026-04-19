#!/usr/bin/env python3
"""Sweep n_qubits × seeds with torch VQC on real data. Fast version of Plan 6 Task 7.

Hypothesis: the Task 14 vs Task 15.5 accuracy gap comes from n_qubits being too small
(4 qubits, seeing only 4 dims of 384-d embedding). Sweep to test.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.routing.text_jepa.dataset import load_domain_corpus
from src.routing.torch_vqc_router import TorchVQCRouter


def _embed(backbone: str, seq_len: int, samples) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(str(backbone), device="cpu")
    tok = st.tokenizer
    tr = st[0].auto_model.to("cpu")
    out = []
    for s in samples:
        enc = tok(s.text, return_tensors="pt", truncation=True, max_length=seq_len, padding="max_length")
        with torch.no_grad():
            h = tr(**enc).last_hidden_state
        out.append(h.squeeze(0).mean(dim=0).cpu().numpy())
    return np.stack(out).astype(np.float64)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--domains", required=True)
    p.add_argument("--max-per-domain", type=int, default=50)
    p.add_argument("--backbone", default="models/niche-embeddings")
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--n-qubits-list", default="4,6,8,10")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    n_qubits_list = [int(x) for x in args.n_qubits_list.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    samples = load_domain_corpus(Path(args.data_dir), domains=domains, max_per_domain=args.max_per_domain)
    print(f"loaded {len(samples)} samples across {len(domains)} domains")

    dom_to_idx = {d: i for i, d in enumerate(domains)}
    labels = np.array([dom_to_idx[s.domain] for s in samples], dtype=np.int64)

    print("computing embeddings …")
    t0 = time.perf_counter()
    embs = _embed(args.backbone, args.seq_len, samples)
    print(f"  done in {time.perf_counter() - t0:.1f}s, shape={embs.shape}")

    rng = np.random.default_rng(0)
    idx = np.arange(len(embs))
    rng.shuffle(idx)
    split = int(0.8 * len(idx))
    tr_idx, te_idx = idx[:split], idx[split:]

    X_tr = torch.from_numpy(embs[tr_idx]).double()
    y_tr = torch.from_numpy(labels[tr_idx])
    X_te = torch.from_numpy(embs[te_idx]).double()
    y_te = labels[te_idx]

    results = []
    print(f"\nSweep: n_qubits={n_qubits_list} × seeds={seeds} ({len(n_qubits_list)*len(seeds)} runs)")
    print(f"{'n_qubits':>8}  {'seed':>4}  {'time_s':>7}  {'train_acc':>10}  {'test_acc':>9}")
    for nq in n_qubits_list:
        for sd in seeds:
            model = TorchVQCRouter(n_qubits=nq, n_layers=6, n_classes=len(domains), lr=args.lr, seed=sd)
            t0 = time.perf_counter()
            losses = model.train_batched(X_tr, y_tr, epochs=args.epochs)
            dt = time.perf_counter() - t0
            with torch.no_grad():
                tr_acc = float((model.predict(X_tr).numpy() == labels[tr_idx]).mean())
                te_acc = float((model.predict(X_te).numpy() == y_te).mean())
            print(f"{nq:>8}  {sd:>4}  {dt:>7.2f}  {tr_acc:>10.3f}  {te_acc:>9.3f}")
            results.append({
                "n_qubits": nq,
                "seed": sd,
                "time_s": dt,
                "train_acc": tr_acc,
                "test_acc": te_acc,
                "final_loss": float(losses[-1]),
            })

    print("\nAggregated by n_qubits (mean ± std over seeds):")
    for nq in n_qubits_list:
        subset = [r for r in results if r["n_qubits"] == nq]
        accs = [r["test_acc"] for r in subset]
        print(f"  n_qubits={nq}: test_acc = {np.mean(accs):.3f} ± {np.std(accs):.3f}")

    out = {
        "config": {
            "domains": domains,
            "n_classes": len(domains),
            "max_per_domain": args.max_per_domain,
            "epochs": args.epochs,
            "lr": args.lr,
            "n_train": len(tr_idx),
            "n_test": len(te_idx),
        },
        "runs": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

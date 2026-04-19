#!/usr/bin/env python3
"""3-axis ablation sweep on rescued VQC: data × regularization × projection architecture.

Builds on projection rescue (nq=4+proj=0.30). Tests whether:
1. More data (50 → 500/domain) closes train/test gap → reaches higher test acc
2. L2 weight decay regularizes → reduces overfit
3. Non-linear MLP projection extracts richer features than linear
"""
from __future__ import annotations

import argparse
import itertools
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
    p.add_argument("--backbone", default="models/niche-embeddings")
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seeds", default="0,1,2")
    args = p.parse_args()

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    seeds = [int(x) for x in args.seeds.split(",")]
    n_classes = len(domains)
    dom_to_idx = {d: i for i, d in enumerate(domains)}

    # Load BOTH data sizes upfront so embedding is only done once per size
    embed_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for max_pd in [50, 500]:
        samples = load_domain_corpus(Path(args.data_dir), domains=domains, max_per_domain=max_pd)
        labels = np.array([dom_to_idx[s.domain] for s in samples], dtype=np.int64)
        print(f"[{max_pd}/dom] loaded {len(samples)} samples, embedding …")
        t0 = time.perf_counter()
        embs = _embed(args.backbone, 32, samples)
        print(f"[{max_pd}/dom]   done in {time.perf_counter() - t0:.1f}s")
        embed_cache[max_pd] = (embs, labels)

    # Grid: data × weight_decay × architecture × seed
    configs = list(itertools.product(
        [50, 500],                    # max_per_domain
        [0.0, 1e-4, 1e-3],           # weight_decay
        [None, 64],                   # hidden_dim (None=linear, 64=MLP)
        seeds,                        # seeds
    ))
    print(f"\n{len(configs)} runs (nq=4, nl=6, proj=on, epochs={args.epochs})")
    print(f"{'data':>5} {'wd':>7} {'arch':>6} {'seed':>4} {'time':>5} {'train':>6} {'test':>6}")

    results = []
    for (max_pd, wd, hdim, sd) in configs:
        embs, labels = embed_cache[max_pd]
        rng = np.random.default_rng(0)
        idx = np.arange(len(embs))
        rng.shuffle(idx)
        split = int(0.8 * len(idx))
        tr, te = idx[:split], idx[split:]

        model = TorchVQCRouter(
            n_qubits=4, n_layers=6, n_classes=n_classes,
            lr=args.lr, seed=sd,
            input_dim=embs.shape[1],
            hidden_dim=hdim,
            weight_decay=wd,
        )
        X_tr = torch.from_numpy(embs[tr]).double()
        y_tr = torch.from_numpy(labels[tr])
        X_te = torch.from_numpy(embs[te]).double()
        y_te = labels[te]

        t0 = time.perf_counter()
        model.train_batched(X_tr, y_tr, epochs=args.epochs)
        dt = time.perf_counter() - t0
        with torch.no_grad():
            tr_acc = float((model.predict(X_tr).numpy() == labels[tr]).mean())
            te_acc = float((model.predict(X_te).numpy() == y_te).mean())
        arch = "mlp" if hdim else "lin"
        print(f"{max_pd:>5} {wd:>7.0e} {arch:>6} {sd:>4}  {dt:>4.1f}s {tr_acc:>6.3f} {te_acc:>6.3f}")
        results.append({
            "max_per_domain": max_pd,
            "weight_decay": wd,
            "hidden_dim": hdim,
            "seed": sd,
            "time_s": dt,
            "train_acc": tr_acc,
            "test_acc": te_acc,
        })

    # Aggregate mean ± std per (data, wd, arch) triplet
    print("\nAggregated (mean test_acc over seeds):")
    print(f"{'data':>5} {'wd':>7} {'arch':>6} {'mean':>6} {'std':>6}")
    for (max_pd, wd, hdim) in itertools.product([50, 500], [0.0, 1e-4, 1e-3], [None, 64]):
        subset = [r for r in results if r["max_per_domain"] == max_pd
                  and r["weight_decay"] == wd and r["hidden_dim"] == hdim]
        accs = [r["test_acc"] for r in subset]
        arch = "mlp" if hdim else "lin"
        print(f"{max_pd:>5} {wd:>7.0e} {arch:>6} {np.mean(accs):>6.3f} {np.std(accs):>6.3f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"runs": results, "n_classes": n_classes}, indent=2))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

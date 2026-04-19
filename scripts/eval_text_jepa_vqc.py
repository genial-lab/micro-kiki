#!/usr/bin/env python3
"""Benchmark VQC router: baseline MiniLM vs Text-JEPA student embeddings.

Writes a JSON blob with accuracy + param count for each condition.

Usage:
    uv run python scripts/eval_text_jepa_vqc.py \
        --data-dir data/final \
        --domains dsp,electronics,emc,embedded,freecad,kicad-dsl,platformio,power,spice,stm32 \
        --checkpoint models/text-jepa/student.pt \
        --output results/text-jepa-vqc.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.routing.quantum_router import QuantumRouter, QuantumRouterConfig
from src.routing.text_jepa.dataset import load_domain_corpus
from src.routing.text_jepa.embed import TextJEPAEmbedder
from src.routing.text_jepa.encoder import StudentEncoder
from src.routing.torch_vqc_router import TorchVQCRouter

logger = logging.getLogger(__name__)


def _make_token_fn(backbone: str, seq_len: int, input_dim: int):
    if backbone == "random":
        def _f(text: str) -> torch.Tensor:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            return torch.from_numpy(rng.standard_normal((seq_len, input_dim)).astype(np.float32))
        return _f

    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(str(backbone))
    tok = st.tokenizer
    transformer = st[0].auto_model

    def _embed(text: str) -> torch.Tensor:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=seq_len, padding="max_length")
        with torch.no_grad():
            out = transformer(**enc).last_hidden_state
        return out.squeeze(0).float()
    return _embed


def _baseline_embed(token_fn, text: str) -> np.ndarray:
    """Mean-pool raw MiniLM tokens (no JEPA student)."""
    with torch.no_grad():
        toks = token_fn(text)
        pooled = toks.mean(dim=0)
    return pooled.cpu().numpy()


def _split(embs, labels, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(embs))
    rng.shuffle(idx)
    split = int(0.8 * len(idx))
    return idx[:split], idx[split:]


def _evaluate_vqc_pennylane(embs, labels, n_classes, epochs, seed):
    tr, te = _split(embs, labels, seed)
    cfg = QuantumRouterConfig(n_qubits=4, n_layers=6, n_classes=n_classes)
    vqc = QuantumRouter(cfg)
    vqc.train(embs[tr], labels[tr].astype(int), epochs=epochs)

    correct = 0
    for e, y in zip(embs[te], labels[te]):
        qubits = vqc.circuit(vqc.weights, e)
        logits = qubits @ vqc.linear_w + vqc.linear_b
        if int(np.argmax(logits)) == int(y):
            correct += 1
    acc = correct / max(len(te), 1)
    n_params = vqc.weights.size + vqc.linear_w.size + vqc.linear_b.size
    return {
        "accuracy": acc,
        "n_test": int(len(te)),
        "n_params": int(n_params),
        "latent_dim": int(embs.shape[1]),
        "backend": "pennylane",
    }


def _evaluate_vqc_torch(embs, labels, n_classes, epochs, seed,
                        use_projection=True, weight_decay=1e-4, lr=0.05):
    """Torch-native VQC with optional learned projection — 3000× PennyLane speed."""
    tr, te = _split(embs, labels, seed)
    kwargs = dict(n_qubits=4, n_layers=6, n_classes=n_classes,
                  lr=lr, seed=seed, weight_decay=weight_decay)
    if use_projection:
        kwargs["input_dim"] = int(embs.shape[1])
    model = TorchVQCRouter(**kwargs)

    X_tr = torch.from_numpy(embs[tr]).double()
    y_tr = torch.from_numpy(labels[tr].astype(np.int64))
    X_te = torch.from_numpy(embs[te]).double()
    y_te = labels[te].astype(np.int64)
    model.train_batched(X_tr, y_tr, epochs=epochs)

    with torch.no_grad():
        preds = model.predict(X_te).numpy()
    acc = float((preds == y_te).mean())
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "accuracy": acc,
        "n_test": int(len(te)),
        "n_params": int(n_params),
        "latent_dim": int(embs.shape[1]),
        "backend": "torch",
        "projection": bool(use_projection),
        "weight_decay": float(weight_decay),
    }


def _evaluate_vqc(embs, labels, n_classes, epochs, seed, backend="torch",
                  use_projection=True, weight_decay=1e-4):
    if backend == "pennylane":
        return _evaluate_vqc_pennylane(embs, labels, n_classes, epochs, seed)
    if backend == "torch":
        return _evaluate_vqc_torch(embs, labels, n_classes, epochs, seed,
                                   use_projection=use_projection,
                                   weight_decay=weight_decay)
    raise ValueError(f"unknown backend {backend!r} — use 'torch' or 'pennylane'")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--domains", required=True, help="comma-separated")
    p.add_argument("--max-per-domain", type=int, default=200)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--backbone", default="models/niche-embeddings")
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--input-dim", type=int, default=384)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", default="torch", choices=["torch", "pennylane"],
                   help="VQC backend (default: torch — ~3000× faster, autograd vs parameter-shift)")
    p.add_argument("--no-projection", action="store_true",
                   help="Disable learned projection (torch backend only) — reproduces PennyLane truncation behavior")
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="L2 regularization for torch backend (default: 1e-4, optimal per sweep)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    samples = load_domain_corpus(Path(args.data_dir), domains=domains, max_per_domain=args.max_per_domain)
    if not samples:
        logger.error("no samples loaded")
        return 2
    logger.info("loaded %d samples", len(samples))

    dom_to_idx = {d: i for i, d in enumerate(domains)}
    labels = np.array([dom_to_idx[s.domain] for s in samples], dtype=np.int64)

    token_fn = _make_token_fn(args.backbone, args.seq_len, args.input_dim)

    logger.info("computing baseline embeddings …")
    baseline_embs = np.stack([_baseline_embed(token_fn, s.text) for s in samples], axis=0)

    logger.info("loading Text-JEPA student from %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    student = StudentEncoder(
        input_dim=int(cfg["input_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        output_dim=int(cfg["latent_dim"]),
    )
    student.load_state_dict(ckpt["student_state_dict"])
    embedder = TextJEPAEmbedder(student=student, token_embed_fn=token_fn, latent_dim=int(cfg["latent_dim"]))
    logger.info("computing Text-JEPA embeddings …")
    jepa_embs = np.stack([embedder.embed(s.text) for s in samples], axis=0)

    n_classes = len(domains)

    eval_kwargs = dict(
        n_classes=n_classes,
        epochs=args.epochs,
        seed=args.seed,
        backend=args.backend,
        use_projection=not args.no_projection,
        weight_decay=args.weight_decay,
    )
    logger.info("training+evaluating baseline VQC (backend=%s) …", args.backend)
    baseline_result = _evaluate_vqc(baseline_embs, labels, **eval_kwargs)
    logger.info("training+evaluating Text-JEPA VQC (backend=%s) …", args.backend)
    jepa_result = _evaluate_vqc(jepa_embs, labels, **eval_kwargs)

    out = {
        "baseline": baseline_result,
        "text_jepa": jepa_result,
        "domains": domains,
        "n_samples": int(len(samples)),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("baseline acc=%.3f  text_jepa acc=%.3f", baseline_result["accuracy"], jepa_result["accuracy"])
    logger.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

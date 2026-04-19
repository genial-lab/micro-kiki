#!/usr/bin/env python3
"""Latent-dim ablation: train Text-JEPA students at multiple latent dims and compare VQC accuracy.

For each latent_dim in the sweep, train a student encoder + predictor, compute embeddings
for the full corpus, then train+evaluate a VQC on an 80/20 split. Also runs the 384-d baseline
(raw MiniLM mean-pool) for reference. Writes a single JSON summary with per-dim results.

Usage:
    uv run python scripts/ablate_text_jepa_dim.py \\
        --data-dir data/final \\
        --domains dsp,electronics,emc,embedded,freecad,kicad-dsl,platformio,power,spice,stm32 \\
        --max-per-domain 100 \\
        --latent-dims 64,128,256 \\
        --train-epochs 2 \\
        --vqc-epochs 2 \\
        --output results/text-jepa-ablation.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.routing.quantum_router import QuantumRouter, QuantumRouterConfig
from src.routing.torch_vqc_router import TorchVQCRouter
from src.routing.text_jepa.dataset import load_domain_corpus
from src.routing.text_jepa.embed import TextJEPAEmbedder
from src.routing.text_jepa.trainer import TextJEPATrainer

logger = logging.getLogger(__name__)


def _make_token_fn(backbone: str, seq_len: int, input_dim: int):
    # Force CPU to avoid MPS placeholder allocation issues.
    torch.set_default_device("cpu")

    if backbone == "random":
        def _f(text: str) -> torch.Tensor:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            arr = rng.standard_normal((seq_len, input_dim)).astype(np.float32)
            return torch.from_numpy(arr).to("cpu")

        return _f

    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(str(backbone), device="cpu")
    tok = st.tokenizer
    transformer = st[0].auto_model.to("cpu")

    def _embed(text: str) -> torch.Tensor:
        enc = tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=seq_len,
            padding="max_length",
        )
        enc = {k: v.to("cpu") for k, v in enc.items()}
        with torch.no_grad():
            out = transformer(**enc).last_hidden_state
        return out.squeeze(0).float().to("cpu")

    return _embed


def _baseline_embed(token_fn, text: str) -> np.ndarray:
    with torch.no_grad():
        toks = token_fn(text)
        pooled = toks.mean(dim=0)
    return pooled.cpu().numpy()


def _train_student(
    samples,
    token_fn,
    input_dim: int,
    latent_dim: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    seed: int,
) -> TextJEPATrainer:
    trainer = TextJEPATrainer(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        predictor_hidden=max(latent_dim // 4, 8),
        lr=1e-3,
        ema_momentum=0.99,
        mask_ratio=0.4,
        min_span=3,
        max_span=5,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    idx = np.arange(len(samples))
    for epoch in range(epochs):
        rng.shuffle(idx)
        losses: list[float] = []
        for i in range(0, len(idx), batch_size):
            chunk = [samples[j] for j in idx[i : i + batch_size]]
            batch = torch.stack([token_fn(s.text) for s in chunk], dim=0)
            loss = trainer.step(batch)
            losses.append(float(loss.item()))
            if trainer.collapsed:
                logger.warning(
                    "collapse at latent_dim=%d epoch=%d", latent_dim, epoch
                )
                return trainer
        logger.info(
            "  train latent_dim=%d epoch=%d/%d avg_loss=%.4f",
            latent_dim,
            epoch + 1,
            epochs,
            float(np.mean(losses)) if losses else 0.0,
        )
    return trainer


def _eval_vqc(
    embs: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    epochs: int,
    seed: int,
    backend: str = "torch",
) -> dict:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(embs))
    rng.shuffle(idx)
    split = int(0.8 * len(idx))
    tr, te = idx[:split], idx[split:]

    if backend == "torch":
        model = TorchVQCRouter(
            n_qubits=4, n_layers=6, n_classes=n_classes,
            lr=0.05, seed=seed, input_dim=int(embs.shape[1]), weight_decay=1e-4,
        )
        X_tr = torch.from_numpy(embs[tr]).double()
        y_tr = torch.from_numpy(labels[tr].astype(np.int64))
        X_te = torch.from_numpy(embs[te]).double()
        model.train_batched(X_tr, y_tr, epochs=epochs)
        with torch.no_grad():
            preds = model.predict(X_te).numpy()
        acc = float((preds == labels[te].astype(np.int64)).mean())
        n_params = sum(p.numel() for p in model.parameters())
        return {"accuracy": acc, "n_test": int(len(te)), "vqc_params": int(n_params), "backend": "torch"}

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
    n_params = int(vqc.weights.size + vqc.linear_w.size + vqc.linear_b.size)
    return {"accuracy": float(acc), "n_test": int(len(te)), "vqc_params": n_params, "backend": "pennylane"}


def _count_params(module: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def main() -> int:
    torch.set_default_device("cpu")

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--domains", required=True)
    p.add_argument("--max-per-domain", type=int, default=500)
    p.add_argument("--latent-dims", required=True, help="comma-separated ints")
    p.add_argument("--train-epochs", type=int, default=3)
    p.add_argument("--vqc-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--output", required=True)
    p.add_argument("--backbone", default="models/niche-embeddings")
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--input-dim", type=int, default=384)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument(
        "--checkpoints-dir",
        default="models/text-jepa",
        help="dir to save student_dim{N}.pt checkpoints",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    latent_dims = [int(x) for x in args.latent_dims.split(",")]

    samples = load_domain_corpus(
        Path(args.data_dir), domains=domains, max_per_domain=args.max_per_domain
    )
    if not samples:
        logger.error("no samples loaded")
        return 2
    logger.info("loaded %d samples across %d domains", len(samples), len(domains))

    dom_to_idx = {d: i for i, d in enumerate(domains)}
    labels = np.array([dom_to_idx[s.domain] for s in samples], dtype=np.int64)
    token_fn = _make_token_fn(args.backbone, args.seq_len, args.input_dim)
    n_classes = len(domains)

    # Baseline: raw mean-pooled token embeddings (input_dim-d)
    logger.info("--- baseline (raw mean-pool, dim=%d) ---", args.input_dim)
    t0 = time.time()
    baseline_embs = np.stack(
        [_baseline_embed(token_fn, s.text) for s in samples], axis=0
    )
    baseline_embed_sec = time.time() - t0
    t0 = time.time()
    baseline_result = _eval_vqc(
        baseline_embs, labels, n_classes, epochs=args.vqc_epochs, seed=args.seed
    )
    baseline_eval_sec = time.time() - t0
    baseline_result["latent_dim"] = int(args.input_dim)
    baseline_result["embed_time_sec"] = round(baseline_embed_sec, 2)
    baseline_result["eval_time_sec"] = round(baseline_eval_sec, 2)
    logger.info(
        "baseline dim=%d accuracy=%.3f", args.input_dim, baseline_result["accuracy"]
    )

    ckpt_dir = Path(args.checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict] = []
    for ld in latent_dims:
        logger.info("--- latent_dim=%d ---", ld)
        t0 = time.time()
        trainer = _train_student(
            samples=samples,
            token_fn=token_fn,
            input_dim=args.input_dim,
            latent_dim=ld,
            hidden_dim=args.hidden_dim,
            epochs=args.train_epochs,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        train_sec = time.time() - t0

        # Save checkpoint for this dim
        ckpt_path = ckpt_dir / f"student_dim{ld}.pt"
        torch.save(
            {
                "student_state_dict": trainer.student.state_dict(),
                "predictor_state_dict": trainer.predictor.state_dict(),
                "config": {
                    "input_dim": args.input_dim,
                    "latent_dim": ld,
                    "hidden_dim": args.hidden_dim,
                    "seq_len": args.seq_len,
                    "backbone": str(args.backbone),
                },
            },
            ckpt_path,
        )

        student_params = _count_params(trainer.student)

        t0 = time.time()
        embedder = TextJEPAEmbedder(
            student=trainer.student, token_embed_fn=token_fn, latent_dim=ld
        )
        embs = np.stack([embedder.embed(s.text) for s in samples], axis=0)
        embed_sec = time.time() - t0

        t0 = time.time()
        result = _eval_vqc(
            embs, labels, n_classes, epochs=args.vqc_epochs, seed=args.seed
        )
        eval_sec = time.time() - t0

        result["latent_dim"] = ld
        result["collapsed"] = bool(trainer.collapsed)
        result["student_params"] = student_params
        result["train_time_sec"] = round(train_sec, 2)
        result["embed_time_sec"] = round(embed_sec, 2)
        result["eval_time_sec"] = round(eval_sec, 2)
        result["compression_ratio"] = round(args.input_dim / ld, 3)
        result["checkpoint"] = str(ckpt_path)
        runs.append(result)
        logger.info(
            "latent_dim=%d accuracy=%.3f collapsed=%s student_params=%d",
            ld,
            result["accuracy"],
            result["collapsed"],
            student_params,
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "baseline": baseline_result,
                "runs": runs,
                "domains": domains,
                "n_samples": int(len(samples)),
                "config": {
                    "max_per_domain": args.max_per_domain,
                    "train_epochs": args.train_epochs,
                    "vqc_epochs": args.vqc_epochs,
                    "batch_size": args.batch_size,
                    "seq_len": args.seq_len,
                    "input_dim": args.input_dim,
                    "hidden_dim": args.hidden_dim,
                    "backbone": str(args.backbone),
                    "seed": args.seed,
                },
            },
            indent=2,
        )
    )
    logger.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

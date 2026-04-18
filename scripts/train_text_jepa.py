#!/usr/bin/env python3
"""Train the Text-JEPA student + predictor on the 10-domain corpus.

Usage:
    uv run python scripts/train_text_jepa.py --config configs/text_jepa.yaml
    uv run python scripts/train_text_jepa.py --help
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.routing.text_jepa.dataset import DomainSample, load_domain_corpus
from src.routing.text_jepa.trainer import TextJEPATrainer

logger = logging.getLogger(__name__)


def _random_backbone(seed: int, seq_len: int, input_dim: int):
    rng = np.random.default_rng(seed)

    def _embed(text: str) -> torch.Tensor:
        local = np.random.default_rng(abs(hash(text)) % (2**32))
        arr = local.standard_normal((seq_len, input_dim)).astype(np.float32)
        return torch.from_numpy(arr)

    return _embed


def _st_backbone(model_dir: Path, seq_len: int):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(model_dir))
    tok = model.tokenizer
    transformer = model[0].auto_model

    def _embed(text: str) -> torch.Tensor:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=seq_len, padding="max_length")
        with torch.no_grad():
            out = transformer(**enc).last_hidden_state
        return out.squeeze(0).float()

    return _embed


def _make_batches(samples: list[DomainSample], batch_size: int, embed_fn, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(samples))
    rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        chunk = [samples[j] for j in idx[i : i + batch_size]]
        tensors = [embed_fn(s.text) for s in chunk]
        yield torch.stack(tensors, dim=0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--domains", default=None, help="comma-separated")
    p.add_argument("--max-per-domain", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--backbone", default=None, help="model dir or 'random' for CI")
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--input-dim", type=int, default=None)
    p.add_argument("--latent-dim", type=int, default=None)
    p.add_argument("--hidden-dim", type=int, default=None)
    args = p.parse_args()

    cfg: dict = {}
    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text())

    def pick(name_cli: str, name_cfg: str, default):
        v = getattr(args, name_cli.replace("-", "_"))
        if v is not None:
            return v
        return cfg.get(name_cfg, default)

    data_dir = Path(pick("data-dir", "data_dir", "data/final"))
    domains = pick("domains", "domains", ["dsp"])
    if isinstance(domains, str):
        domains = [d.strip() for d in domains.split(",") if d.strip()]
    max_per_domain = int(pick("max-per-domain", "max_per_domain", 1000))
    epochs = int(pick("epochs", "epochs", 3))
    batch_size = int(pick("batch-size", "batch_size", 16))
    output = Path(pick("output", "output", "models/text-jepa/student.pt"))
    backbone = pick("backbone", "backbone", "models/niche-embeddings")
    seq_len = int(pick("seq-len", "seq_len", 32))
    input_dim = int(pick("input-dim", "input_dim", 384))
    latent_dim = int(pick("latent-dim", "latent_dim", 128))
    hidden_dim = int(pick("hidden-dim", "hidden_dim", 256))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    samples = load_domain_corpus(data_dir, domains=domains, max_per_domain=max_per_domain)
    if not samples:
        logger.error("no samples loaded — check data-dir and domains")
        return 2
    logger.info("loaded %d samples across %d domains", len(samples), len(domains))

    if backbone == "random":
        embed_fn = _random_backbone(seed=0, seq_len=seq_len, input_dim=input_dim)
    else:
        embed_fn = _st_backbone(Path(backbone), seq_len=seq_len)

    trainer = TextJEPATrainer(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        predictor_hidden=max(latent_dim // 4, 8),
        lr=float(cfg.get("lr", 1e-3)),
        ema_momentum=float(cfg.get("ema_momentum", 0.99)),
        mask_ratio=float(cfg.get("mask_ratio", 0.4)),
        min_span=int(cfg.get("min_span", 3)),
        max_span=int(cfg.get("max_span", 5)),
        collapse_floor=float(cfg.get("collapse_floor", 0.01)),
        collapse_patience=int(cfg.get("collapse_patience", 2)),
        seed=int(cfg.get("seed", 42)),
    )

    for epoch in range(epochs):
        losses: list[float] = []
        for batch in _make_batches(samples, batch_size, embed_fn, seed=epoch):
            loss = trainer.step(batch)
            losses.append(float(loss.item()))
            if trainer.collapsed:
                logger.error("COLLAPSE detected at epoch %d — aborting", epoch)
                return 3
        logger.info("epoch %d/%d avg_loss=%.4f", epoch + 1, epochs, float(np.mean(losses)))

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "student_state_dict": trainer.student.state_dict(),
            "predictor_state_dict": trainer.predictor.state_dict(),
            "config": {
                "input_dim": input_dim,
                "latent_dim": latent_dim,
                "hidden_dim": hidden_dim,
                "seq_len": seq_len,
                "backbone": str(backbone),
            },
        },
        output,
    )
    logger.info("saved student checkpoint to %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

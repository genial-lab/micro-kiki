"""Train the V4 router: sentence-embedding -> linear head over 34 domains.

Pipeline:
    1. Load `data/router-v4/{train,valid}.jsonl` + `label_map.json`.
    2. Encode prompts with sentence-transformers (default: all-MiniLM-L6-v2, 384d).
    3. Train a single linear layer with BCEWithLogits (multi-label-friendly).
    4. Save `router.safetensors` (weight + bias) + `meta.json` to `output/router-v4/`.
    5. `--mode eval` re-computes top-1 / top-3 + a sparse confusion matrix.

Runs on kxkm-ai (RTX 4090); ~34 domains x 2k samples encodes in <2 min on GPU.

Usage:
    python scripts/train_router_v4.py --mode train
    python scripts/train_router_v4.py --mode eval
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file as safe_load
from safetensors.torch import save_file as safe_save

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)

DEFAULT_DATA = REPO_ROOT / "data" / "router-v4"
DEFAULT_OUT = REPO_ROOT / "output" / "router-v4"
DEFAULT_RESULTS = REPO_ROOT / "results" / "router-v4-eval.json"


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_encoder(model_name: str, device: str):
    from sentence_transformers import SentenceTransformer

    logger.info("Loading sentence-transformer: %s", model_name)
    return SentenceTransformer(model_name, device=device)


def encode_texts(
    encoder, texts: list[str], batch_size: int = 128, device: str = "cuda"
) -> torch.Tensor:
    """Encode to a normalised torch.float32 tensor on CPU."""
    t0 = time.time()
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    logger.info(
        "Encoded %d texts in %.1fs (dim=%d)", len(texts), time.time() - t0, embeddings.shape[1]
    )
    return torch.from_numpy(embeddings.astype(np.float32))


def train(args: argparse.Namespace) -> None:
    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_map: dict[str, int] = json.loads((data_dir / "label_map.json").read_text())
    domains = sorted(label_map, key=lambda d: label_map[d])
    num_domains = len(domains)
    logger.info("Training router over %d domains", num_domains)

    train_rows = load_jsonl(data_dir / "train.jsonl")
    valid_rows = load_jsonl(data_dir / "valid.jsonl")
    logger.info("train=%d  valid=%d", len(train_rows), len(valid_rows))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = get_encoder(args.embedding_model, device=device)

    x_train = encode_texts(
        encoder, [r["prompt"] for r in train_rows], batch_size=args.batch_size, device=device
    )
    y_train = torch.tensor([label_map[r["domain"]] for r in train_rows], dtype=torch.long)

    x_valid = encode_texts(
        encoder, [r["prompt"] for r in valid_rows], batch_size=args.batch_size, device=device
    )
    y_valid = torch.tensor([label_map[r["domain"]] for r in valid_rows], dtype=torch.long)

    embed_dim = x_train.shape[1]
    hidden = args.hidden_dim
    if hidden > 0:
        head = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_domains),
        ).to(device)
    else:
        head = nn.Linear(embed_dim, num_domains).to(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_valid = x_valid.to(device)
    y_valid = y_valid.to(device)

    # Multi-label BCE even though labels are one-hot (matches the MetaRouter sigmoid API).
    one_hot = torch.zeros(len(y_train), num_domains, device=device)
    one_hot[torch.arange(len(y_train)), y_train] = 1.0

    optim = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    bs = args.batch_size

    for epoch in range(args.epochs):
        head.train()
        perm = torch.randperm(len(x_train), device=device)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(perm), bs):
            idx = perm[i : i + bs]
            logits = head(x_train[idx])
            loss = loss_fn(logits, one_hot[idx])
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += float(loss.item())
            n_batches += 1

        head.eval()
        with torch.no_grad():
            logits = head(x_valid)
            top1 = (logits.argmax(dim=-1) == y_valid).float().mean().item()
            topk = logits.topk(3, dim=-1).indices
            top3 = (topk == y_valid.unsqueeze(-1)).any(dim=-1).float().mean().item()
        logger.info(
            "epoch %2d  loss=%.4f  val_top1=%.3f  val_top3=%.3f",
            epoch + 1,
            total_loss / max(n_batches, 1),
            top1,
            top3,
        )

    # Save weights + meta
    weights = {k: v.detach().cpu() for k, v in head.state_dict().items()}
    safe_save(weights, str(out_dir / "router.safetensors"))

    meta = {
        "embedding_model": args.embedding_model,
        "embedding_dim": embed_dim,
        "hidden_dim": hidden,
        "num_domains": num_domains,
        "domains": domains,
        "label_map": label_map,
        "arch": (
            "sentence_embedding + MLP(hidden) + linear_head"
            if hidden > 0
            else "sentence_embedding + linear_head"
        )
        + " (sigmoid at inference)",
        "loss": "BCEWithLogitsLoss (multi-label friendly)",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    # Copy label_map to out for runtime convenience
    (out_dir / "label_map.json").write_text(
        json.dumps(label_map, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("Saved router to %s", out_dir)


def evaluate(args: argparse.Namespace) -> None:
    data_dir = Path(args.data)
    out_dir = Path(args.out)
    results_path = Path(args.results)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    meta = json.loads((out_dir / "meta.json").read_text())
    label_map: dict[str, int] = meta["label_map"]
    domains: list[str] = meta["domains"]
    num_domains = len(domains)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = get_encoder(meta["embedding_model"], device=device)

    valid_rows = load_jsonl(data_dir / "valid.jsonl")
    x_valid = encode_texts(
        encoder, [r["prompt"] for r in valid_rows], batch_size=args.batch_size, device=device
    ).to(device)
    y_valid = torch.tensor([label_map[r["domain"]] for r in valid_rows], dtype=torch.long).to(
        device
    )

    weights = safe_load(str(out_dir / "router.safetensors"))
    hidden = int(meta.get("hidden_dim", 0))
    if hidden > 0:
        head = nn.Sequential(
            nn.Linear(meta["embedding_dim"], hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_domains),
        ).to(device)
    else:
        head = nn.Linear(meta["embedding_dim"], num_domains).to(device)
    head.load_state_dict({k: v.to(device) for k, v in weights.items()})
    head.eval()

    with torch.no_grad():
        logits = head(x_valid)
        preds = logits.argmax(dim=-1)
        top1 = (preds == y_valid).float().mean().item()
        topk = logits.topk(3, dim=-1).indices
        top3 = (topk == y_valid.unsqueeze(-1)).any(dim=-1).float().mean().item()

    # Confusion: count mis-predictions as (true, pred, n)
    confusion: Counter[tuple[str, str]] = Counter()
    per_domain_correct: dict[str, int] = defaultdict(int)
    per_domain_total: dict[str, int] = defaultdict(int)
    y_np = y_valid.cpu().numpy()
    p_np = preds.cpu().numpy()
    for yi, pi in zip(y_np, p_np):
        true_name = domains[int(yi)]
        pred_name = domains[int(pi)]
        per_domain_total[true_name] += 1
        if yi == pi:
            per_domain_correct[true_name] += 1
        else:
            confusion[(true_name, pred_name)] += 1

    top_confusions = [
        {"true": t, "pred": p, "count": n}
        for (t, p), n in confusion.most_common(10)
    ]
    per_domain = {
        d: {
            "n": per_domain_total[d],
            "correct": per_domain_correct[d],
            "acc": (per_domain_correct[d] / per_domain_total[d]) if per_domain_total[d] else 0.0,
        }
        for d in domains
    }

    results = {
        "num_domains": num_domains,
        "embedding_model": meta["embedding_model"],
        "embedding_dim": meta["embedding_dim"],
        "valid_size": len(valid_rows),
        "top1_accuracy": top1,
        "top3_accuracy": top3,
        "top_confusions": top_confusions,
        "per_domain_accuracy": per_domain,
    }
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    logger.info("valid_size=%d  top1=%.3f  top3=%.3f", len(valid_rows), top1, top3)
    logger.info("Wrote %s", results_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Train/eval V4 router")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Any sentence-transformers checkpoint (384d default).",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
        help="Hidden MLP width; 0 = pure linear head.",
    )
    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()

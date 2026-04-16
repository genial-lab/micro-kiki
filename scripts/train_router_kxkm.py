#!/usr/bin/env python3
"""Train router v0 on 3-stack domain mix (chat-fr + reasoning + python).

Encodes prompts with the base model tokenizer embeddings, trains a small
MLP classifier (embed_dim -> 512 -> 3 sigmoid) with BCE loss.

Usage:
    cd /home/kxkm/micro-kiki
    UNSLOTH_COMPILE_DISABLE=1 /home/kxkm/KIKI-models-tuning/.venv/bin/python \
        scripts/train_router_kxkm.py 2>&1 | tail -20
"""
import json
import logging
import os
import time

os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_MODEL = "/home/kxkm/models/qwen3.5-4b/bf16/"
DOMAIN_FILES = {
    "chat-fr": "/home/kxkm/micro-kiki/data/distilled/chat-fr.jsonl",
    "reasoning": "/home/kxkm/micro-kiki/data/distilled/reasoning.jsonl",
    "python": "/home/kxkm/micro-kiki/data/distilled/python.jsonl",
}
OUTPUT_DIR = "/home/kxkm/micro-kiki/outputs/router/v0"
DOMAIN_NAMES = sorted(DOMAIN_FILES.keys())
NUM_DOMAINS = len(DOMAIN_NAMES)
EPOCHS = 5
LR = 1e-3
BATCH_SIZE = 32
MAX_PER_DOMAIN = 600  # balance domains


class RouterMLP(nn.Module):
    """Small MLP router: embedding -> domain probabilities."""

    def __init__(self, input_dim: int, hidden_dim: int = 512, num_domains: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_domains),
        )

    def forward(self, x):
        return self.net(x)


class DomainDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def load_domain_prompts():
    """Load prompts from each domain, balanced."""
    all_examples = []
    domain_to_idx = {name: i for i, name in enumerate(DOMAIN_NAMES)}

    for domain, path in DOMAIN_FILES.items():
        path = Path(path)
        if not path.exists():
            logger.warning("Missing %s, skipping", path)
            continue
        prompts = []
        for line in path.read_text().strip().split("\n"):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if rec.get("completion", "").strip():
                    prompts.append(rec["prompt"])
            except json.JSONDecodeError:
                continue

        # Deduplicate and limit
        prompts = list(dict.fromkeys(prompts))[:MAX_PER_DOMAIN]

        label = [0.0] * NUM_DOMAINS
        label[domain_to_idx[domain]] = 1.0

        for p in prompts:
            all_examples.append({"text": p, "label": label, "domain": domain})

        logger.info("Domain %s: %d prompts", domain, len(prompts))

    return all_examples


def encode_prompts(examples, tokenizer, model, device, batch_size=16):
    """Encode prompts to embeddings using mean pooling of last hidden state."""
    embeddings = []
    labels = []

    model.eval()
    for i in range(0, len(examples), batch_size):
        batch = examples[i : i + batch_size]
        texts = [ex["text"] for ex in batch]
        batch_labels = [ex["label"] for ex in batch]

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pooling over sequence length
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            hidden = outputs.last_hidden_state * attention_mask
            pooled = hidden.sum(dim=1) / attention_mask.sum(dim=1)
            embeddings.append(pooled.cpu())
            labels.extend(batch_labels)

        if (i // batch_size) % 10 == 0:
            logger.info("Encoded %d/%d prompts", min(i + batch_size, len(examples)), len(examples))

    return torch.cat(embeddings, dim=0), torch.tensor(labels)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # Load domain prompts
    examples = load_domain_prompts()
    logger.info("Total examples: %d", len(examples))

    if len(examples) < 30:
        logger.error("Too few examples (%d). Need data in all 3 domains.", len(examples))
        return

    # Load tokenizer and model for encoding
    logger.info("Loading tokenizer and model for encoding...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load only the base model (not for generation, just embeddings)
    model = AutoModel.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    # Encode all prompts
    logger.info("Encoding prompts...")
    t0 = time.time()
    embeddings, labels = encode_prompts(examples, tokenizer, model, device)
    logger.info("Encoding done in %.1fs, shape: %s", time.time() - t0, embeddings.shape)

    # Free the large model
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Split train/eval
    embed_dim = embeddings.shape[1]
    dataset = DomainDataset(embeddings, labels)
    n_eval = min(100, len(dataset) // 5)
    n_train = len(dataset) - n_eval
    train_ds, eval_ds = random_split(dataset, [n_train, n_eval],
                                      generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE)

    logger.info("Train: %d, Eval: %d, Embed dim: %d", n_train, n_eval, embed_dim)

    # Train router MLP
    router = RouterMLP(input_dim=embed_dim, num_domains=NUM_DOMAINS).to(device)
    optimizer = torch.optim.Adam(router.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    best_f1 = 0.0
    for epoch in range(EPOCHS):
        router.train()
        total_loss = 0
        for batch_emb, batch_lbl in train_loader:
            batch_emb = batch_emb.to(device).float()
            batch_lbl = batch_lbl.to(device).float()
            optimizer.zero_grad()
            logits = router(batch_emb)
            loss = criterion(logits, batch_lbl)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Eval
        router.eval()
        correct = 0
        total = 0
        per_class_tp = [0] * NUM_DOMAINS
        per_class_fp = [0] * NUM_DOMAINS
        per_class_fn = [0] * NUM_DOMAINS

        with torch.no_grad():
            for batch_emb, batch_lbl in eval_loader:
                batch_emb = batch_emb.to(device).float()
                batch_lbl = batch_lbl.to(device).float()
                logits = router(batch_emb)
                preds = (torch.sigmoid(logits) > 0.5).float()

                # Accuracy (argmax match)
                pred_cls = logits.argmax(dim=1)
                true_cls = batch_lbl.argmax(dim=1)
                correct += (pred_cls == true_cls).sum().item()
                total += len(batch_lbl)

                # Per-class F1
                for c in range(NUM_DOMAINS):
                    per_class_tp[c] += ((preds[:, c] == 1) & (batch_lbl[:, c] == 1)).sum().item()
                    per_class_fp[c] += ((preds[:, c] == 1) & (batch_lbl[:, c] == 0)).sum().item()
                    per_class_fn[c] += ((preds[:, c] == 0) & (batch_lbl[:, c] == 1)).sum().item()

        accuracy = correct / total if total else 0
        f1_scores = []
        for c in range(NUM_DOMAINS):
            prec = per_class_tp[c] / (per_class_tp[c] + per_class_fp[c]) if (per_class_tp[c] + per_class_fp[c]) > 0 else 0
            rec = per_class_tp[c] / (per_class_tp[c] + per_class_fn[c]) if (per_class_tp[c] + per_class_fn[c]) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            f1_scores.append(f1)
            logger.info("  %s: P=%.3f R=%.3f F1=%.3f", DOMAIN_NAMES[c], prec, rec, f1)

        macro_f1 = sum(f1_scores) / len(f1_scores)
        logger.info("Epoch %d/%d: loss=%.4f acc=%.4f macro_F1=%.4f",
                     epoch + 1, EPOCHS, avg_loss, accuracy, macro_f1)

        if macro_f1 > best_f1:
            best_f1 = macro_f1

    # Save router
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    router_path = Path(OUTPUT_DIR) / "router.pt"
    torch.save({
        "state_dict": router.state_dict(),
        "input_dim": embed_dim,
        "num_domains": NUM_DOMAINS,
        "domain_names": DOMAIN_NAMES,
        "base_model": BASE_MODEL,
    }, router_path)

    metrics = {
        "macro_f1": round(best_f1, 4),
        "accuracy": round(accuracy, 4),
        "n_train": n_train,
        "n_eval": n_eval,
        "epochs": EPOCHS,
        "domain_names": DOMAIN_NAMES,
        "embed_dim": embed_dim,
    }
    metrics_path = Path(OUTPUT_DIR) / "router_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    logger.info("=" * 60)
    logger.info("ROUTER RESULT: macro_F1=%.4f accuracy=%.4f", best_f1, accuracy)
    logger.info("PASS: %s (threshold: 0.85)", "YES" if best_f1 >= 0.85 else "NO")
    logger.info("Saved to %s", router_path)


if __name__ == "__main__":
    main()

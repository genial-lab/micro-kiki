"""Train router v0: 3-domain classifier using sentence embeddings from base model."""
import os
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

import json
import logging
import random
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_MODEL = "/home/kxkm/models/qwen3.5-4b/bf16/"
DOMAIN_FILES = {
    "chat-fr": "/home/kxkm/micro-kiki/data/distilled/chat-fr.jsonl",
    "reasoning": "/home/kxkm/micro-kiki/data/distilled/reasoning.jsonl",
    "python": "/home/kxkm/micro-kiki/data/distilled/python.jsonl",
}
OUTPUT_DIR = "/home/kxkm/micro-kiki/outputs/router/v0"
EPOCHS = 5
LR = 1e-3
BATCH_SIZE = 32
HIDDEN_DIM = 256
MAX_EXAMPLES_PER_DOMAIN = 300  # balance domains
SEED = 42


class DomainRouter(nn.Module):
    def __init__(self, input_dim, num_domains, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(self, x):
        return self.net(x)


def load_domain_data():
    """Load prompts with domain labels, balanced."""
    all_data = []
    domain_names = sorted(DOMAIN_FILES.keys())
    domain_to_idx = {name: i for i, name in enumerate(domain_names)}

    for domain, path in DOMAIN_FILES.items():
        p = Path(path)
        if not p.exists():
            logger.warning("Missing %s", p)
            continue
        examples = []
        for line in p.read_text().strip().split("\n"):
            if not line:
                continue
            entry = json.loads(line)
            examples.append({"prompt": entry["prompt"], "domain_idx": domain_to_idx[domain], "domain": domain})
        random.shuffle(examples)
        examples = examples[:MAX_EXAMPLES_PER_DOMAIN]
        all_data.extend(examples)
        logger.info("Domain %s: %d examples", domain, len(examples))

    random.shuffle(all_data)
    return all_data, domain_names


@torch.no_grad()
def extract_embeddings(prompts, tokenizer, model, device, batch_size=8):
    """Mean-pool last hidden state as sentence embedding."""
    embeddings = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        outputs = model(**enc, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]  # (B, seq, dim)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        embeddings.append(pooled.cpu())
    return torch.cat(embeddings, dim=0)


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    data, domain_names = load_domain_data()
    logger.info("Total: %d examples, %d domains: %s", len(data), len(domain_names), domain_names)

    # Split train/val 80/20
    split = int(len(data) * 0.8)
    train_data = data[:split]
    val_data = data[split:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load base model for embedding extraction (4-bit to save VRAM)
    logger.info("Loading base model for embeddings (4-bit)...")
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    # Extract embeddings
    logger.info("Extracting train embeddings (%d)...", len(train_data))
    train_embs = extract_embeddings([d["prompt"] for d in train_data], tokenizer, model, device)
    train_labels = torch.tensor([d["domain_idx"] for d in train_data])

    logger.info("Extracting val embeddings (%d)...", len(val_data))
    val_embs = extract_embeddings([d["prompt"] for d in val_data], tokenizer, model, device)
    val_labels = torch.tensor([d["domain_idx"] for d in val_data])

    # Free base model VRAM
    del model
    torch.cuda.empty_cache()
    logger.info("Base model freed. Training router MLP...")

    input_dim = train_embs.shape[1]
    num_domains = len(domain_names)
    router = DomainRouter(input_dim, num_domains, HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(router.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    train_embs_gpu = train_embs.to(device)
    train_labels_gpu = train_labels.to(device)
    val_embs_gpu = val_embs.to(device)
    val_labels_gpu = val_labels.to(device)

    best_f1 = 0.0
    for epoch in range(EPOCHS):
        router.train()
        perm = torch.randperm(len(train_embs_gpu))
        total_loss = 0
        n_batches = 0
        for i in range(0, len(perm), BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            logits = router(train_embs_gpu[idx])
            loss = criterion(logits, train_labels_gpu[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        # Eval
        router.eval()
        with torch.no_grad():
            val_logits = router(val_embs_gpu)
            val_preds = val_logits.argmax(dim=1).cpu().numpy()
            val_true = val_labels.numpy()
            f1 = f1_score(val_true, val_preds, average="macro")
            val_loss = criterion(val_logits, val_labels_gpu).item()

        logger.info("Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  macro_f1=%.4f",
                     epoch+1, EPOCHS, total_loss/n_batches, val_loss, f1)
        if f1 > best_f1:
            best_f1 = f1

    # Save
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    save_path = output_path / "router.pt"
    torch.save({
        "state_dict": router.state_dict(),
        "input_dim": input_dim,
        "num_domains": num_domains,
        "hidden_dim": HIDDEN_DIM,
        "domain_names": domain_names,
    }, save_path)

    metrics = {
        "best_macro_f1": best_f1,
        "input_dim": input_dim,
        "num_domains": num_domains,
        "domain_names": domain_names,
        "train_size": len(train_data),
        "val_size": len(val_data),
        "epochs": EPOCHS,
    }
    with open(output_path / "router_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Router saved to %s, best macro F1: %.4f", save_path, best_f1)


if __name__ == "__main__":
    main()

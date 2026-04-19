#!/usr/bin/env python3
"""DPO training on kxkm-ai (RTX 4090 24GB) using QLoRA + TRL.

Loads a Qwen model in 4-bit via BitsAndBytes, attaches LoRA via PEFT,
and runs DPO on preference pairs from data/dpo/<domain>/train.jsonl.

Default model: Qwen3.5-4B (fits alongside llama-server).
Optional:      Qwen3.6-35B-A3B (requires killing llama-server first).

Environment: source /home/kxkm/KIKI-models-tuning/.venv/bin/activate

Usage:
    python train_dpo_kxkm.py --domain kicad-dsl
    python train_dpo_kxkm.py --all
    python train_dpo_kxkm.py --all --model Qwen/Qwen3.6-35B-A3B
    python train_dpo_kxkm.py --domain kicad-dsl --dry-run
"""
from __future__ import annotations

# Import TRL BEFORE unsloth can monkey-patch it.
# Unsloth v2026.4.4 replaces trl.DPOTrainer with a VLM-aware version
# that crashes on text-only data (expects 'images' column).
from trl import DPOConfig
from trl.trainer.dpo_trainer import DPOTrainer as _OriginalDPOTrainer

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s -- %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("train_dpo_kxkm")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DPO_DATA = PROJECT_ROOT / "data" / "dpo"
OUTPUT_DIR = PROJECT_ROOT / "output" / "dpo-kxkm"

DPO_DOMAINS = ["kicad-dsl", "electronics", "embedded", "power", "stm32"]

DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_MAX_SEQ = 1024
DEFAULT_LORA_RANK = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LR = 5e-7
DEFAULT_EPOCHS = 2
DEFAULT_BATCH = 1
DEFAULT_GRAD_ACCUM = 4


def load_dpo_data(domain: str) -> list[dict]:
    """Load DPO preference pairs for a domain.

    Expected JSONL: {"prompt": "...", "chosen": "...", "rejected": "..."}
    """
    path = DPO_DATA / domain / "train.jsonl"
    if not path.exists():
        logger.warning("No DPO data for %s at %s", domain, path)
        return []

    pairs = []
    for i, line in enumerate(open(path)):
        try:
            d = json.loads(line.strip())
            if "prompt" in d and "chosen" in d and "rejected" in d:
                pairs.append({
                    "prompt": d["prompt"],
                    "chosen": d["chosen"],
                    "rejected": d["rejected"],
                })
        except json.JSONDecodeError:
            pass

    logger.info("Loaded %d DPO pairs for %s", len(pairs), domain)
    return pairs


def train_dpo_domain(
    model,
    tokenizer,
    domain: str,
    output_dir: Path,
    lr: float,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    max_seq: int,
    dry_run: bool = False,
) -> dict | None:
    """Run DPO training for one domain."""
    from datasets import Dataset

    pairs = load_dpo_data(domain)
    if not pairs:
        return None

    if dry_run:
        logger.info("[DRY-RUN] Would train %s: %d pairs, lr=%.1e, epochs=%d",
                     domain, len(pairs), lr, epochs)
        return {"domain": domain, "pairs": len(pairs), "dry_run": True}

    ds = Dataset.from_list(pairs)
    domain_out = output_dir / domain
    domain_out.mkdir(parents=True, exist_ok=True)

    config = DPOConfig(
        output_dir=str(domain_out),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        num_train_epochs=epochs,
        warmup_ratio=0.1,
        weight_decay=0.01,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        max_length=max_seq,
        max_prompt_length=max_seq // 2,
        report_to="none",
        seed=42,
        loss_type="sigmoid",
        remove_unused_columns=False,
    )

    # Use the original TRL DPOTrainer (not Unsloth's patched version)
    trainer = _OriginalDPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        args=config,
    )

    logger.info("Starting DPO for %s: %d pairs, lr=%.1e, epochs=%d",
                domain, len(pairs), lr, epochs)
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    model.save_pretrained(str(domain_out))
    tokenizer.save_pretrained(str(domain_out))

    meta = {
        "domain": domain,
        "pairs": len(pairs),
        "lr": lr,
        "epochs": epochs,
        "elapsed_s": round(elapsed),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(domain_out / "dpo_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("DPO %s done in %.0fs", domain, elapsed)
    return meta


def load_model_qlora(model_name: str, lora_rank: int, lora_alpha: int, max_seq: int):
    """Load model in 4-bit with QLoRA using transformers + peft."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    logger.info("Loading %s in 4-bit QLoRA (transformers + peft)", model_name)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # safer for QLoRA
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Ensure warnings_issued exists (DPOTrainer needs it)
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    return model, tokenizer


def main():
    ap = argparse.ArgumentParser(description="DPO training on kxkm-ai")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--domain", type=str, help="Single domain to train")
    ap.add_argument("--all", action="store_true", help="Train all DPO domains")
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    ap.add_argument("--lora-rank", type=int, default=DEFAULT_LORA_RANK)
    ap.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    ap.add_argument("--max-seq", type=int, default=DEFAULT_MAX_SEQ)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--local-model", type=str, default=None,
                     help="Local model path (e.g. /home/kxkm/models/qwen3.5-4b/bf16)")
    args = ap.parse_args()

    domains = DPO_DOMAINS if args.all else ([args.domain] if args.domain else [])
    if not domains:
        ap.error("Specify --domain <name> or --all")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.local_model or args.model
    logger.info("DPO on kxkm-ai")
    logger.info("  Model: %s", model_name)
    logger.info("  Domains: %s", ", ".join(domains))
    logger.info("  Output: %s", args.output_dir)
    logger.info("  LoRA r=%d alpha=%d", args.lora_rank, args.lora_alpha)

    if args.dry_run:
        for d in domains:
            train_dpo_domain(None, None, d, args.output_dir,
                             args.lr, args.epochs, args.batch_size,
                             args.grad_accum, args.max_seq, dry_run=True)
        return

    model, tokenizer = load_model_qlora(
        model_name, args.lora_rank, args.lora_alpha, args.max_seq,
    )

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    for domain in domains:
        r = train_dpo_domain(
            model, tokenizer, domain, args.output_dir,
            args.lr, args.epochs, args.batch_size,
            args.grad_accum, args.max_seq,
        )
        if r:
            results.append(r)

    logger.info("=" * 60)
    logger.info("DPO Training Summary")
    for r in results:
        logger.info("  %s: %d pairs, %ds", r["domain"], r["pairs"], r.get("elapsed_s", 0))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

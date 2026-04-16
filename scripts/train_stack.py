#!/usr/bin/env python3
"""Train a single domain LoRA stack on Qwen3.5-35B-A3B.

Usage:
    # Train stack-01 chat-fr:
    UNSLOTH_COMPILE_DISABLE=1 uv run scripts/train_stack.py --domain chat-fr

    # Train with custom config:
    UNSLOTH_COMPILE_DISABLE=1 uv run scripts/train_stack.py --config configs/stack-01-chat-fr.yaml

    # Use existing KIKI-Mac_tunner data (default):
    UNSLOTH_COMPILE_DISABLE=1 uv run scripts/train_stack.py --domain chat-fr --data-source kiki

Prerequisites:
    - Qwen3.5-35B-A3B downloaded to models/qwen3.5-35b-a3b/ (or use HF repo ID)
    - Training data in ~/KIKI-Mac_tunner/data/micro-kiki/<domain>/train.jsonl
    - ~74 GB free RAM for BF16 LoRA training
    - Set UNSLOTH_COMPILE_DISABLE=1 before running
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

# Disable MPS completely — histogram_mps NotImplementedError on MoE routing
import torch
torch.backends.mps.is_available = lambda: False
from pathlib import Path

logger = logging.getLogger(__name__)

KIKI_DATA = Path.home() / "KIKI-Mac_tunner" / "data" / "micro-kiki"
LOCAL_MODEL = "models/qwen3.5-35b-a3b"
HF_MODEL = "Qwen/Qwen3.5-35B-A3B"


def find_training_data(domain: str) -> Path:
    """Find training data for a domain."""
    # Priority: local micro-kiki repo data > KIKI-Mac_tunner
    local = Path(f"data/distilled/{domain}.jsonl")
    if local.exists() and local.stat().st_size > 0:
        return local

    kiki_train = KIKI_DATA / domain / "train.jsonl"
    if kiki_train.exists():
        return kiki_train

    kiki_deduped = KIKI_DATA / "deduped" / f"{domain}.jsonl"
    if kiki_deduped.exists():
        return kiki_deduped

    raise FileNotFoundError(f"No training data found for domain '{domain}'")


def find_base_model() -> str:
    """Find the base model path (local download or HF repo ID)."""
    local = Path(LOCAL_MODEL)
    if local.exists() and any(local.glob("*.safetensors")):
        logger.info("Using local model: %s", local)
        return str(local)
    logger.info("Local model not found, using HF: %s", HF_MODEL)
    return HF_MODEL


def train(domain: str, config_overrides: dict | None = None) -> dict:
    """Train a LoRA adapter for a domain."""
    os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")

    data_path = find_training_data(domain)
    base_model = find_base_model()
    output_dir = f"outputs/stacks/stack-{domain}"

    logger.info("Domain: %s", domain)
    logger.info("Data: %s (%d lines)", data_path,
                sum(1 for l in data_path.read_text().strip().split("\n") if l))
    logger.info("Base: %s", base_model)
    logger.info("Output: %s", output_dir)

    # Import heavy deps only when needed
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer
    from datasets import load_dataset

    config = {
        "lora_rank": 16,
        "lora_alpha": 32,
        "learning_rate": 2e-4,
        "batch_size": 2,  # Conservative for 35B
        "grad_accum": 16,
        "epochs": 3,
        "seq_len": 4096,
        **(config_overrides or {}),
    }

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    logger.info("Loading model (BF16)...")
    t0 = time.time()
    # MPS doesn't support histogram for int (MoE routing bug).
    # Force CPU on macOS — 512 GB unified RAM handles 35B BF16.
    import platform
    if platform.system() == "Darwin":
        device_map = "cpu"
        logger.info("macOS: using CPU (MPS MoE histogram bug)")
    else:
        device_map = "auto"
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )
    logger.info("Model loaded in %.0fs", time.time() - t0)

    lora_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=str(data_path), split="train")
    logger.info("Dataset: %d examples", len(dataset))

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["grad_accum"],
        num_train_epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        bf16=True,
        use_cpu=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        dataloader_num_workers=0,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting training...")
    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = {
        "domain": domain,
        "train_loss": result.training_loss,
        "training_time_min": elapsed / 60,
        "output_dir": output_dir,
        "base_model": base_model,
        "lora_rank": config["lora_rank"],
        "dataset_size": len(dataset),
    }

    metrics_path = Path(output_dir) / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Training complete in %.1f min: %s", elapsed / 60, metrics)
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Train LoRA stack")
    parser.add_argument("--domain", required=True, help="Domain name (e.g. chat-fr)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    train(args.domain, {"learning_rate": args.lr, "epochs": args.epochs, "batch_size": args.batch_size})

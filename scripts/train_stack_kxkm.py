#!/usr/bin/env python3
"""Train a LoRA stack on kxkm-ai (Qwen3.5-4B base, GPU).

Generic training script that works for any domain. Uses the same
hyperparameters as train_stack01.py (LoRA r16, 3 epochs, cosine LR 2e-4).

Usage:
    cd /home/kxkm/micro-kiki
    UNSLOTH_COMPILE_DISABLE=1 /home/kxkm/KIKI-models-tuning/.venv/bin/python \
        scripts/train_stack_kxkm.py --domain reasoning --stack-id 02
    UNSLOTH_COMPILE_DISABLE=1 /home/kxkm/KIKI-models-tuning/.venv/bin/python \
        scripts/train_stack_kxkm.py --domain python --stack-id 03
"""
import argparse
import json
import logging
import os
import time

os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_MODEL = "/home/kxkm/models/qwen3.5-4b/bf16/"


def format_example(example):
    return {
        "text": (
            f"<|im_start|>user\n{example['prompt']}<|im_end|>\n"
            f"<|im_start|>assistant\n{example['completion']}<|im_end|>"
        )
    }


def main():
    parser = argparse.ArgumentParser(description="Train LoRA stack on kxkm-ai")
    parser.add_argument("--domain", required=True, help="Domain name (reasoning, python, etc.)")
    parser.add_argument("--stack-id", required=True, help="Stack number (02, 03, etc.)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--lora-rank", type=int, default=16)
    args = parser.parse_args()

    dataset_path = f"/home/kxkm/micro-kiki/data/distilled/{args.domain}.jsonl"
    output_dir = f"/home/kxkm/micro-kiki/outputs/stacks/stack-{args.stack_id}-{args.domain}"

    logger.info("Domain: %s, Stack: %s", args.domain, args.stack_id)
    logger.info("Dataset: %s", dataset_path)
    logger.info("Output: %s", output_dir)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    logger.info("Loading model from %s", BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Apply LoRA
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load and filter dataset (skip empty completions)
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    original_len = len(dataset)
    dataset = dataset.filter(lambda x: bool(x.get("completion", "").strip()))
    logger.info("Dataset: %d examples (filtered from %d)", len(dataset), original_len)

    if len(dataset) < 10:
        logger.error("Too few examples (%d). Need at least 10.", len(dataset))
        return

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        max_grad_norm=1.0,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        dataloader_num_workers=0,
        report_to="none",
        max_length=args.seq_len,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting training...")
    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0

    logger.info("Saving adapter to %s", output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = {
        "domain": args.domain,
        "stack_id": args.stack_id,
        "train_loss": result.training_loss,
        "training_time_min": round(elapsed / 60, 1),
        "output_dir": output_dir,
        "epochs": args.epochs,
        "base_model": BASE_MODEL,
        "lora_rank": args.lora_rank,
        "dataset_size": len(dataset),
    }
    with open(Path(output_dir) / "train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Training complete in %.1f min: %s", elapsed / 60, json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

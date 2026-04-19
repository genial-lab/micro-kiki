#!/usr/bin/env python3
"""GRPO training on kxkm-ai (RTX 4090 24GB) using QLoRA + TRL.

Group Relative Policy Optimization for reasoning-heavy domains.
Uses prompts from data/merged/<domain>/train.jsonl (SFT data) and a
reward function that scores completions on format + domain keywords.

Default model: Qwen3.5-4B (fits alongside llama-server).
Optional:      Qwen3.6-35B-A3B (requires killing llama-server first).

Environment: source /home/kxkm/KIKI-models-tuning/.venv/bin/activate

Usage:
    python train_grpo_kxkm.py --domain spice
    python train_grpo_kxkm.py --all
    python train_grpo_kxkm.py --all --model Qwen/Qwen3.6-35B-A3B
    python train_grpo_kxkm.py --domain spice --dry-run
"""
from __future__ import annotations

# Import TRL BEFORE unsloth can monkey-patch it.
from trl import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer as _OriginalGRPOTrainer

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import torch

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s -- %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("train_grpo_kxkm")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MERGED_DATA = PROJECT_ROOT / "data" / "merged"
DPO_DATA = PROJECT_ROOT / "data" / "dpo"
OUTPUT_DIR = PROJECT_ROOT / "output" / "grpo-kxkm"

GRPO_DOMAINS = ["spice", "emc", "embedded", "power", "kicad-dsl"]

DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "spice": [".title", ".end", ".tran", ".ac", ".dc", ".subckt", "MOSFET",
              "resistor", "capacitor", "inductor", "voltage", "current"],
    "emc": ["emission", "conducted", "radiated", "ferrite", "shielding",
            "impedance", "decoupling", "ground plane", "CISPR", "FCC"],
    "embedded": ["register", "interrupt", "DMA", "SPI", "I2C", "UART",
                 "GPIO", "timer", "watchdog", "HAL", "RTOS"],
    "power": ["efficiency", "switching", "regulator", "buck", "boost",
              "LDO", "ripple", "capacitor", "inductor", "thermal"],
    "kicad-dsl": [".title", ".end", "KiCad", "schematic", "footprint",
                  "symbol", "net", "pad", "component", "PCB"],
}

DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_MAX_SEQ = 1024
DEFAULT_LORA_RANK = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LR = 5e-6
DEFAULT_EPOCHS = 1
DEFAULT_BATCH = 1
DEFAULT_GRAD_ACCUM = 4
DEFAULT_NUM_GENERATIONS = 4


def load_grpo_prompts(domain: str, max_prompts: int = 200) -> list[dict]:
    """Load prompts for GRPO from merged SFT data or DPO data."""
    candidates = [
        MERGED_DATA / domain / "train.jsonl",
        DPO_DATA / domain / "train.jsonl",
    ]

    prompts = []
    for path in candidates:
        if not path.exists():
            continue
        for line in open(path):
            try:
                d = json.loads(line.strip())
                if "messages" in d:
                    for msg in d["messages"]:
                        if msg.get("role") == "user":
                            prompts.append({"prompt": msg["content"]})
                            break
                elif "prompt" in d:
                    prompts.append({"prompt": d["prompt"]})
            except json.JSONDecodeError:
                continue
            if len(prompts) >= max_prompts:
                break
        if prompts:
            break

    logger.info("Loaded %d GRPO prompts for %s", len(prompts), domain)
    return prompts


def make_reward_fn(domain: str):
    """Create a domain-specific reward function for GRPO."""
    keywords = DOMAIN_KEYWORDS.get(domain, [])
    kw_lower = [k.lower() for k in keywords]

    def reward_fn(completions: list[str], **kwargs) -> list[float]:
        rewards = []
        for text in completions:
            score = 0.0
            text_lower = text.lower()

            if len(text.strip()) < 20:
                rewards.append(-1.0)
                continue

            words = text.split()
            if len(words) > 10:
                unique_ratio = len(set(words)) / len(words)
                if unique_ratio < 0.3:
                    rewards.append(-0.5)
                    continue

            if 50 <= len(text) <= 2000:
                score += 0.3
            elif len(text) > 2000:
                score += 0.1

            hits = sum(1 for kw in kw_lower if kw in text_lower)
            score += min(0.4, hits * 0.1)

            if "```" in text:
                score += 0.1
            if re.search(r'\b(def |class |fn |func |int |void |module )', text):
                score += 0.1
            if "\n" in text.strip():
                score += 0.1

            rewards.append(score)
        return rewards

    return reward_fn


def train_grpo_domain(
    model,
    tokenizer,
    domain: str,
    output_dir: Path,
    lr: float,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    max_seq: int,
    num_generations: int,
    dry_run: bool = False,
) -> dict | None:
    """Run GRPO training for one domain."""
    from datasets import Dataset

    prompts = load_grpo_prompts(domain)
    if not prompts:
        logger.warning("No prompts for %s, skipping", domain)
        return None

    if dry_run:
        logger.info("[DRY-RUN] Would GRPO-train %s: %d prompts, lr=%.1e, epochs=%d, G=%d",
                     domain, len(prompts), lr, epochs, num_generations)
        return {"domain": domain, "prompts": len(prompts), "dry_run": True}

    ds = Dataset.from_list(prompts)
    domain_out = output_dir / domain
    domain_out.mkdir(parents=True, exist_ok=True)

    reward_fn = make_reward_fn(domain)

    config = GRPOConfig(
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
        max_completion_length=max_seq // 2,
        num_generations=num_generations,
        report_to="none",
        seed=42,
    )

    # Use original TRL GRPOTrainer (not Unsloth's patched version)
    trainer = _OriginalGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        reward_funcs=reward_fn,
        args=config,
    )

    logger.info("Starting GRPO for %s: %d prompts, lr=%.1e, epochs=%d, G=%d",
                domain, len(prompts), lr, epochs, num_generations)
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    model.save_pretrained(str(domain_out))
    tokenizer.save_pretrained(str(domain_out))

    meta = {
        "domain": domain,
        "prompts": len(prompts),
        "lr": lr,
        "epochs": epochs,
        "num_generations": num_generations,
        "elapsed_s": round(elapsed),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(domain_out / "grpo_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("GRPO %s done in %.0fs", domain, elapsed)
    return meta


def load_model_qlora(model_name: str, lora_rank: int, lora_alpha: int):
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
        attn_implementation="eager",
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

    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    return model, tokenizer


def main():
    ap = argparse.ArgumentParser(description="GRPO training on kxkm-ai")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--domain", type=str, help="Single domain to train")
    ap.add_argument("--all", action="store_true", help="Train all GRPO domains")
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    ap.add_argument("--lora-rank", type=int, default=DEFAULT_LORA_RANK)
    ap.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    ap.add_argument("--max-seq", type=int, default=DEFAULT_MAX_SEQ)
    ap.add_argument("--num-generations", type=int, default=DEFAULT_NUM_GENERATIONS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--local-model", type=str, default=None)
    args = ap.parse_args()

    domains = GRPO_DOMAINS if args.all else ([args.domain] if args.domain else [])
    if not domains:
        ap.error("Specify --domain <name> or --all")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.local_model or args.model
    logger.info("GRPO on kxkm-ai")
    logger.info("  Model: %s", model_name)
    logger.info("  Domains: %s", ", ".join(domains))
    logger.info("  Output: %s", args.output_dir)
    logger.info("  LoRA r=%d alpha=%d", args.lora_rank, args.lora_alpha)
    logger.info("  Group size G=%d", args.num_generations)

    if args.dry_run:
        for d in domains:
            train_grpo_domain(None, None, d, args.output_dir,
                              args.lr, args.epochs, args.batch_size,
                              args.grad_accum, args.max_seq,
                              args.num_generations, dry_run=True)
        return

    model, tokenizer = load_model_qlora(
        model_name, args.lora_rank, args.lora_alpha,
    )

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    for domain in domains:
        r = train_grpo_domain(
            model, tokenizer, domain, args.output_dir,
            args.lr, args.epochs, args.batch_size,
            args.grad_accum, args.max_seq, args.num_generations,
        )
        if r:
            results.append(r)

    logger.info("=" * 60)
    logger.info("GRPO Training Summary")
    for r in results:
        logger.info("  %s: %d prompts, %ds",
                     r["domain"], r["prompts"], r.get("elapsed_s", 0))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

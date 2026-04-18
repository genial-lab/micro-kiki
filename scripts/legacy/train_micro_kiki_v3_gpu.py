#!/usr/bin/env python3
"""micro-kiki V3 — MoE-LoRA sequential training on CUDA (Unsloth + SFTTrainer).

Trains 35 domain stacks sequentially on Qwen3.5-4B with:
- LoRA rank 16, alpha 32
- 4-bit quantization (fits in 24GB VRAM)
- Per-domain adapters saved separately
- Optional resume from any stack index

Usage:
    python train_micro_kiki_v3_gpu.py
    python train_micro_kiki_v3_gpu.py --resume-from 5
    python train_micro_kiki_v3_gpu.py --domains python,embedded,components
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3.5-4B"
DATA_DIR = Path("/home/kxkm/micro-kiki/data/v3")
OUTPUT_DIR = Path("/home/kxkm/micro-kiki/output/v3-gpu")
MAX_SEQ_LENGTH = 2048
LOAD_4BIT = True

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.01,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
}

TRAINING_CONFIG = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "warmup_ratio": 0.1,
    "num_train_epochs": 1,
    "max_steps": 500,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "logging_steps": 10,
    "save_steps": 250,
    "eval_steps": 50,
    "bf16": True,
    "seed": 42,
}

# Curriculum order (same as Studio MLX training)
CURRICULUM = [
    # Phase 1 — Foundations
    "chat-fr", "reasoning",
    # Phase 2 — Coding core
    "python", "typescript", "cpp", "rust",
    # Phase 3 — Coding secondary
    "html-css", "shell", "sql", "yaml-json", "docker",
    "kicad-dsl", "spice", "lua-upy",
    # Phase 4 — Technical domains
    "embedded", "stm32", "iot", "freecad", "platformio",
    "power", "emc", "dsp", "spice-sim", "electronics", "kicad-pcb",
    # Phase 5 — Applications
    "web-frontend", "web-backend", "music-audio", "devops", "llm-orch",
    # Phase 6 — Complements
    "math", "security",
    # Phase 7 — New V3 domains
    "components", "llm-ops", "ml-training",
]


def load_domain_data(domain: str):
    """Load train + valid JSONL for a domain."""
    from datasets import Dataset

    train_file = DATA_DIR / domain / "train.jsonl"
    valid_file = DATA_DIR / domain / "valid.jsonl"

    if not train_file.exists():
        print(f"  SKIP {domain}: no train.jsonl")
        return None, None

    def load_jsonl(path):
        items = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    msgs = item.get("messages", [])
                    if len(msgs) >= 2:
                        items.append({"messages": msgs})
                except:
                    continue
        return items

    train_items = load_jsonl(train_file)
    valid_items = load_jsonl(valid_file) if valid_file.exists() else train_items[:100]

    if not train_items:
        print(f"  SKIP {domain}: empty train.jsonl")
        return None, None

    return Dataset.from_list(train_items), Dataset.from_list(valid_items)


def train_stack(domain: str, stack_index: int, model, tokenizer):
    """Train one domain stack and save adapter."""
    from trl import SFTTrainer, SFTConfig
    from unsloth import FastLanguageModel

    print(f"\n{'='*60}")
    print(f"[{stack_index}/{len(CURRICULUM)}] Training: {domain}")
    print(f"{'='*60}")

    train_ds, valid_ds = load_domain_data(domain)
    if train_ds is None:
        return False

    print(f"  Train: {len(train_ds)}, Valid: {len(valid_ds)}")

    stack_dir = OUTPUT_DIR / f"stack-{stack_index:02d}-{domain}"
    stack_dir.mkdir(parents=True, exist_ok=True)

    # Cap max_steps based on data size
    max_steps = min(
        TRAINING_CONFIG["max_steps"],
        len(train_ds) // (TRAINING_CONFIG["per_device_train_batch_size"] * TRAINING_CONFIG["gradient_accumulation_steps"]) + 1,
    )
    print(f"  Max steps: {max_steps}")

    # Training
    sft_config = SFTConfig(
        output_dir=str(stack_dir),
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        warmup_ratio=TRAINING_CONFIG["warmup_ratio"],
        max_steps=max_steps,
        learning_rate=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        logging_steps=TRAINING_CONFIG["logging_steps"],
        save_steps=TRAINING_CONFIG["save_steps"],
        eval_strategy="steps",
        eval_steps=TRAINING_CONFIG["eval_steps"],
        bf16=TRAINING_CONFIG["bf16"],
        seed=TRAINING_CONFIG["seed"],
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="",  # use formatting_func
        report_to="none",
    )

    def formatting_func(examples):
        """Convert messages to chat format."""
        texts = []
        for msgs in examples["messages"]:
            # Ensure each message is a dict with role+content
            clean_msgs = []
            for m in msgs:
                if isinstance(m, dict) and "role" in m and "content" in m:
                    clean_msgs.append({"role": m["role"], "content": str(m["content"])})
                elif isinstance(m, str):
                    clean_msgs.append({"role": "user", "content": m})
            if len(clean_msgs) < 2:
                clean_msgs = [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hello!"},
                ]
            try:
                text = tokenizer.apply_chat_template(
                    clean_msgs, tokenize=False, add_generation_prompt=False
                )
            except Exception:
                text = f"<|user|>\n{clean_msgs[0]['content']}\n<|assistant|>\n{clean_msgs[-1]['content']}"
            texts.append(text)
        return texts

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        formatting_func=formatting_func,
        args=sft_config,
    )

    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    # Save adapter
    model.save_pretrained(str(stack_dir / "adapter"))
    tokenizer.save_pretrained(str(stack_dir / "adapter"))

    # Save meta
    meta = {
        "domain": domain,
        "stack_index": stack_index,
        "train_size": len(train_ds),
        "valid_size": len(valid_ds),
        "max_steps": max_steps,
        "elapsed_s": round(elapsed),
        "timestamp": datetime.now().isoformat(),
    }
    with open(stack_dir / "stack_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Eval
    eval_result = trainer.evaluate()
    print(f"  Done in {elapsed:.0f}s — eval_loss: {eval_result.get('eval_loss', '?'):.4f}")

    # Log
    log_file = OUTPUT_DIR / "training.log"
    with open(log_file, "a") as f:
        f.write(f"[{datetime.now().isoformat()}] stack-{stack_index:02d}-{domain}: "
                f"loss={eval_result.get('eval_loss', '?'):.4f}, "
                f"steps={max_steps}, time={elapsed:.0f}s\n")

    return True


def main():
    ap = argparse.ArgumentParser(description="micro-kiki V3 GPU training")
    ap.add_argument("--resume-from", type=int, default=1, help="Resume from stack index")
    ap.add_argument("--domains", type=str, default="", help="Comma-separated domain filter")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Filter domains
    domains = CURRICULUM
    if args.domains:
        requested = set(args.domains.split(","))
        domains = [d for d in CURRICULUM if d in requested]
        print(f"Filtered to {len(domains)} domains: {domains}")

    print(f"micro-kiki V3 GPU Training")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Data: {DATA_DIR}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Domains: {len(domains)}")
    print(f"  Resume from: {args.resume_from}")

    # Load model once
    from unsloth import FastLanguageModel

    print("\nLoading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_4BIT,
        dtype=None,
    )

    # Add LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        target_modules=LORA_CONFIG["target_modules"],
        use_gradient_checkpointing="unsloth",
    )

    print(f"Model loaded. Trainable params: {model.print_trainable_parameters()}")

    # Train each domain
    completed = 0
    failed = []
    for i, domain in enumerate(domains, 1):
        if i < args.resume_from:
            print(f"  Skipping {domain} (before resume point)")
            continue

        try:
            ok = train_stack(domain, i, model, tokenizer)
            if ok:
                completed += 1
        except Exception as e:
            print(f"  ERROR on {domain}: {e}")
            failed.append(domain)
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Training complete: {completed}/{len(domains)} stacks")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

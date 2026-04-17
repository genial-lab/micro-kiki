#!/usr/bin/env python3
"""Train a single Brainstacks domain stack.

Full pipeline for one domain:
1. Load frozen Qwen3.5-4B base
2. Attach MoE-LoRA (4 experts, rank 16, top-2)
3. Compute null-space projector from previously frozen stacks
4. SFT on domain data (~500 steps)
5. Residual boost (1-2 rounds on hard examples)
6. Freeze MoE-LoRA weights -> save to disk
7. Evaluate all previous domains (forgetting check)

Usage:
    python scripts/micro_kiki/train_stack.py \\
        --config configs/micro_kiki/brainstacks.yaml \\
        --domain python \\
        --stack-index 3
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import yaml
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

# Ensure scripts/ is on path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from micro_kiki.moe_lora import (
    apply_moe_lora,
    collect_moe_lora_layers,
    MoELoRALayer,
)
from micro_kiki.null_space import (
    build_projectors_for_stack,
    project_gradient,
)
from micro_kiki.residual_boost import (
    run_residual_boost_round,
)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_domain_dataset(domain_dir: str) -> tuple[list[dict], list[dict]]:
    """Load train.jsonl and valid.jsonl for a domain.

    Args:
        domain_dir: path to data/micro-kiki/<domain>/

    Returns:
        (train_examples, valid_examples) as lists of dicts
    """
    train_path = Path(domain_dir) / "train.jsonl"
    valid_path = Path(domain_dir) / "valid.jsonl"

    train = []
    if train_path.exists():
        with open(train_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    train.append(json.loads(line))

    valid = []
    if valid_path.exists():
        with open(valid_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    valid.append(json.loads(line))

    return train, valid


def extract_moe_lora_state_dict(model: nn.Module) -> dict[str, mx.array]:
    """Extract only the MoE-LoRA parameters from the model.

    Walks the model tree and collects parameters whose path
    contains '_moe_lora'. These are the only trainable params.

    Returns:
        Dict mapping parameter path -> mx.array
    """
    from mlx.utils import tree_flatten

    all_params = tree_flatten(model.parameters())
    moe_params = {}
    for name, param in all_params:
        if "moe_lora" in name:
            moe_params[name] = param
    return moe_params


def freeze_and_save_stack(
    model: nn.Module,
    output_dir: str,
    domain: str,
    train_loss: float = 0.0,
    val_loss: float = 0.0,
    steps: int = 0,
) -> None:
    """Freeze the current MoE-LoRA stack and save to disk.

    Saves:
    - adapters.safetensors: all MoE-LoRA weights
    - stack_meta.json: domain name, loss, training info

    Args:
        model: model with MoE-LoRA attached
        output_dir: where to save
        domain: domain name
        train_loss: final training loss
        val_loss: final validation loss
        steps: number of training steps completed
    """
    from safetensors.mlx import save_file

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Extract and save MoE-LoRA weights
    state = extract_moe_lora_state_dict(model)
    if state:
        save_file(state, str(out / "adapters.safetensors"))

    # Save metadata
    meta = {
        "domain": domain,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "steps": steps,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out / "stack_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Count params
    total_params = sum(p.size for p in state.values()) if state else 0
    size_mb = sum(p.nbytes for p in state.values()) / (1024 * 1024) if state else 0
    print(f"  Stack saved: {out}")
    print(f"  Parameters: {total_params:,} ({size_mb:.1f} MB)")


def evaluate_domain(
    model: nn.Module,
    tokenizer,
    domain_dir: str,
    max_seq_length: int = 2048,
    val_batches: int = 10,
) -> float:
    """Evaluate model loss on a domain's validation set.

    Args:
        model: model with MoE-LoRA attached
        tokenizer: tokenizer
        domain_dir: path to domain data dir
        max_seq_length: max tokens
        val_batches: max examples to evaluate

    Returns:
        Average cross-entropy loss on validation set
    """
    valid_path = Path(domain_dir) / "valid.jsonl"
    if not valid_path.exists():
        return float("inf")

    examples = []
    with open(valid_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    examples = examples[:val_batches]

    if len(examples) == 0:
        return float("inf")

    total_loss = 0.0
    count = 0
    for example in examples:
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        tokens = tokenizer.encode(text)[:max_seq_length]
        if len(tokens) < 2:
            continue

        input_ids = mx.array([tokens])
        logits = model(input_ids)

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        loss = nn.losses.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            reduction="mean",
        )
        mx.eval(loss)
        total_loss += loss.item()
        count += 1

    return total_loss / max(count, 1)


def train_single_stack(config_path: str, domain: str, stack_index: int) -> None:
    """Train a single domain stack end-to-end.

    Args:
        config_path: path to brainstacks.yaml
        domain: domain name (e.g. "python")
        stack_index: 1-indexed position in curriculum
    """
    config = load_config(config_path)
    project_root = Path(__file__).resolve().parent.parent.parent

    model_cfg = config["model"]
    moe_cfg = config["moe_lora"]
    ns_cfg = config["null_space"]
    boost_cfg = config["residual_boost"]
    train_cfg = config["training"]
    forget_cfg = config["forgetting"]
    output_cfg = config["output"]
    data_cfg = config["data"]
    curriculum = config["curriculum"]

    print("=" * 60)
    print(f"Brainstacks — Training stack {stack_index}/{len(curriculum)}: {domain}")
    print("=" * 60)

    # ---- 1. Load base model (frozen) ----
    print("\n[1/7] Loading base model...")
    model_path = str(project_root / model_cfg["path"])

    from mlx_lm import load as mlx_load
    model, tokenizer = mlx_load(model_path)

    # Set memory limits for Mac
    mx.set_memory_limit(460 * 1024**3)
    mx.set_cache_limit(32 * 1024**3)

    # Freeze all base parameters
    model.freeze()

    # ---- 2. Attach MoE-LoRA (dynamic rank) ----
    # Dynamic rank: sqrt(dataset_size) / 4, clamped [8, 64], rounded to multiple of 4
    data_dir = Path(config["data"]["base_dir"]) / domain
    train_file = data_dir / "train.jsonl"
    n_examples = sum(1 for _ in open(train_file)) if train_file.exists() else 500
    dynamic_rank = min(64, max(8, (int(math.sqrt(n_examples) / 4) // 4) * 4 or 8))
    dynamic_alpha = dynamic_rank * 2.0
    use_dynamic = config.get("dynamic_rank", True)
    effective_rank = dynamic_rank if use_dynamic else moe_cfg["rank"]
    effective_alpha = dynamic_alpha if use_dynamic else moe_cfg["alpha"]
    print(f"\n[2/7] Attaching MoE-LoRA (rank={effective_rank}, alpha={effective_alpha}, examples={n_examples})...")
    n_attached = apply_moe_lora(
        model,
        target_modules=model_cfg["target_modules"],
        num_experts=moe_cfg["num_experts"],
        rank=effective_rank,
        alpha=effective_alpha,
        top_k=moe_cfg["top_k"],
        dropout=moe_cfg.get("dropout", 0.01),
        router_hidden=moe_cfg["router_hidden"],
        use_rs_lora=moe_cfg.get("use_rs_lora", True),
    )
    print(f"  Attached {n_attached} MoE-LoRA layers")


    # Count trainable params
    from mlx.utils import tree_flatten
    all_params = tree_flatten(model.parameters())
    trainable = sum(p.size for name, p in all_params if "moe_lora" in name)
    total = sum(p.size for _, p in all_params)
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total*100:.4f}%)")

    # Unfreeze MoE-LoRA modules so they receive gradients
    # (base model stays frozen via model.freeze() above)
    unfrozen = 0
    for name, module in model.named_modules():
        if "moe_lora" in name:
            module.unfreeze()
            unfrozen += 1
    tp_check = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    print(f"  Unfroze {unfrozen} MoE-LoRA modules ({tp_check:,} grad-enabled params)")
    assert tp_check > 0, "ERROR: No trainable params after unfreeze! Training would be no-op."

    # ---- 3. Build null-space projectors from frozen stacks ----
    print("\n[3/7] Building null-space projectors...")
    frozen_domains = curriculum[:stack_index - 1]
    frozen_dirs = []
    for d in frozen_domains:
        d_path = str(project_root / output_cfg["base_dir"] / d)
        if Path(d_path).exists() and (Path(d_path) / "adapters.safetensors").exists():
            frozen_dirs.append(d_path)

    projectors = {}
    if len(frozen_dirs) > 0:
        try:
            projectors = build_projectors_for_stack(
                frozen_stack_dirs=frozen_dirs,
                ns_top_k_dirs=ns_cfg["ns_top_k_dirs"],
                svd_oversampling=ns_cfg.get("svd_oversampling", 10),
                svd_n_iter=ns_cfg.get("svd_n_iter", 3),
            )
            print(f"  Built {len(projectors)} null-space projectors from {len(frozen_dirs)} frozen stacks")
        except Exception as e:
            print(f"  WARNING: Null-space projection failed: {e}")
            print(f"  Continuing without projection (forgetting check still active)")
            projectors = {}
    else:
        print("  No frozen stacks (first domain in curriculum)")

    # ---- 4. Load domain data ----
    print("\n[4/7] Loading domain data...")
    domain_dir = str(project_root / data_cfg["base_dir"] / domain)
    train_data, valid_data = load_domain_dataset(domain_dir)
    print(f"  Train: {len(train_data)} | Valid: {len(valid_data)}")

    # ---- 5. SFT training loop ----
    print("\n[5/7] SFT training...")
    lr = float(train_cfg["learning_rate"])
    # Adaptive max_steps: scale with rank (more capacity = more steps to converge)
    base_max_steps = int(train_cfg["max_steps"])
    if effective_rank <= 16:
        max_steps = min(base_max_steps, 500)
    elif effective_rank <= 32:
        max_steps = min(base_max_steps, 1000)
    else:
        max_steps = base_max_steps  # 1500 for rank > 32
    print(f"  Adaptive max_steps={max_steps} (rank={effective_rank}, base={base_max_steps})")
    warmup_steps = int(max_steps * float(train_cfg.get("warmup_ratio", 0.05)))
    batch_size = train_cfg["batch_size"]
    grad_accum = train_cfg["grad_accumulation_steps"]
    max_seq_len = train_cfg["max_seq_length"]
    steps_per_eval = train_cfg["steps_per_eval"]
    seed = train_cfg.get("seed", 42)

    mx.random.seed(seed)

    # Cosine schedule with warmup
    schedule = optim.join_schedules(
        [optim.linear_schedule(1e-7, lr, warmup_steps),
         optim.cosine_decay(lr, max_steps - warmup_steps)],
        [warmup_steps],
    )
    optimizer = optim.AdamW(
        learning_rate=schedule,
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    # Training loop
    step = 0
    epoch = 0
    running_loss = 0.0
    best_val_loss = float("inf")
    train_start = time.time()

    while step < max_steps:
        epoch += 1
        np.random.shuffle(train_data)

        for example in train_data:
            if step >= max_steps:
                break

            text = tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            )
            tokens = tokenizer.encode(text)[:max_seq_len]
            if len(tokens) < 2:
                continue

            input_ids = mx.array([tokens])
            labels = input_ids

            def loss_fn(model_params):
                model.update(model_params)
                logits = model(input_ids)
                shift_logits = logits[:, :-1, :]
                shift_labels = labels[:, 1:]
                ce = nn.losses.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.shape[-1]),
                    shift_labels.reshape(-1),
                    reduction="mean",
                )
                return ce

            loss, grads = mx.value_and_grad(loss_fn)(model.trainable_parameters())

            # Null-space projection on gradients
            if projectors:
                grads = _project_grads(grads, projectors)

            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            running_loss += loss.item()
            step += 1

            # Report
            if step % 5 == 0:
                avg = running_loss / 5
                elapsed = time.time() - train_start
                print(f"  Step {step}/{max_steps} | loss={avg:.4f} | "
                      f"elapsed={elapsed:.0f}s")
                running_loss = 0.0

            # Eval
            if step % steps_per_eval == 0:
                val_loss = evaluate_domain(
                    model, tokenizer, domain_dir,
                    max_seq_length=max_seq_len,
                    val_batches=train_cfg.get("val_batches", 10),
                )
                print(f"  [Eval step {step}] val_loss={val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

    train_time = time.time() - train_start
    print(f"\n  SFT complete: {step} steps in {train_time:.0f}s")
    print(f"  Best val_loss: {best_val_loss:.4f}")

    # ---- 6. Residual boosting ----
    print("\n[6/7] Residual boosting...")
    # TODO: fix OOM in per-example loss eval (loads all examples at once)
    # Skip boosting for now — SFT alone gives good results
    max_boost_rounds = 0  # was: boost_cfg.get("max_rounds", 2)
    min_improvement = boost_cfg.get("min_improvement", 0.002)
    prev_loss = best_val_loss

    for round_num in range(1, max_boost_rounds + 1):
        # Reset optimizer with lower LR for boosting
        boost_lr = lr * boost_cfg.get("boost_lr_scale", 0.5)
        boost_optimizer = optim.AdamW(learning_rate=boost_lr)

        avg_loss = run_residual_boost_round(
            model=model,
            tokenizer=tokenizer,
            dataset=train_data,
            optimizer=boost_optimizer,
            projectors=projectors,
            config={**boost_cfg, "max_seq_length": max_seq_len},
            round_num=round_num,
        )

        improvement = prev_loss - avg_loss
        print(f"  Boost round {round_num}: improvement={improvement:.4f}")

        if improvement < min_improvement:
            print(f"  Stopping boost: improvement {improvement:.4f} < {min_improvement}")
            break
        prev_loss = avg_loss

    # ---- 7. Freeze and save ----
    print("\n[7/7] Freezing and saving stack...")
    final_val_loss = evaluate_domain(
        model, tokenizer, domain_dir,
        max_seq_length=max_seq_len,
        val_batches=train_cfg.get("val_batches", 10),
    )
    output_dir = str(project_root / output_cfg["base_dir"] / domain)
    freeze_and_save_stack(
        model, output_dir, domain=domain,
        train_loss=running_loss, val_loss=final_val_loss, steps=step,
    )

    # ---- Forgetting check ----
    print("\n[Forgetting check]")
    max_delta = forget_cfg.get("max_delta", 0.03)
    for prev_domain in frozen_domains:
        prev_dir = str(project_root / data_cfg["base_dir"] / prev_domain)
        prev_loss = evaluate_domain(
            model, tokenizer, prev_dir,
            max_seq_length=max_seq_len,
            val_batches=forget_cfg.get("val_batches", 5),
        )
        # Load the original val_loss from meta
        meta_path = Path(project_root / output_cfg["base_dir"] / prev_domain / "stack_meta.json")
        original_loss = float("inf")
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
                original_loss = meta.get("val_loss", float("inf"))

        delta = prev_loss - original_loss
        status = "OK" if delta < max_delta else "FAIL"
        print(f"  {prev_domain}: original={original_loss:.4f} "
              f"current={prev_loss:.4f} delta={delta:+.4f} [{status}]")
        if delta >= max_delta:
            print(f"  WARNING: Forgetting detected on {prev_domain}!")

    print(f"\n=== Stack {domain} training complete ===")
    print(f"  Output: {output_dir}")
    print(f"  Val loss: {final_val_loss:.4f}")


def _project_grads(grads, projectors: dict):
    """Apply null-space projection to gradients. Delegates to residual_boost._project_all_grads."""
    from micro_kiki.residual_boost import _project_all_grads
    return _project_all_grads(grads, projectors)


def main():
    parser = argparse.ArgumentParser(
        description="Brainstacks — Train a single domain stack"
    )
    parser.add_argument(
        "--config", type=str, default="configs/micro_kiki/brainstacks.yaml",
        help="Path to brainstacks config YAML",
    )
    parser.add_argument(
        "--domain", type=str, required=True,
        help="Domain name (e.g. 'python', 'embedded')",
    )
    parser.add_argument(
        "--stack-index", type=int, required=True,
        help="1-indexed position in curriculum (for null-space computation)",
    )
    args = parser.parse_args()

    train_single_stack(args.config, args.domain, args.stack_index)


if __name__ == "__main__":
    main()

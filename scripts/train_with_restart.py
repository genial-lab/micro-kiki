#!/usr/bin/env python3
"""Train LoRA with automatic restart to avoid Metal resource_limit(499000).

The Metal allocation counter never resets within a process. After ~60 iters
on MoE 35B, it hits 499K. This wrapper runs CHUNK iters in a subprocess,
lets it save a checkpoint, kills it, and restarts from the checkpoint.

Usage:
    python3 scripts/train_with_restart.py --domain platformio
    python3 scripts/train_with_restart.py --domain kicad-dsl --chunk 40
    python3 scripts/train_with_restart.py --all --chunk 40
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
import yaml
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("train_restart")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_niches_mlxtune import (
    NICHE_DOMAINS, DOMAIN_ORDER, MODEL_PATH, OUTPUTS_DIR,
    find_training_data, adapter_done, count_examples, log_progress,
)

PYTHON = Path.home() / "KIKI-Mac_tunner" / ".venv" / "bin" / "python3"


def make_train_script(
    config_path: Path,
    data_dir: Path,
    adapter_path: Path,
    resume: bool,
) -> str:
    """Generate the _train.py content for one chunk."""
    resume_line = ""
    if resume and (adapter_path / "adapters.safetensors").exists():
        resume_line = f'            "--resume-adapter-file", "{adapter_path / "adapters.safetensors"}",'

    return f'''import mlx.core as mx
mx.set_memory_limit(460 * 1024**3)
mx.set_cache_limit(32 * 1024**3)
import os, sys
os.environ["PYTHONPATH"] = "/Users/clems/KIKI-Mac_tunner/lib"
sys.path.insert(0, "/Users/clems/KIKI-Mac_tunner/lib")
from mlx_lm_fork.lora import main as lora_main
sys.argv = ["lora",
            "-c", "{config_path}",
            "--data", "{data_dir}",
            "--adapter-path", "{adapter_path}",
{resume_line}]
lora_main()
'''


def train_domain_chunked(domain: str, chunk: int) -> bool:
    """Train one domain with restart every `chunk` iters."""
    rank, epochs, lr, seq_len, dropout = NICHE_DOMAINS[domain]
    output_dir = OUTPUTS_DIR / f"stack-{domain}"
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = find_training_data(domain)
    n_examples = count_examples(data_path)
    total_iters = n_examples * epochs

    logger.info("=" * 60)
    logger.info("%s: %d examples, %d total iters, chunk=%d, ~%d restarts",
                domain, n_examples, total_iters, chunk,
                (total_iters + chunk - 1) // chunk)
    log_progress(f"RESTART-START {domain} r={rank} iters={total_iters} chunk={chunk}")

    # Write config
    config = {
        "model": str(MODEL_PATH) if MODEL_PATH.exists() else "Qwen/Qwen3.5-35B-A3B",
        "fine_tune_type": "lora",
        "lora_parameters": {
            "rank": rank, "alpha": rank * 2,
            "dropout": dropout, "scale": 2.0,
        },
        "num_layers": 40,
        "learning_rate": lr,
        "batch_size": 1,
        "grad_accumulation_steps": 4,
        "iters": chunk,
        "max_seq_length": seq_len,
        "grad_checkpoint": True,
        "save_every": chunk,
        "steps_per_report": 10,
        "steps_per_eval": chunk,
        "val_batches": 0,  # disable val eval (saves Metal allocations)
        "train": True,
        "seed": 42,
    }
    config_path = output_dir / "train_config.yaml"

    done_iters = 0
    chunk_num = 0
    t0 = time.time()

    while done_iters < total_iters:
        remaining = total_iters - done_iters
        this_chunk = min(remaining, chunk)
        chunk_num += 1

        # Update config for this chunk
        config["iters"] = this_chunk
        config["save_every"] = this_chunk
        config["steps_per_eval"] = this_chunk
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        # Generate _train.py
        script_path = output_dir / "_train.py"
        resume = done_iters > 0
        script_path.write_text(
            make_train_script(config_path, data_path.parent, output_dir, resume)
        )

        logger.info("Chunk %d: iters %d..%d / %d (resume=%s)",
                     chunk_num, done_iters, done_iters + this_chunk, total_iters, resume)

        # Run subprocess
        result = subprocess.run(
            [str(PYTHON), str(script_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=False,
        )

        if result.returncode != 0:
            adapter = output_dir / "adapters.safetensors"
            if adapter.exists():
                logger.warning("Chunk %d crashed but adapter saved — continuing", chunk_num)
            else:
                logger.warning("Chunk %d crashed, no adapter — continuing anyway", chunk_num)

        done_iters += this_chunk

        # Sleep to let Metal fully release
        if done_iters < total_iters:
            logger.info("Sleeping 10s for Metal cleanup...")
            time.sleep(10)

    elapsed = time.time() - t0
    adapter = output_dir / "adapters.safetensors"

    if adapter.exists():
        logger.info("DONE %s in %.1f min (%d chunks) → %s",
                     domain, elapsed / 60, chunk_num, output_dir)
        log_progress(f"RESTART-DONE {domain} in {elapsed/60:.1f}min ({chunk_num} chunks)")
        return True
    else:
        logger.error("FAILED %s — no adapter after %d chunks", domain, chunk_num)
        log_progress(f"RESTART-FAIL {domain} after {chunk_num} chunks")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA training with Metal restart")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--domain", help="Single domain")
    group.add_argument("--all", action="store_true", help="All pending domains")
    parser.add_argument("--chunk", type=int, default=40,
                        help="Iters per subprocess (default 40, safe for Metal)")
    args = parser.parse_args()

    if args.domain:
        domains = [args.domain]
    else:
        domains = [d for d in DOMAIN_ORDER if not adapter_done(d)]

    ok = 0
    fail = 0
    for domain in domains:
        if adapter_done(domain):
            logger.info("SKIP %s (already trained)", domain)
            continue
        if train_domain_chunked(domain, args.chunk):
            ok += 1
        else:
            fail += 1

    logger.info("SUMMARY: %d trained, %d failed, %d skipped",
                ok, fail, len(DOMAIN_ORDER) - ok - fail)


if __name__ == "__main__":
    main()

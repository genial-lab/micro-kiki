#!/usr/bin/env python3
"""Sequential LoRA training of 10 niche domains via mlx-tune.

Usage:
    uv run scripts/train_niches_mlxtune.py --all
    uv run scripts/train_niches_mlxtune.py --domain kicad-dsl
    uv run scripts/train_niches_mlxtune.py --all --dry-run
    uv run scripts/train_niches_mlxtune.py --all --start spice
"""
from __future__ import annotations

# Metal buffer fixes — must be set before any model loading.
# Applied only when mlx is available (Mac Studio M3 Ultra target).
try:
    import mlx.core as mx
    mx.set_memory_limit(460 * 1024**3)   # 460 GB
    mx.set_cache_limit(32 * 1024**3)     # 32 GB — forces buffer recycling
except ModuleNotFoundError:
    pass  # mlx not installed; --help / --dry-run still work

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("train_niches_mlxtune")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
MERGED_DATA = PROJECT_ROOT / "data" / "merged"
KIKI_DATA = Path.home() / "KIKI-Mac_tunner" / "data" / "micro-kiki"
MODEL_PATH = PROJECT_ROOT / "models" / "qwen3.5-35b-a3b"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "stacks"
PROGRESS_FILE = PROJECT_ROOT / ".ralph" / "progress.txt"

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]

# domain: (rank, epochs, lr, seq_len, dropout)
NICHE_DOMAINS: dict[str, tuple[int, int, float, int, float]] = {
    "kicad-dsl":   (16, 2, 5e-5, 2048, 0.0),
    "spice":       (16, 2, 5e-5, 2048, 0.0),
    "emc":         (12, 2, 3e-5, 2048, 0.0),
    "stm32":       (8,  2, 3e-5, 2048, 0.0),
    "embedded":    (12, 1, 3e-5, 2048, 0.0),   # huge dataset, 1 epoch
    "freecad":     (4,  2, 2e-5, 2048, 0.1),
    "platformio":  (4,  2, 2e-5, 2048, 0.1),
    "power":       (8,  2, 3e-5, 2048, 0.0),
    "dsp":         (8,  2, 3e-5, 2048, 0.0),
    "electronics": (12, 2, 3e-5, 2048, 0.0),
}

DOMAIN_ORDER = list(NICHE_DOMAINS.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_training_data(domain: str) -> Path:
    """Resolve training data path (merged first, then KIKI fallback)."""
    merged = MERGED_DATA / domain / "train.jsonl"
    if merged.exists() and merged.stat().st_size > 0:
        logger.debug("Data source: merged (%s)", merged)
        return merged

    kiki = KIKI_DATA / domain / "train.jsonl"
    if kiki.exists() and kiki.stat().st_size > 0:
        logger.debug("Data source: KIKI (%s)", kiki)
        return kiki

    raise FileNotFoundError(
        f"No training data for '{domain}'. "
        f"Checked:\n  {merged}\n  {kiki}"
    )


def adapter_done(domain: str) -> bool:
    """Return True if adapter already trained (skip guard)."""
    adapter = OUTPUTS_DIR / f"stack-{domain}" / "adapters.safetensors"
    return adapter.exists()


def log_progress(msg: str) -> None:
    """Append a timestamped line to .ralph/progress.txt."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    with PROGRESS_FILE.open("a") as fh:
        fh.write(line)
    logger.info(msg)


def count_examples(data_path: Path) -> int:
    """Count JSONL lines for display."""
    try:
        return sum(1 for _ in data_path.open())
    except Exception:
        return -1


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------


def dry_run(domains: list[str]) -> None:
    """Print training plan without loading any model."""
    logger.info("DRY RUN — no model will be loaded")
    header = f"{'Domain':<14} {'rank':>4} {'ep':>3} {'lr':>8} {'seq':>5} {'drop':>5}  {'data':<60} {'status'}"
    print(header)
    print("-" * len(header))
    for domain in domains:
        rank, epochs, lr, seq_len, dropout = NICHE_DOMAINS[domain]
        try:
            data = find_training_data(domain)
            n = count_examples(data)
            data_label = f"{data} ({n} ex)"
        except FileNotFoundError as exc:
            data_label = f"MISSING — {exc}"

        status = "DONE (skip)" if adapter_done(domain) else "PENDING"
        print(
            f"{domain:<14} {rank:>4} {epochs:>3} {lr:>8.0e} {seq_len:>5} "
            f"{dropout:>5.1f}  {data_label:<60} {status}"
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def load_dataset(data_path: Path) -> list[dict]:
    """Load JSONL into a list of dicts."""
    records = []
    with data_path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Loaded %d examples from %s", len(records), data_path)
    return records


def train_domain(
    domain: str,
    model,
    tokenizer,
    *,
    dry_run: bool = False,
) -> None:
    """Apply LoRA for domain, train, save adapter, then discard LoRA weights."""
    from mlx_tune import FastLanguageModel, SFTTrainer

    rank, epochs, lr, seq_len, dropout = NICHE_DOMAINS[domain]
    output_dir = str(OUTPUTS_DIR / f"stack-{domain}")

    log_progress(f"START {domain} r={rank} ep={epochs} lr={lr} seq={seq_len}")

    data_path = find_training_data(domain)
    dataset = load_dataset(data_path)

    model = FastLanguageModel.get_peft_model(
        model,
        r=rank,
        lora_alpha=rank * 2,
        target_modules=LORA_TARGETS,
        lora_dropout=dropout,
        max_seq_length=seq_len,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=seq_len,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        logging_steps=10,
        save_steps=100,
        output_dir=output_dir,
        adapter_path=output_dir,
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    log_progress(f"DONE  {domain} in {elapsed:.0f}s → {output_dir}")


def run_training(
    domains: list[str],
    *,
    start_from: str | None = None,
    dry_run_mode: bool = False,
) -> None:
    """Load model once, iterate over domains sequentially."""
    if dry_run_mode:
        dry_run(domains)
        return

    # Resolve start domain
    if start_from is not None:
        if start_from not in DOMAIN_ORDER:
            logger.error("Unknown start domain: %s", start_from)
            sys.exit(1)
        start_idx = DOMAIN_ORDER.index(start_from)
        domains = [d for d in domains if DOMAIN_ORDER.index(d) >= start_idx]
        logger.info("Resuming from domain '%s' (%d remaining)", start_from, len(domains))

    pending = [d for d in domains if not adapter_done(d)]
    skipped = [d for d in domains if adapter_done(d)]

    if skipped:
        logger.info("Skipping %d already-trained domain(s): %s", len(skipped), skipped)

    if not pending:
        logger.info("All requested domains already trained. Nothing to do.")
        return

    # Lazy import — only when actually training
    from mlx_tune import FastLanguageModel

    model_path = str(MODEL_PATH) if MODEL_PATH.exists() else "Qwen/Qwen3.5-35B-A3B"

    results: list[tuple[str, str]] = []

    for domain in pending:
        logger.info("=" * 60)
        logger.info("Training domain: %s (%d/%d)", domain, pending.index(domain) + 1, len(pending))
        try:
            # Reload model fresh per domain — LoRA can't stack on LoRA
            logger.info("Loading base model from %s", model_path)
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_path,
                max_seq_length=4096,
                use_gradient_checkpointing=True,
            )
            train_domain(domain, model, tokenizer)
            results.append((domain, "OK"))
            # Free model memory before next domain
            del model, tokenizer
            import gc; gc.collect()
        except Exception as exc:
            logger.error("FAILED %s: %s", domain, exc, exc_info=True)
            log_progress(f"FAIL  {domain}: {exc}")
            results.append((domain, f"FAILED: {exc}"))

    # Final summary
    logger.info("=" * 60)
    logger.info("TRAINING SUMMARY")
    ok = [(d, s) for d, s in results if s == "OK"]
    failed = [(d, s) for d, s in results if s != "OK"]
    for d, s in skipped:
        print(f"  SKIP     {d}")
    for d, s in ok:
        print(f"  OK       {d}")
    for d, s in failed:
        print(f"  FAILED   {d}  — {s}")

    log_progress(f"SESSION DONE — {len(ok)} trained, {len(failed)} failed, {len(skipped)} skipped")

    if failed:
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_niches_mlxtune",
        description="Sequential LoRA training of 10 niche domains via mlx-tune.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available domains: {', '.join(DOMAIN_ORDER)}",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all",
        action="store_true",
        help="Train all 10 niche domains sequentially.",
    )
    group.add_argument(
        "--domain",
        metavar="NAME",
        help="Train a single domain.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show training plan without loading the model.",
    )
    parser.add_argument(
        "--start",
        metavar="DOMAIN",
        help="Resume from a specific domain (skip earlier ones). Only with --all.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.start and not args.all:
        parser.error("--start requires --all")

    if args.domain:
        if args.domain not in NICHE_DOMAINS:
            parser.error(
                f"Unknown domain '{args.domain}'. "
                f"Valid: {', '.join(DOMAIN_ORDER)}"
            )
        domains = [args.domain]
    else:
        domains = DOMAIN_ORDER

    run_training(
        domains,
        start_from=args.start,
        dry_run_mode=args.dry_run,
    )


if __name__ == "__main__":
    main()

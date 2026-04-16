#!/usr/bin/env python3
"""GRPO training for reasoning niche domains.

Applies Group Relative Policy Optimization (GRPO) after SFT+DPO
for domains that benefit from chain-of-thought: spice, emc, embedded, power.

Usage:
    uv run scripts/train_grpo_niches.py --all
    uv run scripts/train_grpo_niches.py --domain spice
    uv run scripts/train_grpo_niches.py --all --dry-run
"""
from __future__ import annotations

# Metal buffer fixes — must be set before any model loading.
try:
    import mlx.core as mx
    mx.set_memory_limit(460 * 1024**3)
    mx.set_cache_limit(32 * 1024**3)
except ModuleNotFoundError:
    pass

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("train_grpo_niches")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
MERGED_DATA = PROJECT_ROOT / "data" / "merged"
KIKI_DATA = Path.home() / "KIKI-Mac_tunner" / "data" / "micro-kiki"
MODEL_PATH = PROJECT_ROOT / "models" / "qwen3.5-35b-a3b"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "stacks"

# Only reasoning domains get GRPO
GRPO_DOMAINS: list[str] = ["spice", "emc", "embedded", "power"]

# domain: (rank, lr)
GRPO_CONFIG: dict[str, tuple[int, float]] = {
    "spice":    (16, 5e-6),
    "emc":      (12, 3e-6),
    "embedded": (12, 3e-6),
    "power":    (8,  3e-6),
}

# Sample prompts for dry-run scoring
_DRY_RUN_SAMPLES: dict[str, list[str]] = {
    "spice": [
        "Write a SPICE netlist for a 1kHz RC low-pass filter with -3dB at cutoff.",
        "Model a MOSFET switching circuit in SPICE with Vgs=5V and Vds=12V.",
        "Create a SPICE simulation for an inverting op-amp with gain=-10.",
        "Show a SPICE netlist for a full-wave rectifier with smoothing capacitor.",
        "Write .tran simulation for a 555 timer in astable mode at 1kHz.",
    ],
    "emc": [
        "Explain how to calculate the emission margin for a 50MHz clock trace.",
        "What ferrite bead value should I choose to suppress 100MHz switching noise?",
        "Describe a common-mode choke placement strategy for a USB 3.0 interface.",
        "How do you calculate the shielding effectiveness of a metal enclosure?",
        "What are the key differences between conducted and radiated EMI?",
    ],
    "embedded": [
        "Write an STM32 HAL driver to configure SPI in DMA mode.",
        "Implement a circular buffer in C for an embedded UART RX handler.",
        "How do you configure the ESP32 watchdog timer to reset after 5 seconds?",
        "Write a FreeRTOS task that reads ADC and publishes to a queue.",
        "Explain how to debug a hard fault on ARM Cortex-M4.",
    ],
    "power": [
        "Design a synchronous buck converter for 12V input, 3.3V/2A output.",
        "Calculate the inductor value for a boost converter at 400kHz switching.",
        "What is the thermal resistance needed for a MOSFET dissipating 5W?",
        "Explain the difference between CCM and DCM in a flyback converter.",
        "Design an LDO bypass network for low-noise 1.8V rail from 3.3V.",
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_training_data(domain: str) -> Path:
    """Resolve GRPO training data (merged then KIKI fallback)."""
    for base in (MERGED_DATA, KIKI_DATA):
        path = base / domain / "train.jsonl"
        if path.exists() and path.stat().st_size > 0:
            return path
    raise FileNotFoundError(
        f"No training data for '{domain}'. "
        f"Checked:\n  {MERGED_DATA / domain / 'train.jsonl'}"
        f"\n  {KIKI_DATA / domain / 'train.jsonl'}"
    )


def load_prompts(data_path: Path, max_prompts: int = 2000) -> list[str]:
    """Extract prompts from JSONL training data."""
    prompts: list[str] = []
    with data_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Support both chat and instruction formats
            if "messages" in obj:
                user_msgs = [
                    m["content"]
                    for m in obj["messages"]
                    if m.get("role") == "user"
                ]
                if user_msgs:
                    prompts.append(user_msgs[0])
            elif "instruction" in obj:
                prompts.append(obj["instruction"])
            elif "prompt" in obj:
                prompts.append(obj["prompt"])
            if len(prompts) >= max_prompts:
                break
    return prompts


def sft_adapter_exists(domain: str) -> bool:
    """Return True if SFT adapter is available as the starting checkpoint."""
    for suffix in ("adapters.safetensors", "adapter_model.safetensors"):
        if (OUTPUTS_DIR / f"stack-{domain}" / suffix).exists():
            return True
    return False


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------


def dry_run(domains: list[str]) -> None:
    """Show reward function scores on sample prompts without loading a model."""
    from src.eval.reward_functions import combined_reward

    logger.info("DRY RUN — scoring %d sample prompts per domain", 5)

    for domain in domains:
        samples = _DRY_RUN_SAMPLES.get(domain, [])
        if not samples:
            logger.warning("No sample prompts defined for '%s'", domain)
            continue

        print(f"\n{'='*60}")
        print(f"Domain: {domain}")
        print(f"{'='*60}")

        # Use a mock response for scoring (simulate a mediocre answer)
        mock_response = (
            f"For {domain}, the key steps are:\n"
            "1. Analyse the requirements\n"
            "2. Choose appropriate components\n"
            "3. Verify with simulation\n"
            "The result should comply with relevant standards. "
            "R1 = 10kΩ, C1 = 100nF. ```python\n# code here\n```"
        )

        scores = []
        for i, prompt in enumerate(samples[:5]):
            # accuracy_reward makes HTTP calls — skip in dry-run
            score = combined_reward(
                prompt,
                mock_response,
                domain,
                weights={"syntax": 0.3, "format": 0.3, "completeness": 0.4, "accuracy": 0.0},
            )
            scores.append(score)
            trunc = prompt[:70] + "..." if len(prompt) > 70 else prompt
            print(f"  [{i+1}] score={score:.3f}  prompt={trunc!r}")

        avg = sum(scores) / len(scores)
        print(f"  avg={avg:.3f}  sft_adapter={'YES' if sft_adapter_exists(domain) else 'NO'}")

        try:
            data = find_training_data(domain)
            n = sum(1 for _ in data.open())
            print(f"  data={data} ({n} examples)")
        except FileNotFoundError as exc:
            print(f"  data=MISSING — {exc}")


# ---------------------------------------------------------------------------
# GRPO training
# ---------------------------------------------------------------------------


def train_grpo(domain: str) -> None:
    """Run GRPO training for one domain using mlx-tune GRPOTrainer."""
    from src.eval.reward_functions import combined_reward

    rank, lr = GRPO_CONFIG[domain]
    output_dir = OUTPUTS_DIR / f"stack-{domain}-grpo"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_id = str(MODEL_PATH) if MODEL_PATH.exists() else "Qwen/Qwen3.5-35B-A3B"
    sft_path = OUTPUTS_DIR / f"stack-{domain}"

    data_path = find_training_data(domain)
    prompts = load_prompts(data_path)
    logger.info(
        "Domain=%s  rank=%d  lr=%s  prompts=%d  output=%s",
        domain, rank, lr, len(prompts), output_dir,
    )

    # Attempt to import mlx-tune GRPOTrainer (API may differ across versions)
    try:
        from mlx_tune import FastLanguageModel, GRPOConfig, GRPOTrainer  # type: ignore[import]
    except ImportError as exc:
        logger.error(
            "mlx-tune not installed or GRPOTrainer not available: %s\n"
            "Install with: pip install mlx-tune>=0.4\n"
            "Ensure mlx-tune provides GRPOTrainer + GRPOConfig.",
            exc,
        )
        raise SystemExit(1) from exc

    logger.info("Loading model from %s (SFT checkpoint: %s)", model_id, sft_path)
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=2048,
            load_in_4bit=False,  # BF16 for training
        )
    except Exception as exc:
        logger.error("Model loading failed: %s", exc)
        raise

    model = FastLanguageModel.get_peft_model(
        model,
        r=rank,
        lora_alpha=rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing=True,
    )

    # Load from SFT adapter if available
    if sft_path.exists():
        logger.info("Loading SFT adapter weights from %s", sft_path)
        try:
            model.load_adapter(str(sft_path))
        except Exception as exc:
            logger.warning("Could not load SFT adapter: %s — starting from scratch", exc)

    config = GRPOConfig(
        output_dir=str(output_dir),
        learning_rate=lr,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        group_size=4,   # generate 4 responses per prompt for relative reward
        max_new_tokens=512,
        temperature=0.9,
        save_steps=50,
        logging_steps=10,
    )

    # Wrap combined_reward with the domain bound so it matches GRPO signature
    def reward_fn(prompt: str, response: str) -> float:
        return combined_reward(prompt, response, domain)

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        reward_funcs=[reward_fn],
        train_dataset=prompts,
    )

    logger.info("Starting GRPO training for domain=%s", domain)
    try:
        trainer.train()
    except Exception as exc:
        logger.error(
            "GRPO training failed for domain=%s: %s\n"
            "Common causes:\n"
            "  - GRPOConfig field name mismatch (check mlx-tune version)\n"
            "  - Metal OOM (reduce group_size or max_new_tokens)\n"
            "  - reward_funcs signature mismatch",
            domain, exc,
        )
        raise

    logger.info("GRPO training complete. Adapter saved to %s", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_grpo_niches",
        description=(
            "GRPO training for reasoning niche domains "
            "(spice, emc, embedded, power)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Reasoning domains: {', '.join(GRPO_DOMAINS)}",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all",
        action="store_true",
        help="Train all 4 reasoning domains sequentially.",
    )
    group.add_argument(
        "--domain",
        metavar="NAME",
        choices=GRPO_DOMAINS,
        help=f"Train a single domain. Choices: {', '.join(GRPO_DOMAINS)}",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show reward scores on sample prompts without loading the model.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    domains = GRPO_DOMAINS if args.all else [args.domain]

    if args.dry_run:
        dry_run(domains)
        return

    failed: list[str] = []
    for domain in domains:
        logger.info("=" * 60)
        logger.info("GRPO domain: %s", domain)
        try:
            train_grpo(domain)
        except SystemExit:
            raise
        except Exception as exc:
            logger.error("FAILED %s: %s", domain, exc, exc_info=True)
            failed.append(domain)

    logger.info("=" * 60)
    ok = [d for d in domains if d not in failed]
    for d in ok:
        logger.info("OK       %s", d)
    for d in failed:
        logger.info("FAILED   %s", d)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

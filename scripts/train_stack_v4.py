#!/usr/bin/env python3
"""Train a single V4 domain stack with Brainstacks null-space projection.

Usage:
    # Train with null-space projection (auto-discover frozen stacks):
    python scripts/train_stack_v4.py --domain power

    # Train without null-space (ablation baseline):
    python scripts/train_stack_v4.py --domain power --no-null-space

    # Specify frozen stacks explicitly:
    python scripts/train_stack_v4.py --domain power --frozen-stacks chat-fr,python,cpp

    # Override iterations:
    python scripts/train_stack_v4.py --domain power --iters 50
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

# Ensure repo root is importable when run directly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.stacks.train_loop_v4 import train_stack  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_stack_v4")

_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "brainstacks-v4.yaml"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a V4 LoRA stack with Brainstacks null-space projection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--domain",
        required=True,
        help="Domain name to train (e.g. power, chat-fr, python).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help=f"Path to brainstacks YAML config (default: {_DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--frozen-stacks",
        default=None,
        help=(
            "Comma-separated list of domain names whose adapters are frozen "
            "(e.g. chat-fr,python,cpp). If omitted, auto-discovered from adapter_dir."
        ),
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=None,
        help="Override training iteration count from config.",
    )
    parser.add_argument(
        "--no-null-space",
        action="store_true",
        help="Disable null-space projection (ablation baseline).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override output directory for the trained adapter.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_config(path: Path) -> dict[str, Any]:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    logger.info("Config loaded from %s", path)
    return cfg


def _curriculum_tier(domain: str, cfg: dict[str, Any]) -> str:
    """Return 'foundations', 'coding', or 'niche' for a given domain."""
    curriculum = cfg.get("curriculum", {})
    if domain in (curriculum.get("foundations") or []):
        return "foundations"
    if domain in (curriculum.get("coding") or []):
        return "coding"
    return "niche"


def _default_iters(domain: str, cfg: dict[str, Any]) -> int:
    """Return iteration count from config based on curriculum tier."""
    tier = _curriculum_tier(domain, cfg)
    training = cfg.get("training", {})
    key = f"iters_{tier}"
    count = training.get(key)
    if count is None:
        raise KeyError(f"Missing training.{key} in config for tier '{tier}'")
    return int(count)


# ---------------------------------------------------------------------------
# Frozen-stack discovery
# ---------------------------------------------------------------------------

def _auto_discover_frozen_stacks(adapter_dir: Path, domain: str) -> list[str]:
    """Return paths to all adapter dirs in adapter_dir that are NOT the current domain.

    Looks for subdirectories containing adapters.safetensors.
    """
    frozen_paths: list[str] = []
    if not adapter_dir.is_dir():
        logger.warning("adapter_dir %s does not exist — no frozen stacks loaded.", adapter_dir)
        return frozen_paths

    for entry in sorted(adapter_dir.iterdir()):
        if not entry.is_dir():
            continue
        adapter_file = entry / "adapters.safetensors"
        if not adapter_file.exists():
            continue
        if entry.name == domain:
            continue  # skip the domain being trained
        frozen_paths.append(str(entry))

    logger.info(
        "Auto-discovered %d frozen stacks in %s (excluding '%s')",
        len(frozen_paths),
        adapter_dir,
        domain,
    )
    return frozen_paths


def _resolve_frozen_stacks(
    frozen_stacks_arg: str | None,
    adapter_dir: Path,
    domain: str,
) -> list[str]:
    """Build the list of frozen adapter paths from CLI arg or auto-discovery."""
    if frozen_stacks_arg is not None:
        names = [n.strip() for n in frozen_stacks_arg.split(",") if n.strip()]
        paths: list[str] = []
        for name in names:
            candidate = adapter_dir / name
            if not (candidate / "adapters.safetensors").exists():
                logger.warning(
                    "Frozen stack '%s' not found at %s — skipping.", name, candidate
                )
                continue
            paths.append(str(candidate))
        logger.info("Using %d explicitly specified frozen stacks.", len(paths))
        return paths
    return _auto_discover_frozen_stacks(adapter_dir, domain)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file, extracting text from messages or plain text fields."""
    records: list[dict[str, Any]] = []
    with open(path) as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON at line %d: %s", lineno, exc)
                continue

            # Normalise: extract plain text from chat-style messages format.
            if "text" in obj:
                records.append({"text": obj["text"]})
            elif "messages" in obj:
                parts: list[str] = []
                for msg in obj["messages"]:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if content:
                        parts.append(f"<|{role}|>\n{content}")
                text = "\n".join(parts)
                records.append({"text": text})
            else:
                # Fallback: serialise the whole object as text.
                records.append({"text": json.dumps(obj, ensure_ascii=False)})

    logger.info("Loaded %d records from %s", len(records), path)
    return records


# ---------------------------------------------------------------------------
# Forgetting check
# ---------------------------------------------------------------------------

def _run_forgetting_check(
    new_adapter_dir: Path,
    frozen_adapter_paths: list[str],
    domain: str,
) -> None:
    """Run measure_forgetting.py against the most recently frozen adapter (angle-only).

    This is informational (angle-only phase 1a). The full gate with win-rate
    requires running run_forgetting.sh / post_train_gate.py separately.
    """
    if not frozen_adapter_paths:
        logger.info("No frozen stacks — skipping forgetting check.")
        return

    prior_adapter = Path(frozen_adapter_paths[-1]) / "adapters.safetensors"
    new_adapter = new_adapter_dir / "adapters.safetensors"

    if not prior_adapter.exists():
        logger.warning("Prior adapter not found at %s — skipping forgetting check.", prior_adapter)
        return
    if not new_adapter.exists():
        logger.warning("New adapter not found at %s — skipping forgetting check.", new_adapter)
        return

    results_dir = _REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_json = results_dir / f"forgetting-v4-{domain}.json"

    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "measure_forgetting.py"),
        "--prior-adapter", str(prior_adapter),
        "--new-adapter", str(new_adapter),
        "--output", str(output_json),
    ]

    logger.info("Running forgetting check: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if proc.returncode != 0:
        logger.warning(
            "Forgetting check exited %d.\nstdout: %s\nstderr: %s",
            proc.returncode,
            proc.stdout[:2000],
            proc.stderr[:2000],
        )
    else:
        logger.info("Forgetting check passed. Results: %s", output_json)
        # Print the mean angle if available.
        if output_json.exists():
            try:
                report = json.loads(output_json.read_text())
                mean_angle = report.get("angle_degrees_mean")
                if mean_angle is not None:
                    logger.info("Mean forgetting angle: %.2f°", mean_angle)
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # 1. Load config.
    if not args.config.exists():
        logger.error("Config not found: %s", args.config)
        return 3
    cfg = _load_config(args.config)

    lora_cfg = cfg.get("lora", {})
    training_cfg = cfg.get("training", {})
    null_space_cfg = cfg.get("null_space", {})

    model_path = cfg.get("model", "")
    adapter_dir = Path(cfg.get("adapter_dir", ""))
    data_dir = Path(cfg.get("data_dir", ""))

    # 2. Resolve iteration count.
    if args.iters is not None:
        iters = args.iters
        logger.info("Iteration count overridden to %d.", iters)
    else:
        try:
            iters = _default_iters(args.domain, cfg)
        except KeyError as exc:
            logger.error("Could not determine iteration count: %s", exc)
            return 3
        logger.info(
            "Domain '%s' → tier '%s' → %d iters.",
            args.domain,
            _curriculum_tier(args.domain, cfg),
            iters,
        )

    # 3. Resolve output dir.
    output_dir = args.output or (adapter_dir / args.domain)
    logger.info("Output dir: %s", output_dir)

    # 4. Resolve frozen stacks.
    use_null_space = (not args.no_null_space) and null_space_cfg.get("enabled", True)
    if use_null_space:
        frozen_paths = _resolve_frozen_stacks(args.frozen_stacks, adapter_dir, args.domain)
    else:
        frozen_paths = []
        logger.info("Null-space projection DISABLED (ablation mode).")

    # 5. Load training data.
    train_file = data_dir / args.domain / "train.jsonl"
    if not train_file.exists():
        logger.error("Training data not found: %s", train_file)
        return 3
    train_data = _load_jsonl(train_file)
    if not train_data:
        logger.error("Training data is empty: %s", train_file)
        return 3

    # Validation data is optional; pass an empty list if not present.
    val_file = data_dir / args.domain / "val.jsonl"
    val_data = _load_jsonl(val_file) if val_file.exists() else []

    # 6. Load base model + apply LoRA.
    logger.info("Loading base model from %s …", model_path)
    try:
        import mlx.core as mx
        from mlx_lm import load as mlx_load
        from mlx_lm.tuner.utils import apply_lora_layers
    except ImportError as exc:
        logger.error("mlx_lm not available: %s", exc)
        return 3

    # Metal memory limits for M3 Ultra (hard invariant from CLAUDE.md).
    mx.set_memory_limit(460 * 1024 ** 3)
    mx.set_cache_limit(32 * 1024 ** 3)

    model, tokenizer = mlx_load(model_path)

    logger.info(
        "Applying LoRA: rank=%d, alpha=%d, num_layers=%d",
        lora_cfg.get("rank", 16),
        lora_cfg.get("alpha", 16),
        lora_cfg.get("num_layers", 32),
    )
    apply_lora_layers(
        model,
        num_lora_layers=lora_cfg.get("num_layers", 32),
        rank=lora_cfg.get("rank", 16),
    )
    model.train()  # enable gradient tracking

    # 7. Train.
    logger.info(
        "Starting training: domain=%s, iters=%d, null_space=%s, frozen=%d stacks",
        args.domain,
        iters,
        use_null_space,
        len(frozen_paths),
    )
    metrics = train_stack(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        val_data=val_data,
        frozen_adapter_paths=frozen_paths,
        iters=iters,
        batch_size=training_cfg.get("batch_size", 1),
        learning_rate=lora_cfg.get("lr", 1e-5),
        max_seq_length=training_cfg.get("max_seq_length", 1024),
        null_space_top_k=null_space_cfg.get("top_k", 32),
        output_dir=str(output_dir),
    )

    # 8. Print summary.
    losses = metrics.get("train_losses", [])
    if losses:
        final_loss = losses[-1]
        mean_loss = sum(losses) / len(losses)
        logger.info(
            "Training complete — steps=%d, final_loss=%.4f, mean_loss=%.4f",
            len(losses),
            final_loss,
            mean_loss,
        )
    else:
        logger.warning("No training losses recorded.")

    # 9. Forgetting check (angle-only, informational).
    _run_forgetting_check(
        new_adapter_dir=Path(output_dir),
        frozen_adapter_paths=frozen_paths,
        domain=args.domain,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

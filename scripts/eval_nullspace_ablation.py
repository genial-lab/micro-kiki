#!/usr/bin/env python3
"""Ablation: null-space projection vs baseline on niche domain LoRA stacks.

Compares training WITH vs WITHOUT null-space projection on the same domains,
measuring forgetting angles against frozen foundation priors.

The base model is loaded ONCE; LoRA layers are reapplied (re-initialised)
between conditions so the model never carries state across runs.

Usage:
    python scripts/eval_nullspace_ablation.py
    python scripts/eval_nullspace_ablation.py --domains embedded,dsp,power
    python scripts/eval_nullspace_ablation.py --iters 50 --output results/ablation.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval.forgetting import _load_tensors, compute_angles  # noqa: E402
from src.stacks.train_loop_v4 import train_stack  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_nullspace_ablation")

_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "brainstacks-v4.yaml"
_DEFAULT_DOMAINS = ["embedded", "dsp", "power"]
_DEFAULT_FROZEN_PRIORS = ["chat-fr", "python"]
_DEFAULT_ITERS = 100
_DEFAULT_OUTPUT = _REPO_ROOT / "results" / "nullspace_ablation.json"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablation: null-space projection vs baseline on niche domains.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--domains",
        default=",".join(_DEFAULT_DOMAINS),
        help=(
            f"Comma-separated list of niche domains to ablate "
            f"(default: {','.join(_DEFAULT_DOMAINS)})."
        ),
    )
    parser.add_argument(
        "--frozen-priors",
        default=",".join(_DEFAULT_FROZEN_PRIORS),
        help=(
            f"Comma-separated list of frozen prior domain names used for angle "
            f"measurement (default: {','.join(_DEFAULT_FROZEN_PRIORS)})."
        ),
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=_DEFAULT_ITERS,
        help=f"Training iterations per condition (default: {_DEFAULT_ITERS}).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help=f"Path to brainstacks YAML config (default: {_DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Output JSON report path (default: {_DEFAULT_OUTPUT}).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Config / data helpers
# ---------------------------------------------------------------------------

def _load_config(path: Path) -> dict[str, Any]:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    logger.info("Config loaded from %s", path)
    return cfg


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file, normalising records to {text: ...}."""
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
            if "text" in obj:
                records.append({"text": obj["text"]})
            elif "messages" in obj:
                parts: list[str] = []
                for msg in obj["messages"]:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if content:
                        parts.append(f"<|{role}|>\n{content}")
                records.append({"text": "\n".join(parts)})
            else:
                records.append({"text": json.dumps(obj, ensure_ascii=False)})
    logger.info("Loaded %d records from %s", len(records), path)
    return records


# ---------------------------------------------------------------------------
# Angle measurement
# ---------------------------------------------------------------------------

def _measure_angles_vs_priors(
    new_adapter_path: Path,
    prior_adapter_paths: dict[str, Path],
) -> dict[str, float]:
    """Return {prior_name: mean_angle_degrees} for each prior."""
    new_tensors = _load_tensors(new_adapter_path)
    angles: dict[str, float] = {}
    for name, prior_path in prior_adapter_paths.items():
        if not prior_path.exists():
            logger.warning("Prior adapter not found: %s — skipping.", prior_path)
            continue
        prior_tensors = _load_tensors(prior_path)
        per_module = compute_angles(prior_tensors, new_tensors)
        if not per_module:
            logger.warning("No shared LoRA modules between new adapter and %s.", name)
            angles[name] = float("nan")
            continue
        mean = float(sum(per_module.values()) / len(per_module))
        angles[name] = mean
        logger.info("  angle vs %s: %.2f° (across %d modules)", name, mean, len(per_module))
    return angles


# ---------------------------------------------------------------------------
# LoRA reset
# ---------------------------------------------------------------------------

def _remove_all_lora(model: Any) -> None:
    """Remove ALL LoRA layer types (Linear, SwitchLinear, Embedding)."""
    from mlx_lm.tuner.utils import LoRALinear, LoRASwitchLinear, LoRAEmbedding
    from mlx.utils import tree_unflatten

    reset_layers = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            reset_layers.append((name, module.linear))
        elif isinstance(module, LoRASwitchLinear):
            reset_layers.append((name, module.linear))
        elif isinstance(module, LoRAEmbedding):
            reset_layers.append((name, module.embedding))
    if reset_layers:
        model.update_modules(tree_unflatten(reset_layers))


def _apply_fresh_lora(model: Any, lora_cfg: dict[str, Any]) -> None:
    """Strip any existing LoRA state and apply fresh (zero-initialised) layers."""
    from mlx_lm.tuner.utils import linear_to_lora_layers

    _remove_all_lora(model)

    linear_to_lora_layers(
        model,
        num_layers=lora_cfg.get("num_layers", 32),
        config={"rank": lora_cfg.get("rank", 16), "alpha": lora_cfg.get("alpha", 16), "scale": 20.0, "dropout": 0.0},
    )
    model.train()
    logger.info(
        "LoRA reset: rank=%d, alpha=%d, num_layers=%d",
        lora_cfg.get("rank", 16),
        lora_cfg.get("alpha", 16),
        lora_cfg.get("num_layers", 32),
    )


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------

def _run_condition(
    *,
    model: Any,
    tokenizer: Any,
    train_data: list[dict],
    val_data: list[dict],
    frozen_adapter_paths: list[str],
    iters: int,
    lora_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    null_space_cfg: dict[str, Any],
    output_dir: Path,
    use_null_space: bool,
) -> dict[str, Any]:
    """Reset LoRA, train, save, return loss metrics."""
    _apply_fresh_lora(model, lora_cfg)

    effective_frozen = frozen_adapter_paths if use_null_space else []
    ns_label = "with_nullspace" if use_null_space else "without_nullspace"
    logger.info("=== Training condition: %s (%d iters) ===", ns_label, iters)

    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    metrics = train_stack(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        val_data=val_data,
        frozen_adapter_paths=effective_frozen,
        iters=iters,
        batch_size=training_cfg.get("batch_size", 1),
        learning_rate=lora_cfg.get("lr", 1e-5),
        max_seq_length=training_cfg.get("max_seq_length", 1024),
        null_space_top_k=null_space_cfg.get("top_k", 32),
        output_dir=str(output_dir),
    )

    elapsed = time.time() - t0
    losses = metrics.get("train_losses", [])
    final_loss = losses[-1] if losses else float("nan")
    logger.info(
        "Condition %s done in %.1fs — final_loss=%.4f", ns_label, elapsed, final_loss
    )
    return {"final_train_loss": final_loss, "train_losses": losses}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

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

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    frozen_prior_names = [n.strip() for n in args.frozen_priors.split(",") if n.strip()]

    logger.info("Ablation plan: %d domains × 2 conditions = %d runs", len(domains), len(domains) * 2)
    logger.info("Domains: %s", domains)
    logger.info("Frozen priors for angle measurement: %s", frozen_prior_names)

    # Build prior adapter paths (used only for angle measurement after training).
    prior_adapter_paths: dict[str, Path] = {}
    for name in frozen_prior_names:
        p = adapter_dir / name / "adapters.safetensors"
        if not p.exists():
            logger.warning("Prior adapter not found: %s — angles vs this prior will be NaN.", p)
        prior_adapter_paths[name] = p

    # Frozen stack paths for null-space projection during training.
    frozen_paths_for_ns: list[str] = []
    for name in frozen_prior_names:
        d = adapter_dir / name
        if (d / "adapters.safetensors").exists():
            frozen_paths_for_ns.append(str(d))
        else:
            logger.warning("Frozen stack '%s' not found at %s — excluded from projection.", name, d)

    # -------------------------------------------------------------------------
    # Load base model ONCE.
    # -------------------------------------------------------------------------
    logger.info("Loading base model from %s …", model_path)
    try:
        import mlx.core as mx
        from mlx_lm import load as mlx_load
    except ImportError as exc:
        logger.error("mlx_lm not available: %s", exc)
        return 3

    # Metal memory limits (hard invariant — CLAUDE.md).
    mx.set_memory_limit(460 * 1024 ** 3)
    mx.set_cache_limit(32 * 1024 ** 3)

    model, tokenizer = mlx_load(model_path)
    logger.info("Base model loaded.")

    # -------------------------------------------------------------------------
    # Ablation loop.
    # -------------------------------------------------------------------------
    ablation_results: list[dict[str, Any]] = []
    ablation_output_root = _REPO_ROOT / "output" / "ablation"

    for domain in domains:
        logger.info("=== Domain: %s ===", domain)

        # Load training data.
        train_file = data_dir / domain / "train.jsonl"
        if not train_file.exists():
            logger.error("Training data not found: %s — skipping domain.", train_file)
            continue
        train_data = _load_jsonl(train_file)
        if not train_data:
            logger.error("Training data is empty: %s — skipping domain.", train_file)
            continue
        val_file = data_dir / domain / "val.jsonl"
        val_data = _load_jsonl(val_file) if val_file.exists() else []

        domain_result: dict[str, Any] = {
            "domain": domain,
            "frozen_priors": frozen_prior_names,
        }

        # --- WITH null-space ---
        out_ns = ablation_output_root / domain / "with_nullspace"
        ns_metrics = _run_condition(
            model=model,
            tokenizer=tokenizer,
            train_data=train_data,
            val_data=val_data,
            frozen_adapter_paths=frozen_paths_for_ns,
            iters=args.iters,
            lora_cfg=lora_cfg,
            training_cfg=training_cfg,
            null_space_cfg=null_space_cfg,
            output_dir=out_ns,
            use_null_space=True,
        )
        ns_adapter = out_ns / "adapters.safetensors"
        angles_ns = _measure_angles_vs_priors(ns_adapter, prior_adapter_paths)
        finite_ns = [v for v in angles_ns.values() if math.isfinite(v)]
        mean_ns = float(sum(finite_ns) / len(finite_ns)) if finite_ns else float("nan")
        domain_result["with_nullspace"] = {
            "final_train_loss": ns_metrics["final_train_loss"],
            "angles_vs_priors": angles_ns,
            "mean_angle": mean_ns,
        }

        # --- WITHOUT null-space ---
        out_base = ablation_output_root / domain / "without_nullspace"
        base_metrics = _run_condition(
            model=model,
            tokenizer=tokenizer,
            train_data=train_data,
            val_data=val_data,
            frozen_adapter_paths=[],
            iters=args.iters,
            lora_cfg=lora_cfg,
            training_cfg=training_cfg,
            null_space_cfg=null_space_cfg,
            output_dir=out_base,
            use_null_space=False,
        )
        base_adapter = out_base / "adapters.safetensors"
        angles_base = _measure_angles_vs_priors(base_adapter, prior_adapter_paths)
        finite_base = [v for v in angles_base.values() if math.isfinite(v)]
        mean_base = float(sum(finite_base) / len(finite_base)) if finite_base else float("nan")
        domain_result["without_nullspace"] = {
            "final_train_loss": base_metrics["final_train_loss"],
            "angles_vs_priors": angles_base,
            "mean_angle": mean_base,
        }

        # Delta: positive means null-space preserved more orthogonality.
        delta = (
            mean_ns - mean_base
            if math.isfinite(mean_ns) and math.isfinite(mean_base)
            else float("nan")
        )
        domain_result["delta_angle"] = delta

        logger.info(
            "Domain %s — with_ns mean_angle=%.2f°, without_ns mean_angle=%.2f°, delta=%.2f°",
            domain, mean_ns, mean_base, delta,
        )
        ablation_results.append(domain_result)

    # -------------------------------------------------------------------------
    # Write report.
    # -------------------------------------------------------------------------
    report = {"ablation": ablation_results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info("Ablation report written to %s", args.output)

    # Summary table.
    logger.info("\n%-12s  %10s  %13s  %8s", "domain", "with_ns °", "without_ns °", "delta °")
    logger.info("-" * 50)
    for r in ablation_results:
        w = r.get("with_nullspace", {}).get("mean_angle", float("nan"))
        wo = r.get("without_nullspace", {}).get("mean_angle", float("nan"))
        d = r.get("delta_angle", float("nan"))
        logger.info("%-12s  %10.2f  %13.2f  %8.2f", r["domain"], w, wo, d)

    return 0


if __name__ == "__main__":
    sys.exit(main())

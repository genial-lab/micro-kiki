#!/usr/bin/env python3
"""PPL evaluation: V4 LoRA adapters vs base Qwen3.6-35B-A3B on all 35 domains.

Measures per-domain perplexity on valid.jsonl with and without the V4 adapter,
then reports the delta (negative = improvement).

Architecture notes (from CLAUDE.md):
  - Base: models/Qwen3.6-35B-A3B (4-bit, MLX)
  - Adapters: KIKI-Mac_tunner/output/micro-kiki/lora-qwen36-35b-v4-sota/<domain>/
  - Data: KIKI-Mac_tunner/data/micro-kiki/<domain>/valid.jsonl
  - All valid.jsonl files use {"messages": [...]} format

Usage:
    uv run python scripts/eval_ppl_all_domains.py
    uv run python scripts/eval_ppl_all_domains.py --domain kicad-dsl
    uv run python scripts/eval_ppl_all_domains.py --batch-size 2 --max-seq-length 512
    uv run python scripts/eval_ppl_all_domains.py --output results/v4-ppl-35domains/custom.json

Runtime estimate: ~4–5 min/domain on M3 Ultra 512 GB → ~3 h for all 35 domains.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Base model lives in the tuner sibling repo
TUNER_ROOT = Path.home() / "KIKI-Mac_tunner"
BASE_MODEL_PATH = TUNER_ROOT / "models" / "Qwen3.6-35B-A3B"

# V4 adapters root
V4_ADAPTERS_ROOT = TUNER_ROOT / "output" / "micro-kiki" / "lora-qwen36-35b-v4-sota"

# Training data root
DATA_ROOT = TUNER_ROOT / "data" / "micro-kiki"

# Default output
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "v4-ppl-35domains" / "ppl-all-domains.json"

# ---------------------------------------------------------------------------
# Domain registry — 35 V4 domains (matches lora-qwen36-35b-v4-sota/ dirs)
# ---------------------------------------------------------------------------

ALL_DOMAINS: list[str] = [
    # Foundations
    "chat-fr",
    "reasoning",
    "python",
    # Hardware / EDA niches
    "kicad-dsl",
    "kicad-pcb",
    "spice",
    "spice-sim",
    "emc",
    "stm32",
    "embedded",
    "freecad",
    "platformio",
    "power",
    "dsp",
    "electronics",
    "iot",
    "components",
    # General coding
    "cpp",
    "rust",
    "typescript",
    "html-css",
    "shell",
    "sql",
    "yaml-json",
    "lua-upy",
    # Web / backend
    "web-frontend",
    "web-backend",
    "docker",
    "devops",
    # ML / Ops
    "llm-ops",
    "llm-orch",
    "ml-training",
    # Other
    "math",
    "security",
    "music-audio",
]

# ---------------------------------------------------------------------------
# Metal / MLX setup
# ---------------------------------------------------------------------------

MAX_SEQ_LENGTH_DEFAULT = 512
BATCH_SIZE_DEFAULT = 1


def _setup_metal() -> None:
    """Configure Metal buffer and cache limits for Mac Studio M3 Ultra 512 GB."""
    import mlx.core as mx

    mx.set_memory_limit(460 * 1024 * 1024 * 1024)  # 460 GB
    mx.set_cache_limit(32 * 1024 * 1024 * 1024)    # 32 GB
    logger.debug("Metal limits set: memory=460 GB, cache=32 GB")


# ---------------------------------------------------------------------------
# Dataset loading (reuses mlx_lm internals)
# ---------------------------------------------------------------------------


def _load_valid_dataset(domain: str, tokenizer: Any) -> Any:
    """Load valid.jsonl for a domain using mlx_lm dataset utilities.

    Returns a ChatDataset (or compatible) instance, or None if unavailable.
    """
    from mlx_lm.tuner.datasets import create_dataset

    valid_path = DATA_ROOT / domain / "valid.jsonl"
    if not valid_path.exists():
        logger.warning("[%s] valid.jsonl not found at %s", domain, valid_path)
        return None

    data: list[dict[str, Any]] = []
    with valid_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not data:
        logger.warning("[%s] valid.jsonl is empty", domain)
        return None

    # Minimal config object — only fields read by create_dataset
    class _Cfg:
        mask_prompt = False
        prompt_feature = "prompt"
        text_feature = "text"
        completion_feature = "completion"
        chat_feature = "messages"

    try:
        dataset = create_dataset(data, tokenizer, _Cfg())
    except ValueError as exc:
        logger.error("[%s] create_dataset failed: %s", domain, exc)
        return None

    logger.info("[%s] Loaded %d validation examples", domain, len(dataset))
    return dataset


# ---------------------------------------------------------------------------
# PPL computation
# ---------------------------------------------------------------------------


def _compute_ppl(
    model: Any,
    dataset: Any,
    tokenizer: Any,
    batch_size: int,
    max_seq_length: int,
) -> float:
    """Compute perplexity = exp(avg NLL) on the full dataset.

    Tokenizes each example via chat template, computes cross-entropy
    loss directly. Returns float('inf') on failure.
    """
    import mlx.core as mx
    import mlx.nn as nn

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for item in dataset:
        # item is a dict with 'messages' key
        messages = item.get("messages", [])
        if not messages:
            continue
        try:
            tokens = tokenizer.apply_chat_template(
                messages, return_dict=False,
            )
        except Exception:
            continue
        if len(tokens) < 4:
            continue
        tokens = tokens[:max_seq_length]

        x = mx.array([tokens[:-1]])
        y = mx.array([tokens[1:]])
        logits = model(x)
        loss = nn.losses.cross_entropy(logits, y, reduction="sum")
        mx.eval(loss)

        total_loss += loss.item()
        total_tokens += len(tokens) - 1

    if total_tokens == 0:
        return float("inf")

    avg_nll = total_loss / total_tokens
    ppl = math.exp(min(avg_nll, 100))
    return round(ppl, 4)


# ---------------------------------------------------------------------------
# Per-domain evaluation
# ---------------------------------------------------------------------------


def eval_domain(
    domain: str,
    base_model: Any,
    tokenizer: Any,
    batch_size: int,
    max_seq_length: int,
) -> dict[str, Any]:
    """Evaluate one domain: base PPL then adapted PPL.

    Loads the adapter in-place, evaluates, then unloads it to keep VRAM stable.
    """
    from mlx_lm.utils import load

    adapter_dir = V4_ADAPTERS_ROOT / domain
    if not (adapter_dir / "adapters.safetensors").exists():
        logger.error("[%s] adapter not found at %s", domain, adapter_dir)
        return {
            "domain": domain,
            "status": "missing_adapter",
            "base_ppl": None,
            "adapted_ppl": None,
            "delta": None,
        }

    dataset = _load_valid_dataset(domain, tokenizer)
    if dataset is None:
        return {
            "domain": domain,
            "status": "missing_data",
            "base_ppl": None,
            "adapted_ppl": None,
            "delta": None,
        }

    # --- Base PPL (no adapter) ---
    logger.info("[%s] Computing base PPL on %d examples ...", domain, len(dataset))
    t0 = time.perf_counter()
    try:
        base_ppl = _compute_ppl(base_model, dataset, tokenizer, batch_size, max_seq_length)
    except Exception as exc:
        logger.error("[%s] base PPL failed: %s", domain, exc)
        return {
            "domain": domain,
            "status": "error_base",
            "error": str(exc),
            "base_ppl": None,
            "adapted_ppl": None,
            "delta": None,
        }
    dt_base = time.perf_counter() - t0
    logger.info("[%s] base_ppl=%.4f  (%.1fs)", domain, base_ppl, dt_base)

    # --- Adapted PPL (load adapter) ---
    logger.info("[%s] Loading V4 adapter from %s ...", domain, adapter_dir)
    t1 = time.perf_counter()
    try:
        adapted_model, _ = load(
            path_or_hf_repo=str(BASE_MODEL_PATH),
            adapter_path=str(adapter_dir),
        )
        adapted_ppl = _compute_ppl(adapted_model, dataset, tokenizer, batch_size, max_seq_length)
        # Release adapter model explicitly
        del adapted_model
    except Exception as exc:
        logger.error("[%s] adapted PPL failed: %s", domain, exc)
        return {
            "domain": domain,
            "status": "error_adapted",
            "error": str(exc),
            "base_ppl": base_ppl,
            "adapted_ppl": None,
            "delta": None,
        }
    dt_adapted = time.perf_counter() - t1
    logger.info("[%s] adapted_ppl=%.4f  (%.1fs)", domain, adapted_ppl, dt_adapted)

    delta = round(adapted_ppl - base_ppl, 4)
    improved = delta < 0
    logger.info(
        "[%s] delta=%.4f (%s)  base=%.4f  adapted=%.4f",
        domain,
        delta,
        "IMPROVED" if improved else "DEGRADED",
        base_ppl,
        adapted_ppl,
    )

    return {
        "domain": domain,
        "status": "ok",
        "base_ppl": base_ppl,
        "adapted_ppl": adapted_ppl,
        "delta": delta,
        "improved": improved,
        "n_examples": len(dataset),
        "timing": {
            "base_s": round(dt_base, 1),
            "adapted_s": round(dt_adapted, 1),
        },
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def build_report(
    results: list[dict[str, Any]],
    elapsed_s: float,
) -> dict[str, Any]:
    """Assemble summary + per-domain JSON report."""
    ok = [r for r in results if r["status"] == "ok"]
    improved = [r for r in ok if r["improved"]]
    degraded = [r for r in ok if not r["improved"]]

    avg_base = (
        round(sum(r["base_ppl"] for r in ok) / len(ok), 4) if ok else None
    )
    avg_adapted = (
        round(sum(r["adapted_ppl"] for r in ok) / len(ok), 4) if ok else None
    )
    avg_delta = (
        round(sum(r["delta"] for r in ok) / len(ok), 4) if ok else None
    )

    ranked = sorted(ok, key=lambda r: r["delta"])

    return {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "base_model": str(BASE_MODEL_PATH),
            "adapters_root": str(V4_ADAPTERS_ROOT),
            "data_root": str(DATA_ROOT),
            "elapsed_s": round(elapsed_s, 1),
            "domains_total": len(results),
            "domains_ok": len(ok),
            "domains_improved": len(improved),
            "domains_degraded": len(degraded),
        },
        "summary": {
            "avg_base_ppl": avg_base,
            "avg_adapted_ppl": avg_adapted,
            "avg_delta": avg_delta,
            "best_improvement": ranked[0] if ranked else None,
            "worst_regression": ranked[-1] if ranked else None,
        },
        "ranked_by_delta": [
            {"domain": r["domain"], "delta": r["delta"], "base": r["base_ppl"], "adapted": r["adapted_ppl"]}
            for r in ranked
        ],
        "domains": {r["domain"]: r for r in results},
    }


def save_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info("Report saved → %s", output)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate PPL of V4 LoRA adapters vs base Qwen3.6-35B-A3B on all 35 domains.\n"
            "Runs on Mac Studio M3 Ultra via MLX. Estimated runtime: ~3 h for all domains."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Domains:\n  " + "\n  ".join(ALL_DOMAINS),
    )

    parser.add_argument(
        "--domain",
        metavar="DOMAIN",
        help=(
            "Evaluate a single domain (e.g. kicad-dsl). "
            "Defaults to all 35 domains when omitted."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE_DEFAULT,
        metavar="N",
        help=f"Batch size for PPL computation (default: {BATCH_SIZE_DEFAULT}). "
             "Increase to 2 if VRAM allows.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=MAX_SEQ_LENGTH_DEFAULT,
        metavar="L",
        help=f"Max token sequence length (default: {MAX_SEQ_LENGTH_DEFAULT}). "
             "Must match training max_seq_length in adapter_config.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        metavar="PATH",
        help=f"Output JSON file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from an existing output file — skip domains already present. "
            "Useful to restart a long run that was interrupted."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Validate paths
    if not BASE_MODEL_PATH.exists():
        logger.error("Base model not found: %s", BASE_MODEL_PATH)
        raise SystemExit(1)
    if not V4_ADAPTERS_ROOT.exists():
        logger.error("V4 adapters root not found: %s", V4_ADAPTERS_ROOT)
        raise SystemExit(1)

    domains = [args.domain] if args.domain else ALL_DOMAINS
    if args.domain and args.domain not in ALL_DOMAINS:
        logger.warning("Domain '%s' not in standard list — proceeding anyway.", args.domain)

    # Resume: load existing results and skip completed domains
    existing: dict[str, Any] = {}
    if args.resume and args.output.exists():
        try:
            prev = json.loads(args.output.read_text())
            existing = prev.get("domains", {})
            logger.info("Resume: found %d completed domains in %s", len(existing), args.output)
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Could not parse existing output for resume: %s", exc)

    pending = [d for d in domains if d not in existing]
    if not pending:
        logger.info("All domains already completed. Nothing to do.")
        return

    logger.info(
        "PPL eval: %d domains pending  batch_size=%d  max_seq_length=%d",
        len(pending),
        args.batch_size,
        args.max_seq_length,
    )

    # Setup Metal
    _setup_metal()

    # Load base model once — reused for all base PPL evaluations
    from mlx_lm.utils import load

    logger.info("Loading base model from %s ...", BASE_MODEL_PATH)
    t_load = time.perf_counter()
    base_model, tokenizer = load(path_or_hf_repo=str(BASE_MODEL_PATH))
    logger.info("Base model loaded in %.1fs", time.perf_counter() - t_load)

    # Evaluate each domain
    results: list[dict[str, Any]] = list(existing.values())
    t_start = time.perf_counter()

    for i, domain in enumerate(pending, 1):
        logger.info("--- [%d/%d] %s ---", i, len(pending), domain)
        result = eval_domain(
            domain=domain,
            base_model=base_model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
        )
        results.append(result)

        # Incremental save after each domain (crash-safe)
        partial_report = build_report(results, time.perf_counter() - t_start)
        save_report(partial_report, args.output)

    elapsed = time.perf_counter() - t_start

    # Final report
    report = build_report(results, elapsed)
    save_report(report, args.output)

    ok_results = [r for r in results if r["status"] == "ok"]
    improved = [r for r in ok_results if r["improved"]]

    logger.info(
        "Done: %d/%d ok  %d improved  %d degraded  avg_delta=%.4f  %.0fs total",
        len(ok_results),
        len(results),
        len(improved),
        len(ok_results) - len(improved),
        report["summary"]["avg_delta"] or float("nan"),
        elapsed,
    )
    logger.info("Results → %s", args.output)


if __name__ == "__main__":
    main()

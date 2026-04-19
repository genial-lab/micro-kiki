#!/usr/bin/env python
"""Pairwise forgetting-angle sweep across all adapters in a directory.

For every ordered pair ``(prior, new)`` with ``prior != new``, compute the
per-module cos-angle via :func:`src.eval.forgetting.measure_forgetting_signal`
(angle-only mode, no win-rate) and aggregate into a JSON matrix. Flags any
pair whose mean angle falls below the 30° gate threshold.

Usage:
    python scripts/run_forgetting_sweep.py \\
        --adapters-dir <root containing N subdirs each with adapters.safetensors> \\
        --output results/forgetting-matrix.json

Behaviour:
- Scans the directory for immediate subdirs containing ``adapters.safetensors``;
  the subdir name becomes the adapter label.
- Emits a JSON payload on stdout and (optionally) writes it to ``--output``.
- Exit 0 if all pair mean angles ≥ 30°, 1 if any pair falls below (forgetting).
- Progress logs go to stderr (``logging`` — no external deps required so the
  script runs unchanged on the Mac Studio training venv).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from itertools import permutations
from pathlib import Path
from typing import Any

# Ensure ``src`` is importable when running from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval.forgetting import ANGLE_THRESHOLD, measure_forgetting_signal  # noqa: E402

logger = logging.getLogger("forgetting_sweep")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pairwise forgetting-angle matrix across a directory of adapters.",
    )
    parser.add_argument(
        "--adapters-dir",
        required=True,
        type=Path,
        help="Root directory whose immediate subdirs each hold adapters.safetensors.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the JSON matrix (stdout is always written).",
    )
    parser.add_argument(
        "--adapter-filename",
        default="adapters.safetensors",
        help="Expected adapter file inside each subdir (default: adapters.safetensors).",
    )
    return parser.parse_args(argv)


def _discover_adapters(root: Path, filename: str) -> list[tuple[str, Path]]:
    """Return ``[(name, path_to_adapter)]`` for every subdir containing ``filename``."""
    if not root.is_dir():
        raise FileNotFoundError(f"adapters dir not found: {root}")
    found: list[tuple[str, Path]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        candidate = child / filename
        if candidate.is_file():
            found.append((child.name, candidate))
    return found


def _sweep(
    adapters: list[tuple[str, Path]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run pairwise sweeps; return ``(pairs, flags)``."""
    pairs: list[dict[str, Any]] = []
    min_mean = float("inf")
    worst: tuple[str, str] | None = None
    any_below = False

    for prior_name, prior_path in adapters:
        for new_name, new_path in adapters:
            if prior_name == new_name:
                continue
            logger.info("pair: %s -> %s", prior_name, new_name)
            report = measure_forgetting_signal(
                prior_adapter_path=prior_path,
                new_adapter_path=new_path,
            )
            mean_angle = report.get("angle_degrees_mean")
            per_module = report.get("angle_degrees_per_module", {})
            gate_status = report.get("gate_status")
            pair_entry: dict[str, Any] = {
                "prior": prior_name,
                "new": new_name,
                "angle_degrees_mean": mean_angle,
                "angle_degrees_per_module": per_module,
                "gate_status": gate_status,
            }
            # Additive: pass through the per-expert breakdown when the
            # underlying adapters exposed MoE structure. Plain LoRA
            # pairs omit the field entirely so the matrix schema stays
            # backward compatible.
            per_expert = report.get("angle_degrees_per_expert")
            if per_expert:
                pair_entry["angle_degrees_per_expert"] = per_expert
            pairs.append(pair_entry)
            if isinstance(mean_angle, (int, float)) and mean_angle == mean_angle:
                if mean_angle < min_mean:
                    min_mean = mean_angle
                    worst = (prior_name, new_name)
                if mean_angle < ANGLE_THRESHOLD:
                    any_below = True

    flags: dict[str, Any] = {
        "any_pair_below_30": any_below,
        "min_mean_angle": min_mean if min_mean != float("inf") else None,
        "worst_pair": list(worst) if worst else None,
        "angle_threshold_degrees": ANGLE_THRESHOLD,
    }
    return pairs, flags


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = _parse_args(argv)

    adapters = _discover_adapters(args.adapters_dir, args.adapter_filename)
    if len(adapters) < 2:
        logger.error(
            "need at least 2 adapters under %s (found %d)",
            args.adapters_dir,
            len(adapters),
        )
        return 2
    names = [name for name, _ in adapters]
    logger.info("found %d adapters: %s", len(adapters), ", ".join(names))

    n_pairs = len(list(permutations(range(len(adapters)), 2)))
    logger.info("running %d ordered pairs", n_pairs)

    pairs, flags = _sweep(adapters)

    payload: dict[str, Any] = {
        "adapters": names,
        "pairs": pairs,
        "flags": flags,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        logger.info("wrote %s", args.output)

    if flags["any_pair_below_30"]:
        logger.warning(
            "forgetting flag: worst pair %s mean %.2f° < %.1f°",
            flags["worst_pair"],
            flags["min_mean_angle"],
            ANGLE_THRESHOLD,
        )
        return 1
    logger.info(
        "all %d pairs pass %.1f° gate (min mean %.2f°, worst pair %s)",
        n_pairs,
        ANGLE_THRESHOLD,
        flags["min_mean_angle"],
        flags["worst_pair"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

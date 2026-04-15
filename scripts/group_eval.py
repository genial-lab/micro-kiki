"""Run group evaluation across all trained stacks.

Usage: uv run scripts/group_eval.py --stacks 20 --output results/group-eval-after-20.json
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def run_group_eval(max_stack: int, output_path: str) -> dict:
    """Run router eval + forgetting check across all stacks up to max_stack.

    NOTE: Actual evaluation requires torch + trained models.
    This script provides the framework; real eval runs on GPU machines.
    """
    configs_dir = Path("configs")
    stack_configs = sorted(configs_dir.glob("stack-*.yaml"))
    active_stacks = [c for c in stack_configs if int(c.stem.split("-")[1]) <= max_stack]

    logger.info("Group eval for %d stacks (up to stack-%02d)", len(active_stacks), max_stack)

    results = {
        "timestamp": datetime.now().isoformat(),
        "max_stack": max_stack,
        "num_stacks_evaluated": len(active_stacks),
        "stacks": [c.stem for c in active_stacks],
        "status": "framework_ready",
        "note": "Actual eval requires trained models on GPU. Run on kxkm-ai or Mac Studio.",
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    logger.info("Group eval report saved to %s", out)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Group stack evaluation")
    parser.add_argument("--stacks", type=int, required=True, help="Evaluate up to stack N")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()
    run_group_eval(args.stacks, args.output)

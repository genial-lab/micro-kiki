"""Orchestration wrapper: run adapter health + forgetting gate after training.

Intended to be invoked by the training pipeline (manual, launchd trigger, or
shell wrapper around ``mlx_lm lora``) immediately after a new adapter is
produced. Chains:

  1. ``validate_adapter_health`` on the new adapter — fail fast if degenerate.
  2. ``measure_forgetting`` between the new adapter and the previous stack in
     the curriculum — fail if the AND-gate (angle < 30° AND winrate_drop
     > 0.03) triggers on any non-ignored module.

Exit codes:
  0 — gate passed (adapter is healthy, forgetting is within bounds)
  1 — adapter is degenerate (all ``lora_B`` below ε)
  2 — forgetting gate failed (rollback recommended)
  3 — configuration/invocation error

The script is deliberately thin: it delegates to the existing entry points
so behaviour stays consistent with operator-run invocations of each piece.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    combined = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, combined


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--new-adapter", type=Path, required=True)
    parser.add_argument(
        "--prior-adapter",
        type=Path,
        help=(
            "Previous curriculum adapter. If omitted, only the health check "
            "runs (no forgetting comparison)."
        ),
    )
    parser.add_argument(
        "--eval-dataset",
        type=Path,
        help="Optional heldout JSONL for win-rate eval (enables full gate).",
    )
    parser.add_argument("--generate-fn-module", default=None)
    parser.add_argument("--winrate-baseline-score", type=float, default=None)
    parser.add_argument("--scorer-module", default=None)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args(argv)

    if not args.new_adapter.is_file():
        print(f"ERROR: new adapter not found: {args.new_adapter}", file=sys.stderr)
        return 3

    py = sys.executable or shutil.which("python3") or "python3"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== step 1/2: adapter health ({args.new_adapter.name}) ===")
    rc, out = _run([py, str(SCRIPTS / "validate_adapter_health.py"), str(args.new_adapter)])
    print(out.rstrip())
    if rc != 0:
        print("FAIL: adapter is degenerate — training path produced a dead adapter.")
        return 1
    print("PASS: adapter health OK.")

    if args.prior_adapter is None:
        print("=== step 2/2: forgetting gate skipped (no --prior-adapter) ===")
        return 0

    if not args.prior_adapter.is_file():
        print(f"ERROR: prior adapter not found: {args.prior_adapter}", file=sys.stderr)
        return 3

    print(f"=== step 2/2: forgetting gate ({args.prior_adapter.name} -> {args.new_adapter.name}) ===")
    out_json = args.output_dir / f"gate-{args.new_adapter.parent.name}.json"
    fg_cmd = [
        py,
        str(SCRIPTS / "measure_forgetting.py"),
        "--prior-adapter",
        str(args.prior_adapter),
        "--new-adapter",
        str(args.new_adapter),
        "--output",
        str(out_json),
    ]
    if args.eval_dataset is not None:
        fg_cmd.extend(["--eval-dataset", str(args.eval_dataset)])
    if args.generate_fn_module:
        fg_cmd.extend(["--generate-fn-module", args.generate_fn_module])
    if args.winrate_baseline_score is not None:
        fg_cmd.extend(["--winrate-baseline-score", str(args.winrate_baseline_score)])
    if args.scorer_module:
        fg_cmd.extend(["--scorer-module", args.scorer_module])

    rc, out = _run(fg_cmd)
    print(out.rstrip())
    if rc == 0:
        print(f"PASS: forgetting gate cleared. Report at {out_json}.")
        return 0
    print(f"FAIL: forgetting gate triggered. See {out_json} for per-module detail.")
    return 2


if __name__ == "__main__":
    sys.exit(main())

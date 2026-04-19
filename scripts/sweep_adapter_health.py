"""Bulk adapter-health matrix: structured JSON companion to ``validate_adapter_health.py``.

Whereas ``scripts/validate_adapter_health.py`` is the fail-fast CLI guardrail
(exit 1 if any adapter is degenerate), this sweep walks a directory and emits
per-adapter structured metrics so we can build audit matrices across fleets
(pre-pivot vs post-pivot, stack-vs-stack) without ad-hoc stdout scraping.

The health check reproduces the same logic as ``validate_adapter_health``:

* parse every ``*.safetensors`` file via ``safetensors.numpy.load_file``,
* collect Frobenius norms for every tensor whose key ends in ``lora_b`` /
  ``lora_B`` (both MLX and HF PEFT spellings, with or without ``.weight``),
* verdict: healthy when at least one ``lora_B`` norm is above epsilon.

JSON schema (one object per adapter):

    {
      "name": "<dir-name or file-stem>",
      "path": "<absolute safetensors path>",
      "healthy": bool,
      "total_lora_b": int,
      "zero_lora_b": int,      # norms <= epsilon
      "max_norm": float,
      "mean_norm": float,
    }

Top-level wrapper:

    {
      "adapters_dir": "<abs path>",
      "epsilon": float,
      "count": int,
      "healthy": int,
      "degenerate": int,
      "adapters": [ ... per-adapter objects ... ],
    }

Only stdlib + numpy + safetensors — same dependency envelope as the validator,
so it runs on both the Mac Studio training venv and kxkm-ai without extras.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

DEFAULT_EPSILON = 1e-6


def _is_lora_b_key(key: str) -> bool:
    lowered = key.lower()
    if lowered.endswith(".weight"):
        lowered = lowered[: -len(".weight")]
    return lowered.endswith("lora_b") or lowered.endswith(".lora_b")


def _adapter_name(path: Path) -> str:
    # For .../<name>/adapters.safetensors, return <name>; else fall back to stem.
    if path.name == "adapters.safetensors" and path.parent.name:
        return path.parent.name
    return path.stem


def audit_adapter(path: Path, epsilon: float) -> dict:
    tensors = load_file(str(path))
    norms: list[float] = []
    for key, arr in tensors.items():
        if not _is_lora_b_key(key):
            continue
        as_f32 = np.asarray(arr, dtype=np.float32)
        norms.append(float(np.linalg.norm(as_f32)))
    total = len(norms)
    zero = sum(1 for v in norms if v <= epsilon)
    max_norm = max(norms) if norms else 0.0
    mean_norm = float(np.mean(norms)) if norms else 0.0
    return {
        "name": _adapter_name(path),
        "path": str(path),
        "healthy": bool(total > 0 and max_norm > epsilon),
        "total_lora_b": total,
        "zero_lora_b": zero,
        "max_norm": max_norm,
        "mean_norm": mean_norm,
    }


def sweep(adapters_dir: Path, epsilon: float) -> dict:
    files = sorted(adapters_dir.rglob("*.safetensors"))
    adapters = [audit_adapter(p, epsilon) for p in files]
    healthy = sum(1 for a in adapters if a["healthy"])
    return {
        "adapters_dir": str(adapters_dir),
        "epsilon": epsilon,
        "count": len(adapters),
        "healthy": healthy,
        "degenerate": len(adapters) - healthy,
        "adapters": adapters,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bulk LoRA-adapter health matrix (JSON).")
    p.add_argument("--adapters-dir", type=Path, required=True,
                   help="Directory to walk recursively for *.safetensors files.")
    p.add_argument("--output", type=Path, required=True,
                   help="Path to write JSON report.")
    p.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON,
                   help=f"Zero-norm threshold (default {DEFAULT_EPSILON:.0e}).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.adapters_dir.is_dir():
        print(f"ERROR: not a directory: {args.adapters_dir}", file=sys.stderr)
        return 2
    report = sweep(args.adapters_dir, args.epsilon)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"[sweep] {report['healthy']}/{report['count']} healthy "
        f"({report['degenerate']} degenerate) -> {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

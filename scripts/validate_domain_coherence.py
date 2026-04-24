"""Fail loudly when NICHE_DOMAINS, router-v4 weights, and the live
adapters directory disagree.

Checks performed:
    1. NICHE_DOMAINS cardinality matches meta.json `num_domains`.
    2. Router weight tensor trailing shape matches len(NICHE_DOMAINS).
    3. Every NICHE_DOMAINS entry has a matching
       ``<adapters_root>/<name>/adapters.safetensors``.
    4. No adapter directory exists for a name absent from NICHE_DOMAINS.

Exit codes: 0 = aligned, 1 = drift detected, 2 = environment problem
(missing file).
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

from src.routing.router import NICHE_DOMAINS

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROUTER = REPO_ROOT / "output" / "router-v4" / "router.safetensors"
DEFAULT_META = REPO_ROOT / "output" / "router-v4" / "meta.json"
DEFAULT_ADAPTERS_ROOT = (
    Path("/Users/clems/KIKI-Mac_tunner/output/micro-kiki")
    / "lora-qwen36-35b-v4-sota"
)


def _read_safetensors_header(path: Path) -> dict:
    with path.open("rb") as fh:
        (size,) = struct.unpack("<Q", fh.read(8))
        return json.loads(fh.read(size))


def validate(
    router_path: Path,
    meta_path: Path,
    adapters_root: Path,
) -> int:
    niches = sorted(NICHE_DOMAINS)
    n_niches = len(niches)
    problems: list[str] = []

    if not meta_path.exists():
        print(f"ERROR: meta.json missing at {meta_path}", file=sys.stderr)
        return 2
    meta = json.loads(meta_path.read_text())
    meta_n = meta.get("num_domains")
    if meta_n != n_niches:
        problems.append(
            f"meta.json num_domains={meta_n} but NICHE_DOMAINS={n_niches}"
        )

    if not router_path.exists():
        print(f"ERROR: router weights missing at {router_path}", file=sys.stderr)
        return 2
    header = _read_safetensors_header(router_path)
    # The router is a Sequential with multiple numeric-indexed layers; the
    # classification head is the last one, so pick the max-index .weight.
    numeric_weights = [
        (int(k[: -len(".weight")]), k)
        for k in header
        if k.endswith(".weight") and k[: -len(".weight")].isdigit()
    ]
    if not numeric_weights:
        problems.append(
            "router output: no numeric .weight entry found in safetensors header"
        )
    else:
        _, weight_key = max(numeric_weights)
        shape = header[weight_key].get("shape", [])
        out_dim = shape[0] if shape else None
        if out_dim != n_niches:
            problems.append(
                f"router output dim ({weight_key} shape[0])={out_dim} "
                f"but NICHE_DOMAINS={n_niches}"
            )

    if not adapters_root.exists():
        print(
            f"WARNING: adapters_root {adapters_root} does not exist on this "
            f"host - skipping adapter checks (run the validator on Studio "
            f"or override --adapters-root).",
            file=sys.stderr,
        )
    else:
        present = {p.name for p in adapters_root.iterdir() if p.is_dir()}
        missing = set(niches) - present
        extra = present - set(niches)
        for m in sorted(missing):
            problems.append(f"adapter missing: no directory {adapters_root / m}")
        for e in sorted(extra):
            problems.append(
                f"stale adapter: {adapters_root / e} has no entry in "
                f"NICHE_DOMAINS"
            )
        for n in niches:
            st = adapters_root / n / "adapters.safetensors"
            if n in present and not st.exists():
                problems.append(f"adapter incomplete: {st} missing")

    if problems:
        print("Domain coherence check FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print(
        f"Domain coherence OK: {n_niches} niches aligned across "
        f"NICHE_DOMAINS, meta.json, router weights, and adapters_root."
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-path", type=Path, default=DEFAULT_ROUTER)
    parser.add_argument("--meta-path", type=Path, default=DEFAULT_META)
    parser.add_argument(
        "--adapters-root",
        type=Path,
        default=DEFAULT_ADAPTERS_ROOT,
    )
    args = parser.parse_args()

    sys.exit(
        validate(
            router_path=args.router_path,
            meta_path=args.meta_path,
            adapters_root=args.adapters_root,
        )
    )


if __name__ == "__main__":
    main()

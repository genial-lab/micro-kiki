#!/usr/bin/env python3
"""End-to-end dream consolidation cycle driver.

Orchestrates one full dream cycle : dump Aeon trace records -> synthesize a
``DreamEpisode`` -> pass to :class:`MicroKikiSubstrate` -> apply REPLAY +
DOWNSCALE + RESTRUCTURE -> save a consolidated adapter snapshot.

Pipeline
--------

1. Load the Aeon trace JSONL (or synthesize a 5-record fake trace for smoke).
2. Load prior + new LoRA adapter safetensors, compute per-key delta.
3. Build a ``DreamEpisode`` from ``kiki_oniric.axioms`` (or fall back to a
   local mock dict when the upstream package is not installed).
4. Construct a :class:`MicroKikiSubstrate`, load it, consolidate the episode
   and snapshot the resulting adapter to ``--output-snapshot``.

Graceful degradation
--------------------

* ``kiki_oniric`` missing   -> write a mock episode to disk, exit 0.
* ``safetensors`` missing   -> fall back to numpy random-stub adapters.
* ``--real-backend`` unresolved (kxkm-ai) -> substrate falls back to stub.
* Any substrate exception  -> exit 2 with a logged traceback.

Exit codes
----------

* ``0`` success (includes ``--dry-run`` and local-mock path)
* ``1`` invalid arguments (missing required adapter, etc.)
* ``2`` substrate consolidate failed
* ``3`` hard dependency missing (e.g. numpy)

Usage
-----

.. code-block:: bash

    # Smoke test — no adapters, dry-run, mock episode to stdout.
    python scripts/run_dream_cycle.py \\
        --new-adapter /dev/null \\
        --output-snapshot /tmp/dream-snap.npz \\
        --dry-run

    # Full cycle with adapter delta + real backend on Studio.
    python scripts/run_dream_cycle.py \\
        --aeon-trace output/aeon/trace-latest.jsonl \\
        --prior-adapter checkpoints/stack-03/adapters.safetensors \\
        --new-adapter   checkpoints/stack-04/adapters.safetensors \\
        --output-snapshot results/dream/stack-04-consolidated.npz \\
        --real-backend /Volumes/spiking/SpikingKiki-V4

See the sibling ``scripts/dream_prototype_demo.py`` for the substrate-only
surface and ``src/dream/substrate.py`` for the substrate implementation.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# Resolve the repo root so ``import src.dream.substrate`` works when the
# script is invoked with its absolute path from anywhere.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

logger = logging.getLogger("run_dream_cycle")


# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------


def _synthesize_trace(n: int = 5) -> list[dict[str, Any]]:
    """Deterministic fake Aeon trace for smoke tests."""
    return [
        {
            "t": float(i),
            "type": "ATLAS_WRITE",
            "content": {"key": f"trace_{i}", "vec_hint": [0.1 * i] * 8},
        }
        for i in range(n)
    ]


def _load_aeon_trace(path: Path | None) -> list[dict[str, Any]]:
    """Load JSONL records; synthesize if missing or empty.

    Each line must be a JSON dict; malformed lines are logged and skipped
    (aligns with the Aeon trace dumper which appends one record per line).
    """
    if path is None or not path.exists() or path.stat().st_size == 0:
        logger.warning("no aeon trace at %s — using 5-record synthetic trace", path)
        return _synthesize_trace()
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                logger.warning("%s:%d malformed JSON skipped: %s", path, lineno, exc)
    if not records:
        logger.warning("aeon trace %s parsed empty — using synthetic", path)
        return _synthesize_trace()
    return records


# ---------------------------------------------------------------------------
# Adapter loading + delta
# ---------------------------------------------------------------------------


def _load_adapter_safetensors(path: Path | None) -> dict[str, Any]:
    """Load a .safetensors adapter into a ``dict[str, ndarray]``.

    Returns an empty dict on ``None`` / missing. Falls back to a numpy
    random-stub adapter when the ``safetensors`` package is unavailable so
    the smoke path stays end-to-end on kxkm-ai even without deps.
    """
    import numpy as np  # hard dep; exit 3 at main if missing

    if path is None:
        return {}
    if str(path) == "/dev/null" or not path.exists():
        logger.warning("adapter %s not found — using numpy zero-stub", path)
        return {"stub.lora_A": np.zeros((8, 4), dtype=np.float32)}

    try:
        from safetensors.numpy import load_file  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("safetensors missing — returning numpy random-stub")
        rng = np.random.default_rng(0)
        return {"stub.lora_A": rng.standard_normal((8, 4)).astype(np.float32)}

    try:
        raw = load_file(str(path))
    except Exception as exc:  # noqa: BLE001 — safetensors may raise various types
        logger.error("failed to load safetensors %s: %s", path, exc)
        return {}
    # load_file returns a dict[str, np.ndarray] already; be explicit for clarity.
    return {k: np.asarray(v) for k, v in raw.items()}


def _compute_delta(
    prior: dict[str, Any], new: dict[str, Any]
) -> dict[str, Any]:
    """Per-key delta = new - prior ; keys only in one side are copied through.

    Shape mismatch on a shared key logs a warning and emits the new tensor
    unchanged (no broadcast — the downstream substrate treats keys opaquely).
    """
    import numpy as np

    delta: dict[str, Any] = {}
    for k, v_new in new.items():
        v_new = np.asarray(v_new)
        v_prior = prior.get(k)
        if v_prior is None:
            delta[k] = v_new
            continue
        v_prior = np.asarray(v_prior)
        if v_prior.shape != v_new.shape:
            logger.warning(
                "delta shape mismatch on %s (prior=%s, new=%s) — keeping new",
                k,
                v_prior.shape,
                v_new.shape,
            )
            delta[k] = v_new
            continue
        delta[k] = (v_new - v_prior).astype(v_new.dtype, copy=False)
    # Keys only in prior -> negative delta (removal).
    for k, v_prior in prior.items():
        if k not in new:
            delta[k] = (-np.asarray(v_prior)).astype(np.asarray(v_prior).dtype, copy=False)
    return delta


# ---------------------------------------------------------------------------
# DreamEpisode construction — real + mock
# ---------------------------------------------------------------------------


def _build_episode(
    records: list[dict[str, Any]], delta_keys: list[str]
) -> tuple[Any, bool]:
    """Return ``(episode, is_real)`` where is_real is True iff kiki_oniric loaded.

    The mock branch emits a plain dict shaped like the upstream
    ``DreamEpisode`` (``episode_id``, ``operation_set``, ``input_slice``,
    ``trigger``) so downstream mock consumers can walk it without guards.
    """
    beta_records = [
        {
            "context": f"aeon:{r.get('type', 'UNKNOWN')}:{i}",
            "outcome": json.dumps(r.get("content", {}))[:256],
            "saillance_score": 1.0 / (1.0 + i),
            "x": r.get("content", {}).get("vec_hint"),
        }
        for i, r in enumerate(records)
    ]
    try:  # lazy — kiki_oniric is an optional dep on kxkm-ai.
        from kiki_oniric.axioms import (  # type: ignore[import-not-found]
            DreamEpisode,
            DreamOperation,
            DreamTrigger,
        )
    except ImportError:
        logger.warning("kiki_oniric missing — emitting local-mock episode dict")
        mock = {
            "episode_id": f"de-local-{int(time.time())}",
            "trigger": "SCHEDULED",
            "operation_set": ("replay", "downscale", "restructure"),
            "input_slice": {
                "beta_records": beta_records,
                "delta_keys": list(delta_keys),
            },
        }
        return mock, False

    ep = DreamEpisode(
        episode_id=f"de-cycle-{int(time.time())}",
        trigger=DreamTrigger.SCHEDULED,
        operation_set=(
            DreamOperation.REPLAY,
            DreamOperation.DOWNSCALE,
            DreamOperation.RESTRUCTURE,
        ),
        input_slice={
            "beta_records": beta_records,
            "delta_keys": list(delta_keys),
        },
    )
    return ep, True


# ---------------------------------------------------------------------------
# Substrate driver
# ---------------------------------------------------------------------------


def _run_substrate(
    episode: Any,
    adapter: dict[str, Any],
    output_snapshot: Path,
    real_backend: str | None,
    new_adapter_path: Path,
) -> tuple[str, Path | None]:
    """Construct MicroKikiSubstrate, consolidate, snapshot.

    Returns ``(status, snapshot_path_or_None)``. ``status`` is one of
    ``"ok"``, ``"consolidate_failed"``, ``"substrate_import_failed"``.
    """
    try:
        from src.dream.substrate import MicroKikiSubstrate  # noqa: WPS433
    except ImportError as exc:
        logger.error("cannot import MicroKikiSubstrate: %s", exc)
        return "substrate_import_failed", None

    # The substrate prototype does not (yet) accept ``real_backend_path``
    # or ``adapter_path`` as constructor kwargs — we honour the contract by
    # setting them on state post-construction and logging the resolution.
    sub = MicroKikiSubstrate(base_model_path=None, seed=0)
    if real_backend:
        resolved = Path(real_backend)
        if resolved.exists():
            logger.info("real_backend resolved: %s", resolved)
            sub.state.base_model = None  # phase-3 would load MLX here
        else:
            logger.warning(
                "real_backend %s unresolved (kxkm-ai?) — stub fallback", real_backend
            )
    sub.state.adapter = dict(adapter)
    # β-record ingest from the episode (dict or dataclass).
    input_slice = (
        episode.get("input_slice")
        if isinstance(episode, dict)
        else getattr(episode, "input_slice", {})
    ) or {}
    beta_records = list(input_slice.get("beta_records", []))
    if beta_records:
        sub.ingest_beta_records(beta_records)
    logger.info(
        "substrate ready (adapter_keys=%d, β_buffer=%d, new_adapter=%s)",
        len(sub.state.adapter),
        len(sub.state.beta_buffer),
        new_adapter_path,
    )

    try:
        sub.consolidate(episode)
    except Exception as exc:  # noqa: BLE001 — surface all substrate errors
        logger.error("consolidate failed: %s\n%s", exc, traceback.format_exc())
        return "consolidate_failed", None

    snap = sub.snapshot(output_snapshot)
    logger.info("snapshot written: %s", snap)
    return "ok", snap


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "--aeon-trace",
        type=Path,
        default=None,
        help="JSONL Aeon trace file; synthesised if missing / empty.",
    )
    parser.add_argument(
        "--prior-adapter",
        type=Path,
        default=None,
        help="Baseline .safetensors LoRA adapter (β-record reference).",
    )
    parser.add_argument(
        "--new-adapter",
        type=Path,
        required=True,
        help="Latest .safetensors LoRA adapter whose delta is consolidated.",
    )
    parser.add_argument(
        "--output-snapshot",
        type=Path,
        required=True,
        help="Destination .npz for the MicroKikiSubstrate snapshot.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the DreamEpisode, print it as JSON, skip substrate.",
    )
    parser.add_argument(
        "--real-backend",
        type=str,
        default=None,
        help="SpikingKiki-V4 backend dir (Mac Studio); stubbed on kxkm-ai.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # --new-adapter is required; reject missing unless it's the /dev/null
    # sentinel used by the dry-run smoke test.
    if str(args.new_adapter) != "/dev/null" and not args.new_adapter.exists():
        logger.error("--new-adapter not found: %s", args.new_adapter)
        return 1

    try:
        import numpy as np  # noqa: F401  — presence check
    except ImportError as exc:
        logger.error("numpy missing — hard dependency: %s", exc)
        return 3

    t0 = time.monotonic()

    # 1. Trace.
    records = _load_aeon_trace(args.aeon_trace)
    logger.info("loaded %d aeon records", len(records))

    # 2. Adapters + delta.
    prior = _load_adapter_safetensors(args.prior_adapter)
    new = _load_adapter_safetensors(args.new_adapter)
    delta = _compute_delta(prior, new)
    logger.info(
        "adapter keys: prior=%d, new=%d, delta=%d",
        len(prior),
        len(new),
        len(delta),
    )

    # 3. Episode.
    episode, is_real = _build_episode(records, list(delta.keys()))

    # 4. Dry-run path — serialise and bail.
    if args.dry_run:
        if is_real:
            # Upstream DreamEpisode is expected to be JSON-serialisable via a
            # ``.to_dict()`` helper; fall back to ``__dict__`` otherwise.
            to_dict = getattr(episode, "to_dict", None)
            payload = to_dict() if callable(to_dict) else getattr(episode, "__dict__", {})
        else:
            payload = episode
        print(json.dumps(payload, indent=2, default=str))
        logger.info("dry-run complete in %.2fs", time.monotonic() - t0)
        return 0

    # 5. Local-mock short-circuit when kiki_oniric is missing — still
    #    writes the episode JSON beside the requested snapshot so the next
    #    run can pick it up, and exits 0 (documented CI behaviour).
    if not is_real:
        mock_out = args.output_snapshot.with_suffix(".mock-episode.json")
        mock_out.parent.mkdir(parents=True, exist_ok=True)
        mock_out.write_text(json.dumps(episode, indent=2, default=str))
        logger.warning(
            "kiki_oniric unavailable — wrote mock episode to %s and exiting 0",
            mock_out,
        )
        print(
            json.dumps(
                {
                    "status": "local_mock",
                    "n_aeon_records": len(records),
                    "n_delta_keys": len(delta),
                    "mock_episode_path": str(mock_out),
                    "time_elapsed_s": round(time.monotonic() - t0, 3),
                }
            )
        )
        return 0

    # 6. Real substrate consolidate + snapshot.
    status, snap_path = _run_substrate(
        episode=episode,
        adapter=delta,
        output_snapshot=args.output_snapshot,
        real_backend=args.real_backend,
        new_adapter_path=args.new_adapter,
    )
    if status == "substrate_import_failed":
        return 3
    if status == "consolidate_failed":
        return 2

    # 7. Summary.
    summary = {
        "status": status,
        "n_aeon_records": len(records),
        "n_delta_keys": len(delta),
        "substrate_status": status,
        "snapshot_path": str(snap_path) if snap_path else None,
        "time_elapsed_s": round(time.monotonic() - t0, 3),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

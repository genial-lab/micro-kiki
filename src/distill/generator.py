"""Distilled dataset generator.

Emits JSONL rows of the form::

    {"prompt": ..., "completion": ..., "teacher_model": ...,
     "domain": ..., "hash": ..., "params": {...}}

Supports resume-from-checkpoint by scanning the target JSONL for existing
``hash`` values and skipping prompts already completed. The teacher is any
object implementing ``TeacherProtocol`` (a single ``complete`` method),
which keeps the generator decoupled from the HTTP client and easy to mock
in tests.

The ``hash`` is the SHA-256 of ``prompt + teacher_model + sorted(params)``
so identical (prompt, model, params) triples collapse to one checkpoint
entry, while a change in sampling params invalidates the cache as
expected.

Design notes
------------
* Retries are done inside the generator (not delegated to the teacher)
  because the teacher may be a thin in-process stub during tests or a
  remote HTTP client in production. We want a single uniform retry
  surface.
* We never raise on per-prompt failure after ``max_retries`` — we record
  the failure in the returned ``stats`` dict and continue. This matches
  the long-running distillation workflow where partial completion is the
  norm.
* ``n_per_prompt`` produces ``n`` independent completions with incremented
  ``sample_idx`` in the hash so resume works when ``n > 1``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol

logger = logging.getLogger(__name__)


class TeacherProtocol(Protocol):
    """Minimal teacher interface consumed by :func:`generate_examples`."""

    model: str

    def complete(self, prompt: str, **params: Any) -> str:
        """Return a single completion string for ``prompt``."""
        ...


@dataclass
class GeneratorConfig:
    """Configuration for :func:`generate_examples`."""

    n_per_prompt: int = 1
    max_retries: int = 3
    retry_backoff_s: float = 0.1
    domain: str = "unknown"
    params: dict[str, Any] = field(default_factory=dict)
    # Fail-soft: if True, a prompt that exhausts retries is skipped and
    # recorded in stats instead of raising.
    fail_soft: bool = True


def hash_record(
    prompt: str,
    teacher_model: str,
    params: dict[str, Any],
    sample_idx: int = 0,
) -> str:
    """Return the SHA-256 checkpoint hash for a (prompt, model, params, idx).

    Parameters are JSON-dumped with ``sort_keys=True`` so ordering does
    not affect the hash.
    """

    payload = {
        "prompt": prompt,
        "teacher_model": teacher_model,
        "params": params,
        "sample_idx": sample_idx,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def load_existing_hashes(jsonl_path: Path) -> set[str]:
    """Scan ``jsonl_path`` and return the set of ``hash`` values already present.

    Missing files yield the empty set; malformed lines are ignored (logged
    at WARNING). This lets a crashed run resume cleanly.
    """

    done: set[str] = set()
    if not jsonl_path.exists():
        return done
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning(
                    "skipping malformed jsonl line %d in %s", lineno, jsonl_path
                )
                continue
            h = row.get("hash")
            if isinstance(h, str):
                done.add(h)
    return done


def _call_with_retry(
    teacher: TeacherProtocol,
    prompt: str,
    params: dict[str, Any],
    max_retries: int,
    backoff_s: float,
) -> str:
    """Invoke ``teacher.complete`` with exponential backoff.

    Raises the last exception after ``max_retries`` failed attempts.
    """

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return teacher.complete(prompt, **params)
        except Exception as exc:  # noqa: BLE001 — teacher may raise anything
            last_exc = exc
            logger.warning(
                "teacher call failed (attempt %d/%d): %s", attempt, max_retries, exc
            )
            if attempt < max_retries:
                time.sleep(backoff_s * (2 ** (attempt - 1)))
    assert last_exc is not None
    raise last_exc


def generate_examples(
    prompts: Iterable[str],
    teacher: TeacherProtocol,
    output_path: Path,
    config: GeneratorConfig | None = None,
) -> dict[str, int]:
    """Generate distilled examples for ``prompts`` and append to ``output_path``.

    Rows are appended as JSONL. Existing rows in ``output_path`` are
    scanned for ``hash`` values and corresponding (prompt, sample_idx)
    pairs are skipped. Returns a stats dict with ``generated``, ``skipped``,
    and ``failed`` counts.

    Parameters
    ----------
    prompts:
        Iterable of prompt strings. Consumed once.
    teacher:
        Object implementing :class:`TeacherProtocol`.
    output_path:
        JSONL file to append to. Parent directories are created.
    config:
        Optional :class:`GeneratorConfig`. Defaults to ``GeneratorConfig()``.
    """

    cfg = config or GeneratorConfig()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_existing_hashes(output_path)
    stats = {"generated": 0, "skipped": 0, "failed": 0}

    with output_path.open("a", encoding="utf-8") as fh:
        for prompt in prompts:
            for sample_idx in range(cfg.n_per_prompt):
                h = hash_record(prompt, teacher.model, cfg.params, sample_idx)
                if h in existing:
                    stats["skipped"] += 1
                    continue
                try:
                    completion = _call_with_retry(
                        teacher,
                        prompt,
                        cfg.params,
                        cfg.max_retries,
                        cfg.retry_backoff_s,
                    )
                except Exception as exc:  # noqa: BLE001
                    stats["failed"] += 1
                    if cfg.fail_soft:
                        logger.error(
                            "giving up on prompt after %d retries: %s",
                            cfg.max_retries,
                            exc,
                        )
                        continue
                    raise
                row = {
                    "prompt": prompt,
                    "completion": completion,
                    "teacher_model": teacher.model,
                    "domain": cfg.domain,
                    "hash": h,
                    "sample_idx": sample_idx,
                    "params": cfg.params,
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                existing.add(h)
                stats["generated"] += 1
    return stats


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield rows from a JSONL file; skip blank / malformed lines."""

    with Path(path).open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                continue

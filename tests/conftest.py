"""Shared pytest fixtures for micro-kiki.

Three fixtures are exposed:

* :func:`tmp_model_dir` — a scratch directory shaped like
  ``models/qwen3.5-4b/bf16/`` with a ``config.json`` stub; used by
  loader / trainer smoke tests once those modules land (story 3, 10).
* :func:`mock_teacher` — an in-process stand-in for
  :class:`src.distill.teacher_client.TeacherClient` with both
  ``complete`` (sync, generator-compatible) and ``generate`` (async,
  client-compatible) surfaces. Responses deterministic, call log
  accessible via ``mock_teacher.call_log``.
* :func:`sample_prompts` — 10 prompts spanning a handful of the 10
  niche domains + base, kept in this file so tests stay hermetic.

These fixtures are kept narrow on purpose: they exist to let every
module have a one-line smoke test without re-deriving the boilerplate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# tmp_model_dir
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_model_dir(tmp_path: Path) -> Path:
    """Return a fake ``models/qwen3.5-4b/bf16`` tree with a ``config.json``.

    The stub config matches the keys the loader is expected to read
    (``model_type``, ``hidden_size``, ``num_hidden_layers``) so a future
    :mod:`src.base.loader` can be smoke-tested without real weights.
    """
    root = tmp_path / "qwen3.5-4b"
    bf16 = root / "bf16"
    bf16.mkdir(parents=True)
    (bf16 / "config.json").write_text(
        '{"model_type": "qwen3", "hidden_size": 2560, "num_hidden_layers": 49, '
        '"num_attention_heads": 40}\n',
        encoding="utf-8",
    )
    # Empty safetensors placeholder — loader smoke tests check existence,
    # not loadability.
    (bf16 / "model.safetensors").write_bytes(b"")
    return root


# ---------------------------------------------------------------------------
# sample_prompts
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_prompts() -> list[str]:
    """Ten prompts spanning chat-fr, code, math, embedded, reasoning."""
    return [
        "Bonjour, peux-tu me presenter micro-kiki en deux phrases?",
        "Explique la difference entre LoRA et QLoRA.",
        "Write a Python function that checks if a string is a palindrome.",
        "In TypeScript, show a generic identity function.",
        "Solve: integrate x^2 * sin(x) from 0 to pi.",
        "What is the time complexity of a merge sort?",
        "Ecris un prompt systeme pour un assistant medical prudent.",
        "Decris le fonctionnement du DMA sur un ESP32-S3.",
        "Liste 3 causes possibles d un kernel panic sur Linux.",
        "Translate to French: 'the router dispatches tokens to experts'.",
    ]


# ---------------------------------------------------------------------------
# mock_teacher
# ---------------------------------------------------------------------------


@dataclass
class FakeTeacher:
    """Dual-surface mock teacher used by smoke + unit tests.

    Exposes ``complete(prompt, **params)`` (sync) for
    :func:`src.distill.generator.generate_examples` and a coroutine
    ``generate(prompt, model, params=None, **_)`` that mimics
    :meth:`src.distill.teacher_client.TeacherClient.generate`. No
    network I/O, deterministic output.
    """

    model: str = "mock-teacher-v0"
    responses: dict[str, str] = field(default_factory=dict)
    call_log: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    echo_prefix: str = "mock::"

    def complete(self, prompt: str, **params: Any) -> str:
        self.call_log.append((prompt, dict(params)))
        return self.responses.get(prompt, f"{self.echo_prefix}{prompt}")

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        params: Any = None,
        *,
        use_cache: bool = True,  # noqa: ARG002 — mirrors real client
    ) -> str:
        self.call_log.append((prompt, {"model": model, "params": params}))
        return self.responses.get(prompt, f"{self.echo_prefix}{prompt}")

    def generate_sync(
        self,
        prompt: str,
        model: str | None = None,
        params: Any = None,
        *,
        use_cache: bool = True,  # noqa: ARG002
    ) -> str:
        # Keep the sync surface too so tests don't need an event loop.
        self.call_log.append((prompt, {"model": model, "params": params}))
        return self.responses.get(prompt, f"{self.echo_prefix}{prompt}")


@pytest.fixture
def mock_teacher() -> FakeTeacher:
    """Return a fresh :class:`FakeTeacher` with no pre-canned responses."""
    return FakeTeacher()

"""Tests for :mod:`src.distill.generator`."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from src.distill.generator import (
    GeneratorConfig,
    generate_examples,
    hash_record,
    iter_jsonl,
    load_existing_hashes,
)


# ---------------------------------------------------------------------------
# Mock teacher
# ---------------------------------------------------------------------------


@dataclass
class MockTeacher:
    """Deterministic mock teacher used by the generator tests."""

    model: str = "mock-teacher-v0"
    responses: dict[str, str] = field(default_factory=dict)
    fail_first_n: int = 0
    call_log: list[str] = field(default_factory=list)
    _failures_left: int = 0

    def __post_init__(self) -> None:
        self._failures_left = self.fail_first_n

    def complete(self, prompt: str, **params: object) -> str:  # noqa: ARG002
        self.call_log.append(prompt)
        if self._failures_left > 0:
            self._failures_left -= 1
            raise RuntimeError("simulated teacher failure")
        return self.responses.get(prompt, f"echo::{prompt}")


# ---------------------------------------------------------------------------
# hash_record
# ---------------------------------------------------------------------------


def test_hash_record_is_deterministic() -> None:
    h1 = hash_record("hello", "m", {"temperature": 0.7, "top_p": 0.9})
    h2 = hash_record("hello", "m", {"top_p": 0.9, "temperature": 0.7})
    assert h1 == h2


def test_hash_record_changes_with_params() -> None:
    h1 = hash_record("hello", "m", {"temperature": 0.7})
    h2 = hash_record("hello", "m", {"temperature": 0.8})
    assert h1 != h2


def test_hash_record_changes_with_sample_idx() -> None:
    h1 = hash_record("hello", "m", {}, sample_idx=0)
    h2 = hash_record("hello", "m", {}, sample_idx=1)
    assert h1 != h2


# ---------------------------------------------------------------------------
# generate_examples — basic flow
# ---------------------------------------------------------------------------


def test_generate_ten_examples(tmp_path: Path) -> None:
    prompts = [f"prompt-{i}" for i in range(10)]
    teacher = MockTeacher()
    out = tmp_path / "chat-fr.jsonl"

    stats = generate_examples(
        prompts, teacher, out, GeneratorConfig(domain="chat-fr")
    )

    assert stats == {"generated": 10, "skipped": 0, "failed": 0}
    rows = list(iter_jsonl(out))
    assert len(rows) == 10
    for i, row in enumerate(rows):
        assert row["prompt"] == f"prompt-{i}"
        assert row["completion"] == f"echo::prompt-{i}"
        assert row["teacher_model"] == "mock-teacher-v0"
        assert row["domain"] == "chat-fr"
        assert isinstance(row["hash"], str) and len(row["hash"]) == 64


def test_output_is_valid_jsonl(tmp_path: Path) -> None:
    prompts = ["a", "b"]
    teacher = MockTeacher()
    out = tmp_path / "out.jsonl"
    generate_examples(prompts, teacher, out)
    for line in out.read_text(encoding="utf-8").splitlines():
        # Must parse as JSON; no trailing commas or other jsonl mistakes.
        json.loads(line)


# ---------------------------------------------------------------------------
# Resume from checkpoint
# ---------------------------------------------------------------------------


def test_resume_skips_already_done(tmp_path: Path) -> None:
    out = tmp_path / "chat-fr.jsonl"
    teacher = MockTeacher()

    # First pass: 5 prompts.
    generate_examples(
        [f"p-{i}" for i in range(5)], teacher, out, GeneratorConfig()
    )
    assert len(teacher.call_log) == 5

    # Second pass: first 3 overlap with existing, last 4 are new.
    teacher2 = MockTeacher()
    stats = generate_examples(
        [f"p-{i}" for i in range(7)], teacher2, out, GeneratorConfig()
    )
    assert stats["skipped"] == 5
    assert stats["generated"] == 2
    # Teacher only called for the 2 new prompts.
    assert teacher2.call_log == ["p-5", "p-6"]
    # Total rows after resume.
    assert len(list(iter_jsonl(out))) == 7


def test_load_existing_hashes_on_missing_file(tmp_path: Path) -> None:
    assert load_existing_hashes(tmp_path / "does-not-exist.jsonl") == set()


def test_load_existing_hashes_ignores_malformed(tmp_path: Path) -> None:
    p = tmp_path / "mixed.jsonl"
    p.write_text(
        '{"hash": "abc"}\n'
        "not-json-at-all\n"
        '{"hash": "def"}\n'
        "\n",
        encoding="utf-8",
    )
    assert load_existing_hashes(p) == {"abc", "def"}


# ---------------------------------------------------------------------------
# n_per_prompt and retry
# ---------------------------------------------------------------------------


def test_n_per_prompt_emits_multiple_samples(tmp_path: Path) -> None:
    out = tmp_path / "multi.jsonl"
    teacher = MockTeacher()
    stats = generate_examples(
        ["only-one"], teacher, out, GeneratorConfig(n_per_prompt=3)
    )
    assert stats["generated"] == 3
    rows = list(iter_jsonl(out))
    assert [r["sample_idx"] for r in rows] == [0, 1, 2]
    # Each sample has a distinct hash.
    assert len({r["hash"] for r in rows}) == 3


def test_retry_then_succeeds(tmp_path: Path) -> None:
    teacher = MockTeacher(fail_first_n=2)
    out = tmp_path / "retry.jsonl"
    stats = generate_examples(
        ["p"],
        teacher,
        out,
        GeneratorConfig(max_retries=3, retry_backoff_s=0.0),
    )
    assert stats["generated"] == 1
    # 2 failures + 1 success == 3 calls.
    assert len(teacher.call_log) == 3


def test_retry_exhausted_is_fail_soft(tmp_path: Path) -> None:
    teacher = MockTeacher(fail_first_n=10)
    out = tmp_path / "fail.jsonl"
    stats = generate_examples(
        ["p"],
        teacher,
        out,
        GeneratorConfig(max_retries=2, retry_backoff_s=0.0, fail_soft=True),
    )
    assert stats == {"generated": 0, "skipped": 0, "failed": 1}
    assert not out.exists() or out.read_text() == ""


def test_retry_exhausted_raises_when_not_fail_soft(tmp_path: Path) -> None:
    teacher = MockTeacher(fail_first_n=10)
    out = tmp_path / "fail.jsonl"
    with pytest.raises(RuntimeError):
        generate_examples(
            ["p"],
            teacher,
            out,
            GeneratorConfig(
                max_retries=2, retry_backoff_s=0.0, fail_soft=False
            ),
        )

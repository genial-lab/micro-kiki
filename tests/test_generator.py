from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock
from src.distill.generator import generate_examples, load_existing_hashes


@pytest.fixture
def mock_teacher():
    client = AsyncMock()
    client.generate.return_value = "Réponse du teacher."
    return client


class TestGenerator:
    @pytest.mark.asyncio
    async def test_generates_correct_count(self, mock_teacher, tmp_path):
        output = tmp_path / "output.jsonl"
        await generate_examples(["p1", "p2", "p3"], mock_teacher, "test", "chat-fr", output)
        assert len(output.read_text().strip().split("\n")) == 3

    @pytest.mark.asyncio
    async def test_output_format(self, mock_teacher, tmp_path):
        output = tmp_path / "output.jsonl"
        await generate_examples(["test"], mock_teacher, "mistral", "chat-fr", output)
        line = json.loads(output.read_text().strip())
        for key in ("prompt", "completion", "teacher_model", "domain", "hash"):
            assert key in line

    @pytest.mark.asyncio
    async def test_resume_skips_existing(self, tmp_path):
        output = tmp_path / "output.jsonl"
        output.write_text(json.dumps({"prompt": "p", "completion": "c", "teacher_model": "t", "domain": "d", "hash": "abc"}) + "\n")
        assert "abc" in load_existing_hashes(output)

    @pytest.mark.asyncio
    async def test_n_per_prompt(self, tmp_path):
        call_count = 0
        teacher = AsyncMock()

        async def varying_response(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return f"Response variant {call_count}"

        teacher.generate.side_effect = varying_response
        output = tmp_path / "output.jsonl"
        await generate_examples(["p1"], teacher, "test", "chat-fr", output, n_per_prompt=3)
        assert len(output.read_text().strip().split("\n")) == 3

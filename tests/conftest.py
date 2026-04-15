from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock


@pytest.fixture
def tmp_model_dir(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen3.5"}))
    return model_dir


@pytest.fixture
def tmp_stacks_dir(tmp_path):
    stacks = tmp_path / "stacks"
    stacks.mkdir()
    (stacks / "stack-01-chat-fr").mkdir()
    return stacks


@pytest.fixture
def mock_teacher():
    client = AsyncMock()
    client.generate.return_value = "Ceci est une réponse du teacher."
    return client


@pytest.fixture
def sample_prompts():
    return [
        "Explique le fonctionnement d'un condensateur.",
        "Écris une fonction Python qui trie une liste.",
        "Quels sont les avantages de l'architecture MoE?",
        "Décris le protocole I2C en 3 phrases.",
        "Comment fonctionne un MOSFET?",
    ]

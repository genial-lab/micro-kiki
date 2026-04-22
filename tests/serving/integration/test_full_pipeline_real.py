"""Real-MLX integration canaries.

Run on Studio (M3 Ultra 512 GB) only. Each test boots the full pipeline
(FastAPI app + ``mlx_lm`` base model + optional LoRA adapter) and fires a
single ``/v1/chat/completions`` request. Long-running: the first test
pays the ~19 GB base-model cold-load cost; the second reuses the loaded
module (pytest module-scoped fixture).

These tests are the Path B V1.0 acceptance gate: they prove the
orchestrator can degrade gracefully when 4/5 subsystems are stubbed out
and still return a non-empty, coherent 200 response powered by real MLX
inference.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.serving.full_pipeline_server import FullPipelineConfig, make_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    cfg = FullPipelineConfig.defaults()
    cfg.negotiator_k = 1  # single candidate keeps runtime short
    cfg.max_queue_depth = 1
    return TestClient(make_app(cfg))


def test_niche_canary_chat_fr(client: TestClient) -> None:
    """Niche mode bypasses MetaRouter and forces the chat-fr adapter.

    Expected flow:
        recall stub raises -> recalled=[]
        alias.mode == "niche" -> adapters = ["chat-fr"]
        runtime.apply(["chat-fr"])  # real load_adapters on MLX model
        runtime.generate(...)       # 1 candidate (negotiator_k=1)
        negotiator stub raises -> winner = candidates[0]
        antibias stub raises  -> final_text = winner
        write stub raises      -> logged, non-blocking
        -> 200 with non-empty content
    """
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "kiki-niche-chat-fr",
            "messages": [
                {"role": "user", "content": "Bonjour en une phrase."}
            ],
            "max_tokens": 24,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    content = body["choices"][0]["message"]["content"]
    assert content.strip(), "empty content"
    print(f"[canary chat-fr] {content[:200]}")


def test_meta_canary_coding(client: TestClient) -> None:
    """Meta mode invokes MetaRouter.route; stub raises, pipeline degrades to
    base-only inference (adapters=[]) but still returns a valid 200.
    """
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "kiki-meta-coding",
            "messages": [
                {
                    "role": "user",
                    "content": "Return a Python hello world in one line.",
                }
            ],
            "max_tokens": 24,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    content = body["choices"][0]["message"]["content"]
    assert content.strip(), "empty content"
    print(f"[canary meta-coding] {content[:200]}")

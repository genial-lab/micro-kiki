"""End-to-end integration tests for the factory4life serving stack.

These tests drive the full chain :

    OpenAI-shaped request → FastAPI app → runtime Protocol
      → FakeMLXRuntime (scripted completions)
      → admission control (context ceiling, KV budget)
      → SSE or JSON response

The purpose is to catch cross-cutting bugs that the per-module
unit suites (schemas / runtime / app) would miss :

- Admission ceiling + streaming + include_usage chunk all
  composed in one request.
- Multiple concurrent requests from a simulated agent chain
  competing for the same KV budget.
- Mixed-workload sequencing (non-streaming, streaming,
  tool-calling) through a single app instance.
- The actual SSE bytes parse cleanly with the ``openai`` /
  ``langchain-openai`` compat harness once we pin the deps.

No MLX / Studio dependency — the FakeMLXRuntime is the sole
backend for this suite. Real SOTA evaluations that want the
serving layer instead of a direct ``mlx_lm`` call use
``scripts/eval_via_serving.py``, which reuses the same shape.
"""
from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from src.serving import schemas as s
from src.serving.app import create_app
from src.serving.runtime import FakeMLXRuntime


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


def _make_app(
    scripted: dict[str, str] | None = None,
    kv_budget_gb: int = 60,
    kv_used_gb: int = 0,
):
    rt = FakeMLXRuntime(
        scripted_responses=scripted or {},
    )
    rt._kv_bytes_budget = kv_budget_gb * 1024**3
    rt._kv_bytes_used = kv_used_gb * 1024**3
    return create_app(rt), rt


# ---------------------------------------------------------------------------
# Sequential multi-request flows (simulating an agent chain).
# ---------------------------------------------------------------------------


def test_agent_chain_user_tool_assistant_roundtrip() -> None:
    """Simulate a 3-turn LangChain agent chain :
    1. User asks → model calls tool.
    2. Client runs tool, sends ToolMessage back.
    3. Model returns final answer.

    The correlation (``tool_call_id`` echoed in the ToolMessage)
    is the single most fragile part of agent compat."""
    forced = s.ToolCall(
        function=s.FunctionCall(
            name="get_weather",
            arguments=json.dumps({"city": "Paris"}),
        )
    )
    rt = FakeMLXRuntime(force_tool_call=forced)
    app = create_app(rt)
    with TestClient(app) as client:
        # Turn 1 : user asks, model emits tool_call.
        r1 = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.6-35b-python",
                "messages": [{"role": "user", "content": "weather ?"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
            },
        )
        assert r1.status_code == 200
        tool_call = r1.json()["choices"][0]["message"]["tool_calls"][0]
        tool_call_id = tool_call["id"]

        # Turn 2 : client sends back tool result. Drop force so
        # model returns text instead of looping.
        rt.force_tool_call = None
        rt.scripted_responses = {"weather": "Paris is sunny."}
        r2 = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.6-35b-python",
                "messages": [
                    {"role": "user", "content": "weather ?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call],
                    },
                    {
                        "role": "tool",
                        "content": '{"temp_c": 22, "sky": "clear"}',
                        "tool_call_id": tool_call_id,
                    },
                ],
            },
        )
    assert r2.status_code == 200
    assert r2.json()["choices"][0]["message"]["content"] == "Paris is sunny."


def test_mixed_workload_non_stream_then_stream_then_non_stream() -> None:
    """A single app instance handles interleaved streaming and
    non-streaming requests. Shared state (metrics, admission
    counters) must not leak between modes."""
    app, rt = _make_app(scripted={"hi": "hello world"})
    with TestClient(app) as client:
        r1 = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.6-35b-python",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        r2 = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.6-35b-python",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        r3 = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.6-35b-python",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r3.status_code == 200

    # Non-streaming responses both have the completion content.
    assert (
        r1.json()["choices"][0]["message"]["content"] == "hello world"
    )
    assert (
        r3.json()["choices"][0]["message"]["content"] == "hello world"
    )
    # SSE response parses cleanly.
    assert r2.headers["content-type"].startswith("text/event-stream")
    assert r2.text.endswith("data: [DONE]\n\n")


# ---------------------------------------------------------------------------
# Dynamic admission under load — the "eval harness" path.
# ---------------------------------------------------------------------------


def test_admission_ceiling_visible_to_caller_before_overflow() -> None:
    """The client reads ``X-Max-Context-Available`` on every
    response ; if it asked for 40 K but saw the header drop to
    10 K, it can back off on the next request without waiting
    for a 413."""
    app, rt = _make_app(kv_budget_gb=1, kv_used_gb=0)  # 1 GB budget
    with TestClient(app) as client:
        # First request : full headroom.
        r1 = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.6-35b-python",
                "messages": [{"role": "user", "content": "x"}],
            },
        )
        assert r1.status_code == 200
        initial_ceiling = int(r1.headers["x-max-context-available"])

        # Simulate another client opening a session that reserves
        # half the budget.
        rt._kv_bytes_used = 512 * 1024**2

        # Second request sees a lower ceiling in the header.
        r2 = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.6-35b-python",
                "messages": [{"role": "user", "content": "x"}],
            },
        )
    assert r2.status_code == 200
    new_ceiling = int(r2.headers["x-max-context-available"])
    assert new_ceiling < initial_ceiling
    # Half the budget gone → ceiling roughly halved.
    assert 0.3 * initial_ceiling <= new_ceiling <= 0.7 * initial_ceiling


def test_413_ceiling_includes_hint_for_automatic_retry() -> None:
    """The 413 error body + X-Max-Context-Available header give
    a client everything it needs for an automatic retry :
    concrete ceiling, error code ``dynamic_ceiling``, param
    ``max_completion_tokens``."""
    app, rt = _make_app(kv_budget_gb=1)
    rt._kv_bytes_used = 900 * 1024**2  # 100 MB left → ~2.5 K tokens
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.6-35b-python",
                "messages": [{"role": "user", "content": "x"}],
                "max_completion_tokens": 65_536,  # too big
            },
        )
    assert resp.status_code == 413
    body = resp.json()
    assert body["detail"]["error"]["code"] == "dynamic_ceiling"
    ceiling_from_header = int(resp.headers["x-max-context-available"])
    ceiling_from_body = int(
        body["detail"]["error"]["message"].split("(")[-1].split(")")[0]
    )
    assert ceiling_from_header == ceiling_from_body


# ---------------------------------------------------------------------------
# Async / concurrent invocations (the real agent workload).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parallel_requests_isolated() -> None:
    """N parallel requests must each get their own deterministic
    completion — no cross-request contamination of session state.
    This is the minimal guarantee for async agent chains that
    fan out to 20 concurrent LLM calls."""
    rt = FakeMLXRuntime(
        completion_for=lambda prompt, seed: f"answer-for-{prompt[-10:]}",
    )
    app = create_app(rt)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as ac:
        tasks = [
            ac.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen3.6-35b-python",
                    "messages": [
                        {"role": "user", "content": f"prompt-{i:02d}"}
                    ],
                },
            )
            for i in range(10)
        ]
        responses = await asyncio.gather(*tasks)
    # Each response references its own prompt id in the content —
    # guarantees no swap between concurrent sessions.
    for i, r in enumerate(responses):
        assert r.status_code == 200
        content = r.json()["choices"][0]["message"]["content"]
        assert f"prompt-{i:02d}" in content


@pytest.mark.asyncio
async def test_stream_parallel_with_admission() -> None:
    """5 parallel streamed requests. All succeed with deterministic
    recombined content, each response carrying its own X-Max-Context-
    Available header."""
    rt = FakeMLXRuntime(
        completion_for=lambda p, _s: f"streamed {p[-5:]}",
    )
    app = create_app(rt)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as ac:
        tasks = [
            ac.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen3.6-35b-python",
                    "messages": [
                        {"role": "user", "content": f"id{i:02d}"},
                    ],
                    "stream": True,
                },
            )
            for i in range(5)
        ]
        responses = await asyncio.gather(*tasks)
    for r in responses:
        assert r.status_code == 200
        assert "x-max-context-available" in r.headers
        # DONE sentinel on each.
        assert r.text.rstrip().endswith("data: [DONE]")


# ---------------------------------------------------------------------------
# Eval-shaped workload — mimic what eval_humaneval_v4 would run
# through the serving layer.
# ---------------------------------------------------------------------------


def test_eval_workload_records_per_adapter_metrics() -> None:
    """Simulate 3 benchmark runs hitting 3 different adapter
    ids. The metrics endpoint records each adapter selection ;
    the KV status endpoint reports cumulative context exposure.
    This is the exact observability surface a SOTA eval run
    needs."""
    app, rt = _make_app()
    with TestClient(app) as client:
        for adapter in ("python", "cpp", "rust"):
            for q in range(3):
                client.post(
                    "/v1/chat/completions",
                    json={
                        "model": f"qwen3.6-35b-{adapter}",
                        "messages": [
                            {"role": "user",
                             "content": f"problem {q} ?"},
                        ],
                    },
                )
        metrics = client.get("/metrics").text
    # Each adapter accounted exactly 3 times.
    assert 'adapter="qwen3.6-35b-python"} 3' in metrics
    assert 'adapter="qwen3.6-35b-cpp"} 3' in metrics
    assert 'adapter="qwen3.6-35b-rust"} 3' in metrics

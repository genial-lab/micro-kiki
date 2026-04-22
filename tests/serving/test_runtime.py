"""Unit tests for :mod:`src.serving.runtime`.

Exercise :class:`FakeMLXRuntime` end-to-end (non-streaming +
streaming) and cover the helpers that every backend shares
(``stable_seed``, ``resolve_adapter_name``, ``_flatten_prompt``,
``_split_arguments``).

No MLX imports — the runtime module is testable on a bare-bones
kxkm-ai (Linux, no mlx wheel). Any attempt to import MLX here
would violate the T5 mockability contract.
"""
from __future__ import annotations

import json

import pytest

from src.serving import runtime as r
from src.serving import schemas as s


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _req(
    *,
    model: str = "qwen3.6-35b-python",
    user: str = "hello",
    tools: list[dict] | None = None,
    tool_choice=None,
    stream: bool = False,
    include_usage: bool = False,
    seed: int | None = None,
) -> s.ChatCompletionRequest:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user}],
        "stream": stream,
    }
    if tools is not None:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    if include_usage:
        payload["stream_options"] = {"include_usage": True}
    if seed is not None:
        payload["seed"] = seed
    return s.ChatCompletionRequest.model_validate(payload)


# ---------------------------------------------------------------------------
# Pure helpers.
# ---------------------------------------------------------------------------


class TestStableSeed:
    def test_same_input_same_output_cross_invocation(self) -> None:
        a = r.stable_seed(42, "hello")
        b = r.stable_seed(42, "hello")
        assert a == b

    def test_different_prompt_different_seed(self) -> None:
        a = r.stable_seed(42, "hello")
        b = r.stable_seed(42, "world")
        assert a != b

    def test_no_seed_still_deterministic(self) -> None:
        a = r.stable_seed(None, "hello")
        b = r.stable_seed(None, "hello")
        assert a == b

    def test_seed_fits_in_int32(self) -> None:
        """numpy RNGs take uint32 ; 8-hex-char cap keeps us under."""
        value = r.stable_seed(42, "x" * 1000)
        assert 0 <= value < 2**32


class TestResolveAdapterName:
    def test_standard_prefix_strips(self) -> None:
        assert r.resolve_adapter_name("qwen3.6-35b-python") == "python"
        assert r.resolve_adapter_name("qwen3.6-35b-cpp") == "cpp"

    def test_bare_base_returns_none(self) -> None:
        assert r.resolve_adapter_name("base") is None
        assert r.resolve_adapter_name("qwen3.6-35b") is None
        assert r.resolve_adapter_name("qwen3.6-35b-a3b") is None

    def test_auto_alias_sentinel(self) -> None:
        assert r.resolve_adapter_name("qwen3.6-35b-auto") == "__router__"

    def test_user_alias_map(self) -> None:
        aliases = {"code": "qwen3.6-35b-python", "chat": "qwen3.6-35b-chat-fr"}
        assert r.resolve_adapter_name("code", aliases=aliases) == "python"
        assert r.resolve_adapter_name("chat", aliases=aliases) == "chat-fr"


class TestFlattenPrompt:
    def test_multimodal_user_content_keeps_text_drops_image(self) -> None:
        req = s.ChatCompletionRequest.model_validate(
            {
                "model": "x",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "describe this"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:..."},
                            },
                        ],
                    }
                ],
            }
        )
        out = r._flatten_prompt(req.messages)
        assert "describe this" in out
        assert "image_url" not in out
        assert "data:" not in out

    def test_tool_flow_roundtrips_in_prompt(self) -> None:
        req = s.ChatCompletionRequest.model_validate(
            {
                "model": "x",
                "messages": [
                    {"role": "user", "content": "q"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_" + "a" * 20,
                                "type": "function",
                                "function": {
                                    "name": "f",
                                    "arguments": '{"k": 1}',
                                },
                                "index": 0,
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "content": "result",
                        "tool_call_id": "call_" + "a" * 20,
                    },
                ],
            }
        )
        out = r._flatten_prompt(req.messages)
        assert "[user] q" in out
        assert "[tool_call:f]" in out
        assert "[tool_result:call_aaaaa" in out
        assert "result" in out


class TestSplitArguments:
    def test_preserves_concatenation(self) -> None:
        args = '{"a":1,"b":[2,3],"c":"hello"}'
        parts = r._split_arguments(args, n=4)
        assert "".join(parts) == args
        assert len(parts) == 4

    def test_short_string_returned_verbatim(self) -> None:
        parts = r._split_arguments("{}", n=4)
        assert parts == ["{}"]

    def test_n_one_returns_single(self) -> None:
        parts = r._split_arguments("abcdef", n=1)
        assert parts == ["abcdef"]


# ---------------------------------------------------------------------------
# FakeMLXRuntime — non-streaming.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fake_runtime_generate_returns_deterministic_content() -> None:
    rt = r.FakeMLXRuntime()
    req = _req(user="hello world")
    r1 = await rt.generate(req)
    r2 = await rt.generate(req)
    # Same request → same content (both runs use stable_seed).
    assert r1.choices[0].message.content == r2.choices[0].message.content


@pytest.mark.asyncio
async def test_fake_runtime_generate_response_shape() -> None:
    rt = r.FakeMLXRuntime()
    req = _req(user="hello")
    resp = await rt.generate(req)
    assert resp.id.startswith("chatcmpl-")
    assert resp.object == "chat.completion"
    assert resp.choices[0].finish_reason == "stop"
    assert resp.choices[0].message.content is not None
    assert resp.choices[0].message.tool_calls is None
    assert resp.usage is not None
    assert resp.usage.total_tokens == (
        resp.usage.prompt_tokens + resp.usage.completion_tokens
    )


@pytest.mark.asyncio
async def test_fake_runtime_scripted_response_hit() -> None:
    """Scripted responses override the deterministic fallback."""
    rt = r.FakeMLXRuntime(
        scripted_responses={"weather": "It is sunny."},
    )
    req = _req(user="what's the weather in Paris ?")
    resp = await rt.generate(req)
    assert resp.choices[0].message.content == "It is sunny."


@pytest.mark.asyncio
async def test_fake_runtime_custom_completion_fn() -> None:
    rt = r.FakeMLXRuntime(
        completion_for=lambda prompt, seed: f"prompt-length-{len(prompt)}"
    )
    req = _req(user="abc")
    resp = await rt.generate(req)
    assert resp.choices[0].message.content.startswith("prompt-length-")


# ---------------------------------------------------------------------------
# FakeMLXRuntime — tool calling.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fake_runtime_tool_call_when_forced_and_tool_declared() -> None:
    forced = s.ToolCall(
        function=s.FunctionCall(
            name="get_weather",
            arguments=json.dumps({"city": "Paris"}),
        )
    )
    rt = r.FakeMLXRuntime(force_tool_call=forced)
    req = _req(
        user="weather ?",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                },
            }
        ],
    )
    resp = await rt.generate(req)
    msg = resp.choices[0].message
    assert msg.content is None  # Silent-killer #1 guard.
    assert msg.tool_calls is not None
    assert msg.tool_calls[0].function.name == "get_weather"
    assert resp.choices[0].finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_fake_runtime_forced_tool_ignored_when_not_declared() -> None:
    """If the forced tool isn't in request.tools, we fall back to
    plain text — simulating a well-behaved real runtime."""
    forced = s.ToolCall(
        function=s.FunctionCall(name="undeclared_tool", arguments="{}")
    )
    rt = r.FakeMLXRuntime(force_tool_call=forced)
    req = _req(user="hi")  # no tools
    resp = await rt.generate(req)
    assert resp.choices[0].message.tool_calls is None


# ---------------------------------------------------------------------------
# FakeMLXRuntime — streaming.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fake_runtime_stream_text_emits_role_then_tokens() -> None:
    rt = r.FakeMLXRuntime(scripted_responses={"hello": "one two three"})
    req = _req(user="hello world", stream=True)
    chunks = [c async for c in rt.generate_stream(req)]
    # First chunk is role marker.
    assert chunks[0].choices[0].delta.role == "assistant"
    assert chunks[0].choices[0].delta.content is None
    # Middle chunks carry content, no finish_reason.
    middle = chunks[1:-1]
    assert all(c.choices[0].finish_reason is None for c in middle)
    # Final chunk has finish_reason="stop" + empty delta.
    final = chunks[-1]
    assert final.choices[0].finish_reason == "stop"
    assert final.choices[0].delta.content is None
    # Concatenation of content deltas reproduces the script.
    recomposed = "".join(
        c.choices[0].delta.content
        for c in middle
        if c.choices[0].delta.content is not None
    )
    assert recomposed.strip() == "one two three"


@pytest.mark.asyncio
async def test_fake_runtime_stream_tool_call_deltas_concatenate() -> None:
    """LangChain concatenates ``tool_calls[].function.arguments``
    deltas by ``index``. The fake runtime splits the arguments
    into 4 chunks so this test exercises the accumulator
    behaviour a real test harness will see."""
    forced = s.ToolCall(
        function=s.FunctionCall(
            name="emit_json",
            arguments='{"key":"value","n":42}',
        )
    )
    rt = r.FakeMLXRuntime(force_tool_call=forced)
    req = _req(
        user="emit",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "emit_json",
                    "parameters": {"type": "object"},
                },
            }
        ],
        stream=True,
    )
    chunks = [c async for c in rt.generate_stream(req)]
    tool_chunks = [
        c
        for c in chunks
        if c.choices
        and c.choices[0].delta.tool_calls
    ]
    # 4 arg-split chunks (verify deterministic split).
    assert len(tool_chunks) == 4
    # Recompose arguments by index=0.
    recomposed = "".join(
        tc.function.arguments
        for c in tool_chunks
        for tc in (c.choices[0].delta.tool_calls or [])
        if tc.index == 0 and tc.function is not None
    )
    assert recomposed == '{"key":"value","n":42}'
    # First tool chunk carries id + type, later chunks do not.
    first = tool_chunks[0].choices[0].delta.tool_calls[0]
    assert first.id is not None
    assert first.type == "function"
    later = tool_chunks[1].choices[0].delta.tool_calls[0]
    assert later.id is None
    assert later.type is None
    # Final chunk has finish_reason="tool_calls".
    assert chunks[-1].choices[0].finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_fake_runtime_stream_include_usage_emits_final_usage_chunk() -> None:
    """LangChain requires a final chunk with choices=[] + usage
    populated when include_usage=True."""
    rt = r.FakeMLXRuntime(scripted_responses={"x": "a b c d"})
    req = _req(user="x", stream=True, include_usage=True)
    chunks = [c async for c in rt.generate_stream(req)]
    usage_chunk = chunks[-1]
    assert usage_chunk.choices == []
    assert usage_chunk.usage is not None
    assert usage_chunk.usage.total_tokens > 0


@pytest.mark.asyncio
async def test_fake_runtime_stream_no_usage_when_not_requested() -> None:
    rt = r.FakeMLXRuntime()
    req = _req(user="x", stream=True, include_usage=False)
    chunks = [c async for c in rt.generate_stream(req)]
    # No empty-choices chunk at the end — final chunk has the
    # finish_reason, not usage.
    assert chunks[-1].choices != []


# ---------------------------------------------------------------------------
# Runtime Protocol conformance.
# ---------------------------------------------------------------------------


def test_fake_runtime_satisfies_protocol() -> None:
    """Structural subtyping : FakeMLXRuntime must be a
    :class:`MLXRuntime` via ``isinstance`` (Protocol is
    ``runtime_checkable``)."""
    rt = r.FakeMLXRuntime()
    assert isinstance(rt, r.MLXRuntime)


def test_health_and_list_adapters_shape() -> None:
    rt = r.FakeMLXRuntime(adapters=["a", "b", "c"])
    assert rt.list_adapters() == ["a", "b", "c"]
    health = rt.health()
    assert health["runtime"] == "fake"
    assert health["adapters_warm"] == 3
    assert "uptime_s" in health
    # 1M context is the post-YaRN ceiling the backend advertises.
    assert health["max_context_tokens"] == 1_048_576


def test_kv_stats_default_values() -> None:
    rt = r.FakeMLXRuntime()
    stats = rt.kv_stats()
    # Sanity : all mandatory keys present, numbers coherent.
    assert stats["max_context_tokens"] == 1_048_576
    assert stats["kv_bytes_per_token"] == 40 * 1024
    assert stats["sessions_active"] == 0
    assert stats["kv_bytes_used"] == 0
    assert stats["kv_bytes_free"] == stats["kv_bytes_budget"]
    assert stats["prefix_cache_entries"] == 0
    assert stats["prefix_cache_hit_rate"] == 0.0


def test_kv_stats_hit_rate_computes_correctly() -> None:
    rt = r.FakeMLXRuntime()
    rt._prefix_cache_hits = 7
    rt._prefix_cache_lookups = 10
    stats = rt.kv_stats()
    assert stats["prefix_cache_hit_rate"] == 0.7


def test_kv_stats_percentiles_handle_small_samples() -> None:
    """p50 / p99 must not crash on a 1-element or empty
    observed-context list (common at startup)."""
    rt = r.FakeMLXRuntime()
    # Empty.
    stats_empty = rt.kv_stats()
    assert stats_empty["context_tokens_p50"] == 0
    assert stats_empty["context_tokens_p99"] == 0
    # Single sample.
    rt._context_tokens_observed = [2048]
    stats_one = rt.kv_stats()
    assert stats_one["context_tokens_p50"] == 2048
    assert stats_one["context_tokens_p99"] == 2048


def test_kv_stats_supports_1m_context_advertisement() -> None:
    """Operators need to lift the cap per deploy (YaRN config).
    The runtime advertises whatever its ``max_context_tokens``
    attribute says — tests assert the max can go to 1 M."""
    rt = r.FakeMLXRuntime(max_context_tokens=1_048_576)
    assert rt.kv_stats()["max_context_tokens"] == 1_048_576
    rt.max_context_tokens = 262_144  # native Qwen3.6 ceiling
    assert rt.kv_stats()["max_context_tokens"] == 262_144


# ---------------------------------------------------------------------------
# Error types.
# ---------------------------------------------------------------------------


def test_runtime_errors_serialize_to_openai_shape() -> None:
    err = r.AdapterNotFound("adapter 'xxx' not in registry")
    payload = err.to_error(param="model").model_dump()
    assert payload["error"]["type"] == "adapter_not_found"
    assert payload["error"]["message"] == "adapter 'xxx' not in registry"
    assert payload["error"]["param"] == "model"


def test_json_schema_validation_error_is_400() -> None:
    err = r.JSONSchemaValidationError("guided decoding unavailable")
    assert err.http_status == 400
    # Instructor retries on exactly this ``type``.
    assert err.openai_type == "json_schema_validation"

"""Unit tests for :mod:`src.serving.schemas`.

Covers the quirks that the module docstring lists as "the
silent killers" in the 4 OpenAI-compat SDKs (openai, langchain,
instructor, litellm). Each test pins one concrete behaviour
that, if broken, would cause a downstream agent chain to fail
silently — correlating ids, content-vs-tool_calls split,
streaming delta shape, permissive request parsing, etc.

No MLX / mlx_lm / safetensors imports — the schemas module is
pure Pydantic and these tests must run in a bare-bones CI
environment with only ``pydantic`` and ``pytest``.
"""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.serving import schemas as s


# ---------------------------------------------------------------------------
# Request parsing — OpenAI baseline shape.
# ---------------------------------------------------------------------------


def test_basic_chat_completion_request_roundtrips() -> None:
    """A minimal user-message request parses and round-trips
    exactly. Guards against accidental required-field additions."""
    payload = {
        "model": "qwen3.6-35b-python",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.7,
    }
    req = s.ChatCompletionRequest.model_validate(payload)
    assert req.model == "qwen3.6-35b-python"
    assert len(req.messages) == 1
    assert req.messages[0].role == "user"
    # Round-trip preserves all declared fields.
    assert req.model_dump()["model"] == "qwen3.6-35b-python"


def test_empty_messages_list_rejected() -> None:
    """OpenAI API requires at least one message — silent empty
    lists produce nonsense completions downstream."""
    with pytest.raises(ValidationError):
        s.ChatCompletionRequest.model_validate(
            {"model": "x", "messages": []}
        )


# ---------------------------------------------------------------------------
# LiteLLM + provider-specific extras must be silently dropped.
# ---------------------------------------------------------------------------


def test_litellm_extra_body_dropped() -> None:
    """LiteLLM injects provider-specific kwargs via ``extra_body``.
    A strict schema would reject them — we need ``extra=ignore``
    on every inbound model."""
    payload = {
        "model": "x",
        "messages": [{"role": "user", "content": "hi"}],
        "custom_openrouter_header": "foo",
        "anthropic_metadata": {"tenant": "a"},
    }
    req = s.ChatCompletionRequest.model_validate(payload)
    dumped = req.model_dump()
    assert "custom_openrouter_header" not in dumped
    assert "anthropic_metadata" not in dumped


# ---------------------------------------------------------------------------
# LangChain ``stream_options.include_usage`` — must be round-tripped.
# ---------------------------------------------------------------------------


def test_langchain_stream_options_include_usage() -> None:
    req = s.ChatCompletionRequest.model_validate(
        {
            "model": "x",
            "messages": [{"role": "user", "content": "x"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    )
    assert req.stream is True
    assert req.stream_options is not None
    assert req.stream_options.include_usage is True


# ---------------------------------------------------------------------------
# Tool / function calling — the compat-hot area.
# ---------------------------------------------------------------------------


def test_tools_array_parses() -> None:
    req = s.ChatCompletionRequest.model_validate(
        {
            "model": "x",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Fetch current weather.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
            "tool_choice": "auto",
        }
    )
    assert req.tools is not None
    assert req.tools[0].function.name == "get_weather"
    assert req.tool_choice == "auto"


def test_tool_choice_structured_function_form() -> None:
    """``tool_choice = {"type": "function", "function": {"name": "f"}}``
    is the forced-tool form — LangChain emits this when
    ``bind_tools(..., tool_choice="f")`` is used."""
    req = s.ChatCompletionRequest.model_validate(
        {
            "model": "x",
            "messages": [{"role": "user", "content": "x"}],
            "tool_choice": {
                "type": "function",
                "function": {"name": "f"},
            },
        }
    )
    assert isinstance(req.tool_choice, s.ToolChoice)
    assert req.tool_choice.function.name == "f"


def test_tool_choice_required_literal() -> None:
    """``tool_choice = "required"`` (OpenAI 2024-06) forces at
    least one tool call. AutoGen uses it for gate actions."""
    req = s.ChatCompletionRequest.model_validate(
        {
            "model": "x",
            "messages": [{"role": "user", "content": "x"}],
            "tool_choice": "required",
        }
    )
    assert req.tool_choice == "required"


# ---------------------------------------------------------------------------
# Tool-call id correlation — the Instructor / AutoGen chain-breaker.
# ---------------------------------------------------------------------------


def test_assistant_tool_calls_then_tool_message_correlates() -> None:
    """Assistant response carries ``tool_calls[].id``. The
    client replies with a ToolMessage carrying the same id in
    ``tool_call_id``. AutoGen and Instructor both rely on this
    exact-string correlation to advance the chain."""
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
                            "id": "call_abc123def456ghi789jk",
                            "type": "function",
                            "function": {
                                "name": "f",
                                "arguments": "{}",
                            },
                            "index": 0,
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "result",
                    "tool_call_id": "call_abc123def456ghi789jk",
                },
            ],
        }
    )
    assistant_msg = req.messages[1]
    tool_msg = req.messages[2]
    assert isinstance(assistant_msg, s.AssistantMessage)
    assert isinstance(tool_msg, s.ToolMessage)
    assert assistant_msg.tool_calls is not None
    assert tool_msg.tool_call_id == assistant_msg.tool_calls[0].id


def test_tool_call_id_must_start_with_call_prefix() -> None:
    with pytest.raises(ValidationError) as exc:
        s.ToolCall(function=s.FunctionCall(name="f"), id="bad")
    assert "call_" in str(exc.value)


def test_tool_call_arguments_always_string() -> None:
    """``arguments`` must be a JSON-encoded string, never a dict.
    OpenAI chose string to allow streaming deltas ; Instructor
    parses with ``json.loads(args)`` and a dict breaks it."""
    tc = s.ToolCall(
        function=s.FunctionCall(
            name="f", arguments=json.dumps({"city": "Paris"})
        )
    )
    assert isinstance(tc.function.arguments, str)
    assert json.loads(tc.function.arguments) == {"city": "Paris"}


# ---------------------------------------------------------------------------
# Assistant content-vs-tool_calls split (the silent killer #1).
# ---------------------------------------------------------------------------


def test_assistant_pure_tool_call_has_null_content() -> None:
    """When the assistant's response is purely tool calls,
    ``content`` must be ``None``, not ``""``. The OpenAI SDK
    parses both but Instructor relies on ``None`` to detect
    "no text output"."""
    msg = s.AssistantMessage(
        content=None,
        tool_calls=[
            s.ToolCall(
                function=s.FunctionCall(name="f", arguments="{}")
            )
        ],
    )
    dumped = msg.model_dump()
    assert dumped["content"] is None
    assert dumped["content"] != ""


# ---------------------------------------------------------------------------
# JSON mode / structured outputs.
# ---------------------------------------------------------------------------


def test_response_format_json_object() -> None:
    req = s.ChatCompletionRequest.model_validate(
        {
            "model": "x",
            "messages": [{"role": "user", "content": "x"}],
            "response_format": {"type": "json_object"},
        }
    )
    assert req.response_format is not None
    assert req.response_format.type == "json_object"


def test_response_format_json_schema_strict() -> None:
    """OpenAI 2024-08 structured outputs. ``schema`` is a
    Pydantic-reserved word so we alias it to ``schema_`` on the
    Python side but keep the wire name."""
    payload = {
        "model": "x",
        "messages": [{"role": "user", "content": "x"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}},
                    "required": ["a"],
                },
                "strict": True,
            },
        },
    }
    req = s.ChatCompletionRequest.model_validate(payload)
    assert req.response_format is not None
    assert req.response_format.type == "json_schema"
    assert req.response_format.json_schema is not None
    assert req.response_format.json_schema.name == "answer"
    # ``schema`` comes in on the wire, lives as ``schema_`` in Python.
    assert req.response_format.json_schema.schema_ == payload[
        "response_format"
    ]["json_schema"]["schema"]


# ---------------------------------------------------------------------------
# Response shape.
# ---------------------------------------------------------------------------


def test_chat_completion_response_has_required_openai_fields() -> None:
    resp = s.ChatCompletion(
        model="qwen3.6-35b-python",
        choices=[
            s.ChatCompletionChoice(
                index=0,
                message=s.AssistantMessage(content="hello"),
                finish_reason="stop",
            )
        ],
        usage=s.Usage(
            prompt_tokens=5, completion_tokens=3, total_tokens=8
        ),
    )
    payload = resp.model_dump()
    # Strict OpenAI contract : these 5 fields are MUSTs.
    assert payload["id"].startswith("chatcmpl-")
    assert payload["object"] == "chat.completion"
    assert isinstance(payload["created"], int)
    assert payload["model"] == "qwen3.6-35b-python"
    assert len(payload["choices"]) == 1
    # Usage shape MUST have all three fields when present.
    assert payload["usage"] == {
        "prompt_tokens": 5,
        "completion_tokens": 3,
        "total_tokens": 8,
    }


def test_completion_id_shape() -> None:
    """The regex ``^chatcmpl-[A-Za-z0-9]+$`` is what the OpenAI
    SDK parses ids against."""
    cid = s.new_completion_id()
    assert cid.startswith("chatcmpl-")
    # After the prefix, only hex chars (from uuid4().hex).
    suffix = cid[len("chatcmpl-"):]
    assert all(c in "0123456789abcdef" for c in suffix)


# ---------------------------------------------------------------------------
# Streaming chunks.
# ---------------------------------------------------------------------------


def test_stream_chunk_intermediate_has_no_finish_reason() -> None:
    """The OpenAI SDK raises if any intermediate chunk carries a
    non-null finish_reason. Only the final chunk may have one."""
    chunk = s.ChatCompletionChunk(
        id="chatcmpl-abc",
        model="x",
        choices=[
            s.ChatCompletionChunkChoice(
                index=0,
                delta=s.ChoiceDelta(content="tok"),
            )
        ],
    )
    assert chunk.choices[0].finish_reason is None
    assert chunk.object == "chat.completion.chunk"


def test_stream_chunk_tool_call_delta_with_index() -> None:
    """LangChain concatenates tool-call arguments deltas by
    (id, index). Missing index breaks the accumulator."""
    chunk = s.ChatCompletionChunk(
        id="chatcmpl-abc",
        model="x",
        choices=[
            s.ChatCompletionChunkChoice(
                index=0,
                delta=s.ChoiceDelta(
                    tool_calls=[
                        s.ChoiceDeltaToolCall(
                            index=0,
                            function=s.FunctionCall(
                                name="f", arguments='{"a":'
                            ),
                        )
                    ]
                ),
            )
        ],
    )
    assert chunk.choices[0].delta.tool_calls is not None
    assert chunk.choices[0].delta.tool_calls[0].index == 0


def test_stream_chunk_final_carries_usage() -> None:
    """When ``stream_options.include_usage=True`` was requested,
    the final chunk emits ``usage`` populated + ``choices=[]``
    empty. LangChain's callback manager requires this chunk to
    record token counts."""
    final = s.ChatCompletionChunk(
        id="chatcmpl-abc",
        model="x",
        choices=[],
        usage=s.Usage(
            prompt_tokens=5, completion_tokens=3, total_tokens=8
        ),
    )
    assert final.choices == []
    assert final.usage is not None
    assert final.usage.total_tokens == 8


# ---------------------------------------------------------------------------
# Error shape.
# ---------------------------------------------------------------------------


def test_error_response_shape() -> None:
    """LiteLLM retries on 5xx but not on 4xx ; a wrong status
    code will cause either self-DDoS (retry on 500) or lost
    signal (swallow on 400). Validate the error envelope
    Instructor inspects for structured errors."""
    err = s.ErrorResponse(
        error=s.ErrorDetail(
            message="prompt too long",
            type="invalid_request_error",
            param="messages",
            code="context_length_exceeded",
        )
    )
    dumped = err.model_dump()
    assert dumped["error"]["type"] == "invalid_request_error"
    assert dumped["error"]["code"] == "context_length_exceeded"


# ---------------------------------------------------------------------------
# /v1/models list.
# ---------------------------------------------------------------------------


def test_models_list_shape() -> None:
    ml = s.ModelList(
        data=[
            s.ModelEntry(id="qwen3.6-35b-python"),
            s.ModelEntry(id="qwen3.6-35b-cpp"),
        ]
    )
    dumped = ml.model_dump()
    assert dumped["object"] == "list"
    assert len(dumped["data"]) == 2
    assert dumped["data"][0]["object"] == "model"
    assert dumped["data"][0]["owned_by"] == "factory4life"


# ---------------------------------------------------------------------------
# Multi-modal user content (vision-ready).
# ---------------------------------------------------------------------------


def test_user_message_accepts_string_and_list_forms() -> None:
    """Qwen3.6 base is text-only but the runtime should accept
    multi-modal user content and drop the image parts rather than
    400 the request — vision-capable clients are common."""
    req_str = s.ChatCompletionRequest.model_validate(
        {
            "model": "x",
            "messages": [{"role": "user", "content": "hi"}],
        }
    )
    req_list = s.ChatCompletionRequest.model_validate(
        {
            "model": "x",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,..."},
                        },
                    ],
                }
            ],
        }
    )
    assert isinstance(req_str.messages[0].content, str)
    assert isinstance(req_list.messages[0].content, list)

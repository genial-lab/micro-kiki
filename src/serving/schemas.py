"""OpenAI-compatible Pydantic schemas for factory4life serving.

Covers the minimum surface required by the 4 SDKs that factory4life
chains call into: ``openai>=1.54``, ``langchain-openai>=0.3``,
``instructor>=1.7``, ``litellm>=1.55``. The schemas mirror the
OpenAI Platform API ``/v1/chat/completions`` spec as of
2026-04 — tool calling, JSON mode (json_object + json_schema),
streaming deltas with partial tool_call.arguments, ``seed`` for
deterministic replay, ``stream_options.include_usage`` for
LangChain's callback manager.

Design choices worth flagging :

- ``model_config = {"extra": "ignore"}`` on every request-side model.
  LiteLLM passes through provider-specific fields that OpenAI never
  defined ; a strict schema would reject them and kill every
  LiteLLM-backed chain. We accept and silently drop.
- ``content: str | None`` — not ``str | None = None``. When the
  assistant emits tool_calls, ``content`` must be explicitly
  ``null``, never ``""``. The OpenAI SDK parses both but Instructor
  relies on ``None`` to detect "pure tool call" responses.
- ``id: str`` must match ``^chatcmpl-[A-Za-z0-9]+$`` client-side.
  The runtime wrapper generates ``chatcmpl-<uuid4-hex>``.
- ``tool_calls[].id`` must match ``^call_[A-Za-z0-9]{20,}$``.
  Instructor correlates response tool_calls to the next
  request's tool message via this id ; a shorter / differently
  formatted id breaks the correlation silently.
- ``finish_reason`` on streaming chunks : ``None`` on every
  intermediate chunk, string on the final chunk only. Emitting a
  reason on an intermediate chunk raises in the OpenAI Python SDK.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# ID helpers — enforce the regex shapes that downstream SDKs expect.
# ---------------------------------------------------------------------------


def new_completion_id() -> str:
    """Return a fresh ``chatcmpl-<hex>`` id.

    The OpenAI Python SDK parses this shape with a regex ; a UUID4
    hex (no dashes) is the safest format across 4+ SDK versions.
    """
    return f"chatcmpl-{uuid.uuid4().hex}"


def new_tool_call_id() -> str:
    """Return a fresh ``call_<20+ char hex>`` id for a tool call.

    Instructor correlates response tool_calls to subsequent tool
    messages by this id ; OpenAI ids are 24 chars (call_ + 20
    base62), we use 24 hex to be safe across parsers.
    """
    return f"call_{uuid.uuid4().hex[:20]}"


# ---------------------------------------------------------------------------
# Tool / function calling schemas.
# ---------------------------------------------------------------------------


class FunctionSpec(BaseModel):
    """Tool-side function definition (what the caller advertises).

    Matches ``tools[].function`` in the OpenAI request. The
    ``parameters`` is a JSON Schema object — we accept any dict
    shape and defer schema validation to the JSON-mode guided
    decoder (outlines / xgrammar) or to the agent framework
    (Instructor) that parses the response.
    """

    model_config = ConfigDict(extra="ignore")

    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    strict: bool | None = None  # OpenAI strict-schema flag, 2024-08


class Tool(BaseModel):
    """Top-level tool entry in ``request.tools``."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["function"] = "function"
    function: FunctionSpec


class ToolChoiceFunction(BaseModel):
    """``tool_choice = {"type": "function", "function": {"name": "..."}}``."""

    model_config = ConfigDict(extra="ignore")

    name: str


class ToolChoice(BaseModel):
    """Structured tool_choice form ; string forms ``auto`` /
    ``none`` / ``required`` are carried as bare strings on the
    request and need no wrapper model."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["function"] = "function"
    function: ToolChoiceFunction


class FunctionCall(BaseModel):
    """Response-side : a single function invocation requested by
    the model. ``arguments`` is always a JSON string (not a dict),
    even when the function signature is trivially ``{}``. Matches
    the OpenAI quirk that arguments travel as a string so clients
    can stream partial JSON deltas."""

    model_config = ConfigDict(extra="ignore")

    name: str
    arguments: str = ""  # JSON-encoded string (not dict)


class ToolCall(BaseModel):
    """Response-side tool_calls entry.

    ``index`` matters for streaming : LangChain concatenates
    arguments deltas by (tool_call_id, index). A missing index
    breaks the accumulator. On non-streaming responses, ``index``
    is still emitted (0, 1, 2...) to keep the response shape
    uniform.
    """

    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=new_tool_call_id)
    type: Literal["function"] = "function"
    function: FunctionCall
    index: int = 0

    @field_validator("id")
    @classmethod
    def _id_shape(cls, v: str) -> str:
        if not v.startswith("call_"):
            raise ValueError(
                f"tool_call id must start with 'call_', got {v!r}"
            )
        return v


# ---------------------------------------------------------------------------
# Message schemas — request (input) + response (output) sides.
# ---------------------------------------------------------------------------


class SystemMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["system"] = "system"
    content: str
    name: str | None = None


class UserMessage(BaseModel):
    """User message. OpenAI supports multi-modal content (text /
    image_url) as a list of parts ; we accept both the string
    form and the list form to stay compatible with vision-capable
    clients even though the Qwen3.6 base is text-only (vision
    parts are dropped at the runtime layer, not here)."""

    model_config = ConfigDict(extra="ignore")

    role: Literal["user"] = "user"
    content: str | list[dict[str, Any]]
    name: str | None = None


class AssistantMessage(BaseModel):
    """Assistant message in the request (for multi-turn
    conversations) OR response. The ``content | tool_calls`` split
    is the single most brittle edge case in OpenAI-compat — see
    module docstring."""

    model_config = ConfigDict(extra="ignore")

    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    name: str | None = None
    # Legacy "function_call" form (pre-2023-11). A few older
    # LangChain versions still send it ; we accept but never emit.
    function_call: FunctionCall | None = None


class ToolMessage(BaseModel):
    """Client-side tool output message (role="tool"). The
    ``tool_call_id`` is the correlation key back to the assistant
    message that requested the tool. AutoGen / LangGraph chains
    loop forever if this id does not match exactly."""

    model_config = ConfigDict(extra="ignore")

    role: Literal["tool"] = "tool"
    content: str
    tool_call_id: str


Message = SystemMessage | UserMessage | AssistantMessage | ToolMessage


# ---------------------------------------------------------------------------
# Response format (JSON mode) schemas.
# ---------------------------------------------------------------------------


class JSONSchemaSpec(BaseModel):
    """OpenAI structured-outputs JSON schema descriptor (2024-08)."""

    model_config = ConfigDict(extra="ignore")

    name: str
    description: str | None = None
    schema_: dict[str, Any] = Field(alias="schema")
    strict: bool | None = None


class ResponseFormat(BaseModel):
    """``response_format`` on the request.

    Three supported shapes, ordered by strictness :

    - ``{"type": "text"}`` — no constraint (default).
    - ``{"type": "json_object"}`` — free-form JSON, guided by a
      prompt instruction. The runtime layer injects a "respond in
      JSON" hint if the caller didn't.
    - ``{"type": "json_schema", "json_schema": {...}}`` — strict
      structured output. Requires guided decoding (outlines /
      xgrammar) ; if the runtime lacks it, return a 400 with
      ``error.type = "json_schema_validation"`` so Instructor can
      retry with a simpler schema.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: JSONSchemaSpec | None = None


# ---------------------------------------------------------------------------
# Streaming options.
# ---------------------------------------------------------------------------


class StreamOptions(BaseModel):
    """``stream_options`` on the request.

    LangChain always sends ``include_usage: true`` ; we MUST emit a
    final chunk with ``usage`` populated and ``choices: []`` empty.
    Without it, LangChain's token-counting callback records 0
    tokens and downstream billing / observability breaks.
    """

    model_config = ConfigDict(extra="ignore")

    include_usage: bool | None = None


# ---------------------------------------------------------------------------
# Request.
# ---------------------------------------------------------------------------


class ChatCompletionRequest(BaseModel):
    """Incoming ``POST /v1/chat/completions`` body.

    Permissive by design (``extra=ignore``) so LiteLLM extra-body
    fields, provider-specific kwargs, and future OpenAI params
    don't 400 the request. The runtime layer picks up the fields
    it understands.
    """

    model_config = ConfigDict(extra="ignore")

    model: str  # e.g. "qwen3.6-35b-python" or alias "qwen3.6-35b-auto"
    messages: list[Message]

    # Sampling.
    temperature: float = 0.7
    top_p: float = 1.0
    n: int = 1
    max_tokens: int | None = None
    max_completion_tokens: int | None = None  # OpenAI renamed in 2024
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int | None = None
    stop: str | list[str] | None = None
    logit_bias: dict[str, float] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None

    # Tool / function calling.
    tools: list[Tool] | None = None
    tool_choice: Literal["auto", "none", "required"] | ToolChoice | None = None
    parallel_tool_calls: bool | None = None

    # Output format.
    response_format: ResponseFormat | None = None

    # Streaming.
    stream: bool = False
    stream_options: StreamOptions | None = None

    # Pass-through metadata.
    user: str | None = None

    @field_validator("messages")
    @classmethod
    def _non_empty(cls, v: list[Message]) -> list[Message]:
        if not v:
            raise ValueError("messages must contain at least one entry")
        return v


# ---------------------------------------------------------------------------
# Response (non-streaming).
# ---------------------------------------------------------------------------


class Usage(BaseModel):
    """Token accounting block. When emitted, all three fields are
    required — partial emission breaks OpenAI SDK's usage parser."""

    model_config = ConfigDict(extra="ignore")

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(BaseModel):
    model_config = ConfigDict(extra="ignore")

    index: int
    message: AssistantMessage
    finish_reason: Literal[
        "stop", "length", "tool_calls", "content_filter", "function_call"
    ]
    logprobs: dict[str, Any] | None = None


class ChatCompletion(BaseModel):
    """Full non-streaming response body."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=new_completion_id)
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage | None = None
    system_fingerprint: str | None = None


# ---------------------------------------------------------------------------
# Response (streaming chunks).
# ---------------------------------------------------------------------------


class ChoiceDeltaToolCall(BaseModel):
    """Streaming delta for a single tool call.

    Only the fields that changed relative to the previous chunk
    should be present. For ``arguments`` specifically, LangChain
    concatenates deltas — the server must emit incremental
    substrings, never the full accumulated string. Emitting
    ``arguments="{\"a\":1"`` on chunk 1 and ``arguments="{\"a\":1,\"b\":2}"``
    on chunk 2 results in ``{"a":1{"a":1,"b":2}`` in the client
    buffer and a parse failure.
    """

    model_config = ConfigDict(extra="ignore")

    index: int
    id: str | None = None
    type: Literal["function"] | None = None
    function: FunctionCall | None = None


class ChoiceDelta(BaseModel):
    """One streaming delta — the portion of the assistant message
    produced since the previous chunk."""

    model_config = ConfigDict(extra="ignore")

    role: Literal["assistant"] | None = None
    content: str | None = None
    tool_calls: list[ChoiceDeltaToolCall] | None = None


class ChatCompletionChunkChoice(BaseModel):
    model_config = ConfigDict(extra="ignore")

    index: int
    delta: ChoiceDelta
    finish_reason: (
        Literal["stop", "length", "tool_calls", "content_filter", "function_call"]
        | None
    ) = None
    logprobs: dict[str, Any] | None = None


class ChatCompletionChunk(BaseModel):
    """One streaming chunk — wire-formatted as
    ``data: <chunk_json>\\n\\n`` in the SSE response."""

    model_config = ConfigDict(extra="ignore")

    id: str  # stable across all chunks of a single response
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChunkChoice]
    usage: Usage | None = None  # populated on the final chunk only
    system_fingerprint: str | None = None


# ---------------------------------------------------------------------------
# /v1/models response.
# ---------------------------------------------------------------------------


class ModelEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "factory4life"


class ModelList(BaseModel):
    model_config = ConfigDict(extra="ignore")

    object: Literal["list"] = "list"
    data: list[ModelEntry]


# ---------------------------------------------------------------------------
# Error response.
# ---------------------------------------------------------------------------


class ErrorDetail(BaseModel):
    model_config = ConfigDict(extra="ignore")

    message: str
    type: str
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    error: ErrorDetail


__all__ = [
    "AssistantMessage",
    "ChatCompletion",
    "ChatCompletionChoice",
    "ChatCompletionChunk",
    "ChatCompletionChunkChoice",
    "ChatCompletionRequest",
    "ChoiceDelta",
    "ChoiceDeltaToolCall",
    "ErrorDetail",
    "ErrorResponse",
    "FunctionCall",
    "FunctionSpec",
    "JSONSchemaSpec",
    "Message",
    "ModelEntry",
    "ModelList",
    "ResponseFormat",
    "StreamOptions",
    "SystemMessage",
    "Tool",
    "ToolCall",
    "ToolChoice",
    "ToolChoiceFunction",
    "ToolMessage",
    "Usage",
    "UserMessage",
    "new_completion_id",
    "new_tool_call_id",
]

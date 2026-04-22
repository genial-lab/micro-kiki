"""MLX runtime abstraction for the factory4life serving layer.

Separates the OpenAI-compat endpoint code (``app.py``, T6) from
the underlying inference backend. Exposes two things :

- :class:`MLXRuntime` — ``Protocol`` describing the contract that
  any backend (vanilla LoRA, MoE-native LoRA, mock) must satisfy.
  The endpoint code talks only to the Protocol.
- :class:`FakeMLXRuntime` — pure-Python implementation that
  returns scripted responses. Used by tests T6/T7/T8 and by
  local dev without Studio hardware.

The real Vanilla / MoE implementations live in separate modules
(``vanilla_mlx_runtime.py``, ``moe_lora_runtime.py``) and lazy-
import MLX so this file stays importable on CI without the
Apple Silicon wheel.

Why a Protocol, not an ABC ?

- Structural subtyping lets ``moe_lora_runtime.MoELoRARuntime``
  (already written, 802 LOC) satisfy the contract without any
  inheritance ; we don't have to touch the existing MoE code.
- Tests can pass a bare dataclass that only implements the
  methods they exercise — no risk of "forgot to implement one
  method" tripping the whole test suite.

Determinism note
----------------

``seed`` is hashed via sha256 (not ``hash(seed)``, randomised per
process since Python 3.3) so two runtime instances at different
times / on different hosts give the same output for the same
prompt + seed. Crucial for Instructor retries and for
reproducible agent chain debugging.
"""
from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Protocol, runtime_checkable

from . import schemas as s


# ---------------------------------------------------------------------------
# Public exceptions.
# ---------------------------------------------------------------------------


class RuntimeError_(RuntimeError):
    """Base class for runtime-originated errors that should be
    surfaced to the HTTP layer as a structured OpenAI error."""

    openai_type: str = "server_error"
    http_status: int = 500

    def to_error(self, param: str | None = None) -> s.ErrorResponse:
        return s.ErrorResponse(
            error=s.ErrorDetail(
                message=str(self), type=self.openai_type, param=param,
            )
        )


class AdapterNotFound(RuntimeError_):
    openai_type = "adapter_not_found"
    http_status = 404


class JSONSchemaValidationError(RuntimeError_):
    """Raised when the runtime can't honor a
    ``response_format: json_schema`` — typically because guided
    decoding is unavailable. Instructor retries on this exact
    type string."""

    openai_type = "json_schema_validation"
    http_status = 400


class ContextLengthExceeded(RuntimeError_):
    openai_type = "context_length_exceeded"
    http_status = 400


# ---------------------------------------------------------------------------
# Runtime Protocol.
# ---------------------------------------------------------------------------


@runtime_checkable
class MLXRuntime(Protocol):
    """Contract that every serving backend must satisfy.

    Two generation methods (one-shot + streaming) so the endpoint
    doesn't need to know whether ``stream=True`` was requested
    ahead of dispatching to the backend.
    """

    async def generate(
        self, request: s.ChatCompletionRequest,
    ) -> s.ChatCompletion:
        """Non-streaming generate. Returns a complete
        :class:`ChatCompletion` — the full assistant message,
        tool_calls (if any), usage totals."""
        ...

    async def generate_stream(
        self, request: s.ChatCompletionRequest,
    ) -> AsyncIterator[s.ChatCompletionChunk]:
        """Streaming generate. Yields a sequence of chunks ending
        in the SSE sentinel chunk. The caller is responsible for
        wrapping each chunk in ``data: {...}\\n\\n`` and appending
        ``data: [DONE]\\n\\n``."""
        ...

    def list_adapters(self) -> list[str]:
        """Available adapter names — becomes
        ``/v1/models.data[].id``. Each runtime exposes its
        adapters ; the endpoint may namespace-prefix them (e.g.
        ``qwen3.6-35b-{name}``)."""
        ...

    def health(self) -> dict[str, Any]:
        """``/health`` payload. Free-form ; the endpoint wraps
        it with uptime + process info."""
        ...

    def kv_stats(self) -> dict[str, Any]:
        """KV cache + session pool snapshot.

        Backs the ``/v1/internal/kv-status`` endpoint. The
        endpoint dimensions the serving fleet under load — operators
        need to see sessions_active, kv_bytes_used, prefix_cache
        hit_rate, and the runtime's advertised max_context_tokens
        (Qwen3.6-35B-A3B native is 262K, YaRN-extended up to 1M).

        Standardised keys (all optional — runtimes that can't
        observe a metric omit it) :

        - ``sessions_active`` : int, live sessions holding a KV cache.
        - ``kv_bytes_used`` : int, total bytes currently allocated
          to KV cache across all sessions.
        - ``kv_bytes_budget`` : int, soft cap before eviction kicks in.
        - ``prefix_cache_entries`` : int, distinct cached prefixes.
        - ``prefix_cache_hit_rate`` : float in [0, 1].
        - ``max_context_tokens`` : int, the runtime's hard context
          ceiling. 262144 native, up to 1048576 with YaRN scaling.
        - ``context_tokens_p50`` / ``_p99`` : int, per-session
          context-length distribution on the trailing N requests.
        - ``runtime`` : str, runtime class identifier.
        """
        ...

    def current_context_ceiling(self) -> int:
        """Live context ceiling for a *new* session right now.

        Computes ``min(max_context_tokens, kv_free_bytes / kv_per_token)``.
        Used by the ``/v1/chat/completions`` admission step to
        clamp (or reject) requests whose reserved context would
        overflow the remaining KV pool.

        The ceiling is dynamic — it shrinks as more sessions open,
        grows again as they close / get evicted. The endpoint
        echoes it in the ``X-Max-Context-Available`` response
        header so clients can pre-empt the next request before
        hitting a 413.

        Backends that don't enforce a KV budget may return
        ``max_context_tokens`` verbatim ; the interface still
        exists so callers can bind to a single signature.
        """
        ...


# ---------------------------------------------------------------------------
# Seed helpers.
# ---------------------------------------------------------------------------


def stable_seed(seed: int | None, prompt: str) -> int:
    """Hash the caller-supplied seed + prompt to a 32-bit int.

    If ``seed`` is ``None``, derives a stable pseudo-seed from the
    prompt alone — the same prompt produces the same pseudo-seed
    across processes. If ``seed`` is given, the prompt is mixed in
    so that two different prompts with the same seed still get
    different RNG states (matches OpenAI's documented behaviour).
    """
    payload = f"{seed if seed is not None else 'no-seed'}::{prompt}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


# ---------------------------------------------------------------------------
# Adapter name resolution.
# ---------------------------------------------------------------------------


def resolve_adapter_name(
    model_id: str,
    aliases: dict[str, str] | None = None,
) -> str | None:
    """Map ``request.model`` to an adapter name or ``None`` for
    base-only.

    Recognised forms :

    - ``"qwen3.6-35b-{adapter}"`` → strip prefix, return adapter.
    - ``"qwen3.6-35b"`` / ``"base"`` → ``None`` (no adapter).
    - ``"qwen3.6-35b-auto"`` → reserved alias, caller routes via
      the v4 router ; returned as ``"__router__"``.
    - Any entry in ``aliases`` maps to its target.

    Returns ``None`` when the model id resolves to the bare base.
    """
    if aliases and model_id in aliases:
        model_id = aliases[model_id]
    if model_id in {"base", "qwen3.6-35b", "qwen3.6-35b-a3b"}:
        return None
    if model_id == "qwen3.6-35b-auto":
        return "__router__"
    prefix = "qwen3.6-35b-"
    if model_id.startswith(prefix):
        return model_id[len(prefix):]
    # Fall back : treat the id itself as an adapter name.
    return model_id


# ---------------------------------------------------------------------------
# FakeMLXRuntime — deterministic scripted backend for tests + dev.
# ---------------------------------------------------------------------------


@dataclass
class FakeMLXRuntime:
    """Scripted backend — no MLX, no real tokenization, no GPU.

    Useful for :

    - T6-T8 endpoint tests that don't need real completions.
    - Local dev on kxkm-ai without the 70 GB base model resident.
    - Compat tests against OpenAI / LangChain / Instructor SDKs
      where the actual text is irrelevant but the response shape
      must be OpenAI-spec.

    Behaviour :

    - Default ``completion_for`` returns a deterministic string
      derived from the prompt + seed via sha256.
    - ``scripted_responses`` lets a test pin the content for a
      specific prompt substring.
    - ``tool_calls`` is ``None`` unless ``force_tool_call`` is
      set and the request included at least one tool.
    """

    adapters: list[str] = field(
        default_factory=lambda: ["python", "cpp", "reasoning", "chat-fr"],
    )
    scripted_responses: dict[str, str] = field(default_factory=dict)
    force_tool_call: s.ToolCall | None = None
    completion_for: Callable[[str, int], str] | None = None
    stream_token_delay_s: float = 0.0
    started_at: float = field(default_factory=time.time)
    # Qwen3.6-35B-A3B native is 262 144 tokens. With YaRN /
    # NTK-aware RoPE scaling, we can push to 1 048 576. The
    # fake backend reports the ceiling the real runtime will be
    # configured with ; tests can bump this to exercise clamp
    # logic without needing MLX.
    max_context_tokens: int = 1_048_576
    # Session / KV cache accounting — the fake backend doesn't
    # really hold sessions, but it maintains the counters so
    # the ``/v1/internal/kv-status`` endpoint has something to
    # show and downstream dashboards exercise their render paths.
    _sessions_active: int = 0
    _kv_bytes_used: int = 0
    _kv_bytes_budget: int = 60 * 1024**3  # 60 GB default pool
    _prefix_cache_entries: int = 0
    _prefix_cache_hits: int = 0
    _prefix_cache_lookups: int = 0
    _context_tokens_observed: list[int] = field(default_factory=list)

    # -------------------------------------------------------------------
    # Protocol methods.
    # -------------------------------------------------------------------

    def list_adapters(self) -> list[str]:
        return list(self.adapters)

    def health(self) -> dict[str, Any]:
        return {
            "runtime": "fake",
            "adapters_warm": len(self.adapters),
            "max_context_tokens": self.max_context_tokens,
            "uptime_s": round(time.time() - self.started_at, 1),
        }

    def current_context_ceiling(self) -> int:
        """Live ceiling = min(configured max, what the KV pool
        can still absorb)."""
        kv_per_token = 40 * 1024
        free = max(
            0, self._kv_bytes_budget - self._kv_bytes_used,
        )
        # Always admit at least 1 token of ceiling so the handler
        # can surface an explicit "context_length_exceeded" 413
        # rather than a divide-by-zero.
        fit = max(1, free // kv_per_token) if kv_per_token else 1
        return min(self.max_context_tokens, fit)

    def kv_stats(self) -> dict[str, Any]:
        """Synthesise a plausible KV / session snapshot.

        Real runtimes would read live counters from their
        tokenizer + cache layers ; the fake returns whatever the
        tests pushed into the accumulator attributes so the
        endpoint shape can be asserted without a real model.
        """
        hit_rate = 0.0
        if self._prefix_cache_lookups > 0:
            hit_rate = round(
                self._prefix_cache_hits / self._prefix_cache_lookups,
                3,
            )
        ctx = sorted(self._context_tokens_observed)
        p50 = p99 = 0
        if ctx:
            p50 = ctx[len(ctx) // 2]
            p99 = ctx[min(len(ctx) - 1, int(len(ctx) * 0.99))]
        # Typical Qwen3.6-35B-A3B hybrid : 10 full_attn layers,
        # GQA (8 KV heads × 128 head_dim), BF16 → 40 KB per
        # token total across all full-attn layers. Only the
        # full-attn layers grow with context ; linear_attn
        # layers keep a fixed state so context_tokens × 40 KB
        # is the right approximation.
        # Admission control : how many concurrent sessions can
        # the KV budget support at a given target context ? This
        # is the explicit operator knob — "raise max context to
        # 1 M if you cap concurrency at 2" vs "hold 10 sessions
        # at 200 K context", same RAM.
        kv_per_token = 40 * 1024
        def _max_concurrent_at(ctx: int) -> int:
            if ctx <= 0:
                return 0
            return max(1, self._kv_bytes_budget // (ctx * kv_per_token))
        admission = {
            str(ctx): _max_concurrent_at(ctx)
            for ctx in (4096, 16384, 32768, 65536, 131072,
                        262144, 524288, 1_048_576)
        }
        return {
            "runtime": "fake",
            "max_context_tokens": self.max_context_tokens,
            "current_context_ceiling": self.current_context_ceiling(),
            "kv_bytes_per_token": kv_per_token,
            "sessions_active": self._sessions_active,
            "kv_bytes_used": self._kv_bytes_used,
            "kv_bytes_budget": self._kv_bytes_budget,
            "kv_bytes_free": max(
                0, self._kv_bytes_budget - self._kv_bytes_used,
            ),
            "prefix_cache_entries": self._prefix_cache_entries,
            "prefix_cache_hit_rate": hit_rate,
            "prefix_cache_hits": self._prefix_cache_hits,
            "prefix_cache_lookups": self._prefix_cache_lookups,
            "context_tokens_p50": p50,
            "context_tokens_p99": p99,
            "context_tokens_observed": len(self._context_tokens_observed),
            # Admission table : context_tokens → max concurrent
            # sessions the KV budget can hold at that context.
            # Operators pick a row and either (a) cap context
            # via ``max_completion_tokens`` + input clamp or
            # (b) cap concurrency via a semaphore in the app.
            "admission": admission,
        }

    async def generate(
        self, request: s.ChatCompletionRequest,
    ) -> s.ChatCompletion:
        prompt = _flatten_prompt(request.messages)
        seed = stable_seed(request.seed, prompt)
        content, tool_calls = self._script(prompt, seed, request)
        finish: s.ChatCompletionChoice.model_fields["finish_reason"].annotation  # type: ignore[assignment]
        finish = "tool_calls" if tool_calls else "stop"  # type: ignore[assignment]

        # Fake token counting — one "token" per whitespace-split.
        prompt_tokens = max(1, len(prompt.split()))
        completion_tokens = max(1, len((content or "").split()))
        usage = s.Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

        return s.ChatCompletion(
            model=request.model,
            choices=[
                s.ChatCompletionChoice(
                    index=0,
                    message=s.AssistantMessage(
                        content=content if not tool_calls else None,
                        tool_calls=tool_calls,
                    ),
                    finish_reason=finish,  # type: ignore[arg-type]
                )
            ],
            usage=usage,
        )

    async def generate_stream(
        self, request: s.ChatCompletionRequest,
    ) -> AsyncIterator[s.ChatCompletionChunk]:
        prompt = _flatten_prompt(request.messages)
        seed = stable_seed(request.seed, prompt)
        content, tool_calls = self._script(prompt, seed, request)
        cid = s.new_completion_id()

        # First chunk : role marker (OpenAI emits role on chunk 0).
        yield s.ChatCompletionChunk(
            id=cid,
            model=request.model,
            choices=[
                s.ChatCompletionChunkChoice(
                    index=0,
                    delta=s.ChoiceDelta(role="assistant"),
                )
            ],
        )

        if tool_calls:
            # Emit the tool call as a single delta (split arguments
            # deterministically into 4 chunks so tests can exercise
            # the concatenate-by-index logic).
            for tc in tool_calls:
                args = tc.function.arguments or ""
                parts = _split_arguments(args, n=4)
                name_emitted = False
                for i, part in enumerate(parts):
                    if self.stream_token_delay_s:
                        await asyncio.sleep(self.stream_token_delay_s)
                    fn = s.FunctionCall(
                        name=tc.function.name if not name_emitted else "",
                        arguments=part,
                    )
                    name_emitted = True
                    yield s.ChatCompletionChunk(
                        id=cid,
                        model=request.model,
                        choices=[
                            s.ChatCompletionChunkChoice(
                                index=0,
                                delta=s.ChoiceDelta(
                                    tool_calls=[
                                        s.ChoiceDeltaToolCall(
                                            index=tc.index,
                                            id=tc.id if i == 0 else None,
                                            type="function" if i == 0 else None,
                                            function=fn,
                                        )
                                    ]
                                ),
                            )
                        ],
                    )
            finish = "tool_calls"
        else:
            # Plain text — stream one word per chunk.
            for word in (content or "").split():
                if self.stream_token_delay_s:
                    await asyncio.sleep(self.stream_token_delay_s)
                yield s.ChatCompletionChunk(
                    id=cid,
                    model=request.model,
                    choices=[
                        s.ChatCompletionChunkChoice(
                            index=0,
                            delta=s.ChoiceDelta(content=word + " "),
                        )
                    ],
                )
            finish = "stop"

        # Final chunk : empty delta + finish_reason.
        yield s.ChatCompletionChunk(
            id=cid,
            model=request.model,
            choices=[
                s.ChatCompletionChunkChoice(
                    index=0,
                    delta=s.ChoiceDelta(),
                    finish_reason=finish,  # type: ignore[arg-type]
                )
            ],
        )

        # Usage chunk if the caller asked for it (LangChain does).
        include_usage = (
            request.stream_options is not None
            and request.stream_options.include_usage is True
        )
        if include_usage:
            prompt_tokens = max(1, len(prompt.split()))
            completion_tokens = max(
                1, len((content or "").split())
                if not tool_calls
                else sum(
                    len((tc.function.arguments or "").split()) + 1
                    for tc in tool_calls
                ),
            )
            yield s.ChatCompletionChunk(
                id=cid,
                model=request.model,
                choices=[],
                usage=s.Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )

    # -------------------------------------------------------------------
    # Internals.
    # -------------------------------------------------------------------

    def _script(
        self,
        prompt: str,
        seed: int,
        request: s.ChatCompletionRequest,
    ) -> tuple[str | None, list[s.ToolCall] | None]:
        """Return (content, tool_calls) deterministically.

        Priority order :
        1. Explicit ``force_tool_call`` + request has tools → fire it.
        2. Explicit ``scripted_responses[substring]`` → return that.
        3. User-supplied ``completion_for(prompt, seed)`` → run it.
        4. Fallback : "[fake runtime] <short deterministic tag>"
        """
        if self.force_tool_call is not None and request.tools:
            # Validate the forced call references an advertised tool.
            declared = {t.function.name for t in request.tools}
            if self.force_tool_call.function.name in declared:
                return None, [self.force_tool_call]

        for substring, reply in self.scripted_responses.items():
            if substring in prompt:
                return reply, None

        if self.completion_for is not None:
            return self.completion_for(prompt, seed), None

        tag = hashlib.sha256(f"{seed}:{prompt}".encode()).hexdigest()[:6]
        return f"[fake runtime seed={tag}] ok", None


# ---------------------------------------------------------------------------
# Helpers shared with real backends.
# ---------------------------------------------------------------------------


def _flatten_prompt(messages: list[s.Message]) -> str:
    """Fold a message list into a single string for hashing +
    fake generation. Real backends ignore this ; they apply the
    tokenizer chat template instead."""
    parts: list[str] = []
    for m in messages:
        if isinstance(m, s.SystemMessage):
            parts.append(f"[system] {m.content}")
        elif isinstance(m, s.UserMessage):
            content = m.content
            if isinstance(content, list):
                # Drop multimodal parts ; keep text parts.
                text_parts: list[str] = [
                    p.get("text", "")
                    for p in content
                    if p.get("type") == "text"
                ]
                content = " ".join(text_parts)
            parts.append(f"[user] {content}")
        elif isinstance(m, s.AssistantMessage):
            if m.content:
                parts.append(f"[assistant] {m.content}")
            if m.tool_calls:
                for tc in m.tool_calls:
                    parts.append(
                        f"[tool_call:{tc.function.name}] "
                        f"{tc.function.arguments}"
                    )
        elif isinstance(m, s.ToolMessage):
            parts.append(
                f"[tool_result:{m.tool_call_id}] {m.content}"
            )
    return "\n".join(parts)


def _split_arguments(args: str, *, n: int) -> list[str]:
    """Split a JSON-arguments string into ``n`` roughly equal
    chunks so streaming tests can exercise the ``index``-keyed
    concatenate logic. Guarantees ``"".join(result) == args``."""
    if n <= 1 or len(args) <= n:
        return [args]
    size = len(args) // n
    chunks = [args[i * size : (i + 1) * size] for i in range(n - 1)]
    chunks.append(args[(n - 1) * size:])
    return chunks


__all__ = [
    "AdapterNotFound",
    "ContextLengthExceeded",
    "FakeMLXRuntime",
    "JSONSchemaValidationError",
    "MLXRuntime",
    "RuntimeError_",
    "_flatten_prompt",
    "_split_arguments",
    "resolve_adapter_name",
    "stable_seed",
]

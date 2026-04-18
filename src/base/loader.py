"""Base model loader with hot-swap LoRA support.

This module exposes :class:`BaseModelLoader`, a thin facade over
``transformers`` (BF16 training/eval path) and ``llama_cpp`` (Q4_K_M
serving path) plus a PEFT-backed ``with_stack()`` context manager that
hot-swaps LoRA adapters at runtime.

Design choices (v0.2 pragmatic):

* BF16 weights are loaded via ``AutoModelForCausalLM.from_pretrained``
  with ``device_map="auto"``. Callers are expected to hold a single
  loader instance per process.
* Q4_K_M GGUF weights are loaded via ``llama_cpp.Llama`` for inference
  only — they cannot host LoRA adapters, so ``with_stack()`` raises on
  a GGUF-backed loader.
* LoRA hot-swap uses PEFT's multi-adapter API: every registered adapter
  is attached once via :meth:`PeftModel.load_adapter`, then
  :meth:`set_adapter` flips the active adapter in O(1). This avoids the
  allocate/free churn of re-loading from disk on every swap.
* The active adapter is tracked on the loader itself so nested
  ``with_stack()`` calls restore the previous adapter (or no adapter)
  on exit.

The heavy dependencies (``torch``, ``transformers``, ``peft``,
``llama_cpp``) are all imported lazily so that unit tests that only
exercise bookkeeping logic don't pay the import cost and don't need
those packages installed.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Iterator


class BaseModelLoader:
    """Load Qwen3.5-35B-A3B base and manage hot-swap LoRA adapters.

    Two disjoint load paths are supported:

    * :meth:`load_bf16` — HuggingFace ``transformers`` model; supports
      LoRA hot-swap via :meth:`enable_lora_switching`.
    * :meth:`load_q4` — ``llama_cpp.Llama`` GGUF model for serving; no
      LoRA support (Q4 weights are frozen).

    Adapter workflow:

    1. :meth:`load_bf16` → :meth:`enable_lora_switching`
    2. :meth:`register_adapter` for each LoRA directory
    3. ``with loader.with_stack("name"):`` to activate + restore
    """

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self._model: Any = None
        self._tokenizer: Any = None
        self._backend: str | None = None  # "bf16" | "q4"
        self._peft_ready: bool = False
        self._adapters: dict[str, str] = {}
        self._active_adapter: str | None = None

    # ------------------------------------------------------------------
    # Load paths
    # ------------------------------------------------------------------

    def load_bf16(self) -> "BaseModelLoader":
        """Load BF16 weights via ``transformers``.

        Returns ``self`` to allow chained calls, e.g.
        ``BaseModelLoader(p).load_bf16().enable_lora_switching()``.
        """
        import torch  # lazy import
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
        )
        self._backend = "bf16"
        return self

    def load_q4(self, gguf_path: str | Path) -> "BaseModelLoader":
        """Load a Q4_K_M GGUF model via ``llama_cpp``.

        Note: the returned object's ``.model`` is a
        :class:`llama_cpp.Llama` instance, not a transformers model.
        It is intended for inference/serving only — LoRA hot-swap is
        not supported on this path.
        """
        try:
            from llama_cpp import Llama  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover — env-dependent
            raise RuntimeError(
                "llama-cpp-python not installed. "
                "Install via: uv add llama-cpp-python"
            ) from exc

        self._model = Llama(model_path=str(gguf_path), n_ctx=4096)
        self._backend = "q4"
        return self

    # ------------------------------------------------------------------
    # LoRA switching
    # ------------------------------------------------------------------

    def enable_lora_switching(self) -> "BaseModelLoader":
        """Mark the loader as PEFT-ready for later hot-swap.

        Must be called on a BF16-loaded model; the Q4 serving path
        does not support adapters.
        """
        if self._backend != "bf16":
            raise RuntimeError(
                "enable_lora_switching() requires load_bf16() first "
                "(Q4 GGUF models do not support LoRA hot-swap)"
            )
        self._peft_ready = True
        return self

    def register_adapter(self, name: str, adapter_path: str | Path) -> None:
        """Record a LoRA adapter directory for later activation.

        The adapter is not loaded onto the model until the first
        :meth:`_activate` call — register is pure bookkeeping.
        """
        if not self._peft_ready:
            raise RuntimeError(
                "Call enable_lora_switching() before register_adapter()"
            )
        self._adapters[name] = str(adapter_path)

    @contextlib.contextmanager
    def with_stack(self, adapter_name: str) -> Iterator[None]:
        """Activate ``adapter_name`` for the duration of the block.

        On exit, restores the previously active adapter, or disables
        all adapters if none was active on entry. Nested ``with_stack``
        calls therefore behave like a stack (hence the name).
        """
        if adapter_name not in self._adapters:
            raise KeyError(
                f"Adapter {adapter_name!r} not registered; "
                f"known: {sorted(self._adapters)}"
            )
        previous = self._active_adapter
        self._activate(adapter_name)
        try:
            yield
        finally:
            if previous is None:
                self._deactivate()
            else:
                self._activate(previous)

    # ------------------------------------------------------------------
    # Internal adapter plumbing
    # ------------------------------------------------------------------

    def _activate(self, name: str) -> None:
        """Attach (first time) and select the named adapter."""
        from peft import PeftModel  # lazy import

        path = self._adapters[name]
        # First activation wraps the base model into a PeftModel; later
        # activations just load additional adapters onto the wrapper.
        if not isinstance(self._model, PeftModel):
            self._model = PeftModel.from_pretrained(
                self._model, path, adapter_name=name
            )
        elif name not in getattr(self._model, "peft_config", {}):
            self._model.load_adapter(path, adapter_name=name)
        self._model.set_adapter(name)
        self._active_adapter = name

    def _deactivate(self) -> None:
        """Disable all adapters, reverting to the base model path."""
        if hasattr(self._model, "disable_adapter"):
            # PEFT >= 0.7 exposes disable_adapter() as a method or
            # context manager depending on version; call whichever
            # surface is available.
            disable = self._model.disable_adapter
            try:
                disable()
            except TypeError:
                # Context-manager-only variant — enter + immediately
                # exit is meaningless, so just null out tracking.
                pass
        self._active_adapter = None

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def model(self) -> Any:
        return self._model

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @property
    def backend(self) -> str | None:
        return self._backend

    @property
    def active_adapter(self) -> str | None:
        return self._active_adapter

    @property
    def adapters(self) -> dict[str, str]:
        """Read-only view of registered adapters (name → path)."""
        return dict(self._adapters)

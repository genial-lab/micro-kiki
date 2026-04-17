"""Runtime MoE-LoRA inference engine with per-token expert routing.

Unlike the fuse approach (scripts/fuse_moe_lora.py) which averages all
experts into a single static delta (~3e-8 magnitude, effectively invisible),
this engine applies MoE-LoRA dynamically: for each token, the router MLP
selects top-k experts and applies only those LoRA deltas weighted by their
gate scores.

Flow per patched linear layer:
    hidden -> router MLP -> softmax -> top-k selection
    delta = sum(gate_i * (x @ A_i) @ B_i) * scaling  for top-k experts
    output = base_linear(hidden) + delta

Adapter key format (from training):
    language_model.model.layers.{L}.{proj}_moe_lora.experts.{E}.lora_a
    language_model.model.layers.{L}.{proj}_moe_lora.router_w1.weight
    language_model.model.layers.{L}.{proj}_moe_lora.router_w2.weight
    (and corresponding biases)

Supports both MLX (Apple Silicon) and PyTorch backends, auto-detected.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend auto-detection
# ---------------------------------------------------------------------------

_BACKEND: str = "none"

try:
    import mlx.core as mx
    import mlx.nn as mlx_nn
    _BACKEND = "mlx"
except ImportError:
    pass

if _BACKEND == "none":
    try:
        import torch
        import torch.nn as torch_nn
        _BACKEND = "torch"
    except ImportError:
        pass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_ALPHA = 32.0
DEFAULT_RANK = 16
DEFAULT_TOP_K = 2
DEFAULT_NUM_EXPERTS = 4
DEFAULT_ROUTER_HIDDEN = 64

PROJ_NAMES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

SUB_MODULES = ["self_attn", "mlp"]


@dataclass(frozen=True)
class MoELoRAConfig:
    """Immutable MoE-LoRA inference configuration."""

    alpha: float = DEFAULT_ALPHA
    rank: int = DEFAULT_RANK
    top_k: int = DEFAULT_TOP_K
    num_experts: int = DEFAULT_NUM_EXPERTS
    router_hidden: int = DEFAULT_ROUTER_HIDDEN
    use_rs_lora: bool = False
    target_modules: tuple[str, ...] = tuple(PROJ_NAMES)

    @property
    def scaling(self) -> float:
        """LoRA scaling factor: alpha/sqrt(rank) for rsLoRA, alpha/rank otherwise."""
        if self.use_rs_lora:
            return self.alpha / math.sqrt(self.rank)
        return self.alpha / self.rank


# ---------------------------------------------------------------------------
# Per-projection MoE-LoRA state (backend-agnostic)
# ---------------------------------------------------------------------------

@dataclass
class MoELoRAProjection:
    """Holds the expert weights and router for one linear projection.

    All tensors are stored as the native backend type (mx.array or torch.Tensor).
    """

    # Expert LoRA matrices: lists of length num_experts
    # lora_a[i]: (in_dim, rank)
    # lora_b[i]: (rank, out_dim)
    lora_a: list[Any] = field(default_factory=list)
    lora_b: list[Any] = field(default_factory=list)

    # Router MLP weights
    # router_w1: (router_hidden, in_dim), router_b1: (router_hidden,)
    # router_w2: (num_experts, router_hidden), router_b2: (num_experts,)
    router_w1: Any = None
    router_b1: Any = None
    router_w2: Any = None
    router_b2: Any = None

    scaling: float = 2.0
    top_k: int = 2
    num_experts: int = 4


# ---------------------------------------------------------------------------
# Backend operations
# ---------------------------------------------------------------------------

def _to_float32(t: Any) -> Any:
    """Cast tensor to float32 for accumulation."""
    if _BACKEND == "mlx":
        return t.astype(mx.float32)
    elif _BACKEND == "torch":
        return t.float()
    raise RuntimeError(f"No backend available (detected: {_BACKEND})")


def _softmax(x: Any, axis: int = -1) -> Any:
    """Softmax along given axis."""
    if _BACKEND == "mlx":
        return mx.softmax(x, axis=axis)
    elif _BACKEND == "torch":
        return torch.softmax(x, dim=axis)
    raise RuntimeError(f"No backend available (detected: {_BACKEND})")


def _gelu(x: Any) -> Any:
    """GELU activation."""
    if _BACKEND == "mlx":
        return mlx_nn.gelu(x)
    elif _BACKEND == "torch":
        return torch.nn.functional.gelu(x)
    raise RuntimeError(f"No backend available (detected: {_BACKEND})")


def _argsort(x: Any, axis: int = -1) -> Any:
    """Argsort along axis."""
    if _BACKEND == "mlx":
        return mx.argsort(x, axis=axis)
    elif _BACKEND == "torch":
        return torch.argsort(x, dim=axis)
    raise RuntimeError(f"No backend available (detected: {_BACKEND})")


def _take_along_axis(x: Any, indices: Any, axis: int) -> Any:
    """Gather values along axis using indices."""
    if _BACKEND == "mlx":
        return mx.take_along_axis(x, indices, axis=axis)
    elif _BACKEND == "torch":
        return torch.gather(x, axis, indices)
    raise RuntimeError(f"No backend available (detected: {_BACKEND})")


def _expand_dims(x: Any, axis: int) -> Any:
    """Add dimension at axis."""
    if _BACKEND == "mlx":
        return mx.expand_dims(x, axis=axis)
    elif _BACKEND == "torch":
        return x.unsqueeze(axis)
    raise RuntimeError(f"No backend available (detected: {_BACKEND})")


def _broadcast_to(x: Any, shape: tuple[int, ...]) -> Any:
    """Broadcast tensor to shape."""
    if _BACKEND == "mlx":
        return mx.broadcast_to(x, shape)
    elif _BACKEND == "torch":
        return x.expand(shape)
    raise RuntimeError(f"No backend available (detected: {_BACKEND})")


def _stack(tensors: list[Any], axis: int = 0) -> Any:
    """Stack tensors along new axis."""
    if _BACKEND == "mlx":
        return mx.stack(tensors, axis=axis)
    elif _BACKEND == "torch":
        return torch.stack(tensors, dim=axis)
    raise RuntimeError(f"No backend available (detected: {_BACKEND})")


# ---------------------------------------------------------------------------
# MoE-LoRA forward pass (the core per-token routing logic)
# ---------------------------------------------------------------------------

def moe_lora_forward(
    x: Any,
    base_output: Any,
    proj: MoELoRAProjection,
) -> Any:
    """Apply MoE-LoRA delta to base linear output with per-token routing.

    For each token position, the router MLP computes expert logits, selects
    top-k, normalizes via softmax, then computes:
        delta = sum_topk(gate_i * (x @ A_i) @ B_i) * scaling

    Args:
        x: Input hidden state, shape (..., in_dim). Supports 2D or 3D.
        base_output: Output from frozen base linear, shape (..., out_dim).
        proj: MoELoRAProjection holding expert weights and router.

    Returns:
        base_output + delta, same shape as base_output.
    """
    needs_squeeze = False
    if x.ndim == 2:
        # (seq, in_dim) -> (1, seq, in_dim)
        needs_squeeze = True
        if _BACKEND == "mlx":
            x = mx.expand_dims(x, axis=0)
            base_output = mx.expand_dims(base_output, axis=0)
        elif _BACKEND == "torch":
            x = x.unsqueeze(0)
            base_output = base_output.unsqueeze(0)

    batch_size, seq_len, in_dim = x.shape

    # Router forward: x -> GELU(x @ W1.T + b1) -> logits @ W2.T + b2
    # router_w1: (router_hidden, in_dim), x: (B, T, in_dim)
    x_f32 = _to_float32(x)
    h = x_f32 @ _to_float32(proj.router_w1).T  # (B, T, router_hidden)
    if proj.router_b1 is not None:
        h = h + _to_float32(proj.router_b1)
    h = _gelu(h)

    logits = h @ _to_float32(proj.router_w2).T  # (B, T, num_experts)
    if proj.router_b2 is not None:
        logits = logits + _to_float32(proj.router_b2)

    # Top-k selection
    sorted_indices = _argsort(logits, axis=-1)
    # Take the last top_k indices (highest logits)
    if _BACKEND == "mlx":
        top_k_indices = sorted_indices[..., -proj.top_k:]  # (B, T, k)
    elif _BACKEND == "torch":
        top_k_indices = sorted_indices[..., -proj.top_k:]  # (B, T, k)

    top_k_logits = _take_along_axis(logits, top_k_indices, axis=-1)  # (B, T, k)
    weights = _softmax(top_k_logits, axis=-1)  # (B, T, k)

    # Compute all expert outputs: expert_i(x) = (x @ A_i) @ B_i * scaling
    # Each A_i: (in_dim, rank), B_i: (rank, out_dim)
    expert_outputs = []
    for i in range(proj.num_experts):
        a_i = _to_float32(proj.lora_a[i])  # (in_dim, rank)
        b_i = _to_float32(proj.lora_b[i])  # (rank, out_dim)
        # x_f32: (B, T, in_dim) @ (in_dim, rank) -> (B, T, rank)
        h_i = x_f32 @ a_i
        # (B, T, rank) @ (rank, out_dim) -> (B, T, out_dim)
        out_i = h_i @ b_i
        expert_outputs.append(out_i * proj.scaling)

    # Stack: (B, T, num_experts, out_dim)
    all_experts = _stack(expert_outputs, axis=-2)

    # Gather the top-k expert outputs
    out_dim = base_output.shape[-1]
    idx = _expand_dims(top_k_indices, axis=-1)  # (B, T, k, 1)
    idx = _broadcast_to(idx, (batch_size, seq_len, proj.top_k, out_dim))
    selected = _take_along_axis(all_experts, idx, axis=2)  # (B, T, k, out_dim)

    # Weighted sum over top-k experts
    w = _expand_dims(weights, axis=-1)  # (B, T, k, 1)
    if _BACKEND == "mlx":
        delta = mx.sum(selected * w, axis=2)  # (B, T, out_dim)
    elif _BACKEND == "torch":
        delta = torch.sum(selected * w, dim=2)  # (B, T, out_dim)

    # Cast delta to match base_output dtype
    if _BACKEND == "mlx":
        delta = delta.astype(base_output.dtype)
    elif _BACKEND == "torch":
        delta = delta.to(base_output.dtype)

    result = base_output + delta

    if needs_squeeze:
        if _BACKEND == "mlx":
            result = result.squeeze(axis=0)
        elif _BACKEND == "torch":
            result = result.squeeze(0)

    return result


# ---------------------------------------------------------------------------
# Adapter loading
# ---------------------------------------------------------------------------

def _find_adapter_prefixes(
    adapter_keys: set[str],
) -> dict[tuple[int, str, str], str]:
    """Scan adapter keys and return {(layer_idx, sub_module, proj_name): prefix}.

    Detects both naming conventions:
      language_model.model.layers.{L}.{sub}.{proj}_moe_lora.experts.0.lora_a
      model.layers.{L}.{sub}.{proj}_moe_lora.experts.0.lora_a

    Returns:
        Mapping from (layer_idx, sub_name, proj_name) to adapter key prefix.
    """
    mapping: dict[tuple[int, str, str], str] = {}

    for key in adapter_keys:
        if ".experts.0.lora_a" not in key:
            continue

        # Extract the prefix before .experts.0.lora_a
        prefix = key.replace(".experts.0.lora_a", "")

        # Parse layer index, sub_module, and projection name
        # prefix: ...layers.{L}.{sub}.{proj}_moe_lora
        # or: ...layers.{L}.{proj}_moe_lora (flat layout)
        parts = prefix.split(".")

        # Find "layers" and extract layer index
        try:
            layers_idx = parts.index("layers")
        except ValueError:
            continue

        layer_num = int(parts[layers_idx + 1])

        # The suffix after layers.{N} is: sub.proj_moe_lora or proj_moe_lora
        suffix_parts = parts[layers_idx + 2:]
        suffix = ".".join(suffix_parts)

        # Match against known projections in sub_modules
        for sub in SUB_MODULES:
            for proj in PROJ_NAMES:
                if suffix == f"{sub}.{proj}_moe_lora":
                    mapping[(layer_num, sub, proj)] = prefix
                    break
                # Flat layout: proj_moe_lora (no sub-module prefix)
                if suffix == f"{proj}_moe_lora":
                    # Infer sub from proj name
                    if proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                        inferred_sub = "self_attn"
                    else:
                        inferred_sub = "mlp"
                    mapping[(layer_num, inferred_sub, proj)] = prefix
                    break

    return mapping


def load_adapter_projections(
    adapter_path: str | Path,
    config: MoELoRAConfig,
) -> dict[tuple[int, str, str], MoELoRAProjection]:
    """Load adapter safetensors and parse into MoELoRAProjection objects.

    Args:
        adapter_path: Path to directory containing adapters.safetensors.
        config: MoE-LoRA configuration.

    Returns:
        Mapping from (layer_idx, sub_name, proj_name) to MoELoRAProjection.
    """
    adapter_path = Path(adapter_path)
    safetensors_file = adapter_path / "adapters.safetensors"
    if not safetensors_file.exists():
        raise FileNotFoundError(f"Adapter file not found: {safetensors_file}")

    # Load tensors
    if _BACKEND == "mlx":
        tensors = mx.load(str(safetensors_file))
    elif _BACKEND == "torch":
        from safetensors.torch import load_file
        tensors = load_file(str(safetensors_file))
    else:
        raise RuntimeError("No backend available for loading tensors")

    adapter_keys = set(tensors.keys())
    prefixes = _find_adapter_prefixes(adapter_keys)

    logger.info(
        "Found %d MoE-LoRA projections in %s", len(prefixes), safetensors_file.name
    )

    projections: dict[tuple[int, str, str], MoELoRAProjection] = {}

    for (layer_idx, sub, proj), prefix in prefixes.items():
        # Load expert weights
        lora_a_list = []
        lora_b_list = []
        for e in range(config.num_experts):
            a_key = f"{prefix}.experts.{e}.lora_a"
            b_key = f"{prefix}.experts.{e}.lora_b"
            if a_key not in tensors or b_key not in tensors:
                raise KeyError(
                    f"Missing expert {e} for {prefix}: "
                    f"need {a_key} and {b_key}"
                )
            lora_a_list.append(tensors[a_key])
            lora_b_list.append(tensors[b_key])

        # Load router weights
        def _get(suffix: str) -> Any:
            key = f"{prefix}.{suffix}"
            return tensors.get(key)

        proj_state = MoELoRAProjection(
            lora_a=lora_a_list,
            lora_b=lora_b_list,
            router_w1=_get("router_w1.weight"),
            router_b1=_get("router_w1.bias"),
            router_w2=_get("router_w2.weight"),
            router_b2=_get("router_w2.bias"),
            scaling=config.scaling,
            top_k=config.top_k,
            num_experts=config.num_experts,
        )

        if proj_state.router_w1 is None or proj_state.router_w2 is None:
            raise KeyError(
                f"Missing router weights for {prefix}: "
                f"need router_w1.weight and router_w2.weight"
            )

        projections[(layer_idx, sub, proj)] = proj_state

    return projections


# ---------------------------------------------------------------------------
# Model patching (MLX)
# ---------------------------------------------------------------------------

class _MoELoRAPatchedLinear:
    """Replaces an nn.Linear with a version that adds MoE-LoRA delta per token.

    MLX resolves __call__ through the class MRO, so we replace the module
    attribute on the parent with this wrapper. For PyTorch we use a similar
    approach via __call__ override.
    """

    def __init__(self, base_linear: Any, projection: MoELoRAProjection) -> None:
        self._base = base_linear
        self._proj = projection
        # Forward attribute access for .weight, .bias, etc.
        if hasattr(base_linear, "weight"):
            self.weight = base_linear.weight
        if hasattr(base_linear, "bias"):
            self.bias = base_linear.bias

    def __call__(self, x: Any) -> Any:
        base_out = self._base(x)
        return moe_lora_forward(x, base_out, self._proj)

    def __getattr__(self, name: str) -> Any:
        # Delegate attribute access not found here to the base linear
        return getattr(self._base, name)

    @property
    def is_moe_lora_patched(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Main runtime class
# ---------------------------------------------------------------------------

class MoELoRARuntime:
    """Runtime MoE-LoRA inference engine with per-token expert routing.

    Loads a base model, patches linear layers with MoE-LoRA wrappers that
    perform dynamic top-k expert routing at each token position.

    Supports adapter hot-swap: unpatching the current adapter and loading
    a new one without reloading the base model.

    Usage::

        runtime = MoELoRARuntime()
        runtime.load_base_model("models/qwen3.5-4b")
        runtime.load_adapter("outputs/stacks/stack-01-kicad")
        # Model is now patched — generate normally
        output = runtime.generate("Design a 3.3V LDO circuit")
        # Hot-swap to another domain
        runtime.load_adapter("outputs/stacks/stack-05-embedded")
    """

    def __init__(self, config: MoELoRAConfig | None = None) -> None:
        self._config = config or MoELoRAConfig()
        self._model: Any = None
        self._tokenizer: Any = None
        self._base_model_path: str | None = None
        self._current_adapter: str | None = None
        self._patched_count: int = 0
        self._projections: dict[tuple[int, str, str], MoELoRAProjection] = {}

    @property
    def config(self) -> MoELoRAConfig:
        return self._config

    @property
    def backend(self) -> str:
        return _BACKEND

    @property
    def model(self) -> Any:
        return self._model

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @property
    def current_adapter(self) -> str | None:
        return self._current_adapter

    @property
    def patched_count(self) -> int:
        return self._patched_count

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def is_patched(self) -> bool:
        return self._patched_count > 0

    def load_base_model(self, model_path: str | Path) -> None:
        """Load the base model and tokenizer.

        Auto-detects MLX or PyTorch backend. For MLX, uses mlx_lm.load.
        For PyTorch, uses transformers AutoModelForCausalLM.

        Args:
            model_path: Path to HuggingFace model directory.
        """
        model_path = str(model_path)
        start = time.monotonic()

        if _BACKEND == "mlx":
            from mlx_lm import load as mlx_load
            self._model, self._tokenizer = mlx_load(model_path)
        elif _BACKEND == "torch":
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self._model.eval()
        else:
            raise RuntimeError("No backend (mlx or torch) available")

        self._base_model_path = model_path
        elapsed = time.monotonic() - start
        logger.info("Loaded base model from %s in %.1fs (%s)", model_path, elapsed, _BACKEND)

    def load_base_model_from_objects(self, model: Any, tokenizer: Any) -> None:
        """Attach an already-loaded model and tokenizer.

        Useful for testing or when the model is loaded externally.

        Args:
            model: Pre-loaded model object.
            tokenizer: Pre-loaded tokenizer object.
        """
        self._model = model
        self._tokenizer = tokenizer
        self._base_model_path = "<external>"
        logger.info("Attached external model and tokenizer")

    def load_adapter(self, adapter_path: str | Path) -> int:
        """Load a MoE-LoRA adapter and patch the model.

        If another adapter is already loaded, it is unpatched first (hot-swap).

        Args:
            adapter_path: Path to adapter directory containing adapters.safetensors.

        Returns:
            Number of linear projections patched.
        """
        if self._model is None:
            raise RuntimeError("Base model not loaded — call load_base_model() first")

        adapter_path = Path(adapter_path)

        # Hot-swap: unpatch current adapter if loaded
        if self._current_adapter is not None:
            self.unpatch()

        start = time.monotonic()
        self._projections = load_adapter_projections(adapter_path, self._config)
        self._patched_count = self._patch_model()
        self._current_adapter = str(adapter_path)
        elapsed = time.monotonic() - start

        logger.info(
            "Loaded adapter %s: %d projections patched in %.1fms",
            adapter_path.name,
            self._patched_count,
            elapsed * 1000,
        )
        return self._patched_count

    def load_adapter_from_projections(
        self,
        projections: dict[tuple[int, str, str], MoELoRAProjection],
        adapter_id: str = "<manual>",
    ) -> int:
        """Patch the model with pre-built MoELoRAProjection objects.

        Useful for testing with synthetic weights.

        Args:
            projections: Mapping from (layer_idx, sub, proj) to projection.
            adapter_id: Identifier string for the adapter.

        Returns:
            Number of linear projections patched.
        """
        if self._model is None:
            raise RuntimeError("Base model not loaded — call load_base_model() first")

        if self._current_adapter is not None:
            self.unpatch()

        self._projections = projections
        self._patched_count = self._patch_model()
        self._current_adapter = adapter_id
        return self._patched_count

    def unpatch(self) -> int:
        """Remove MoE-LoRA patches, restoring original linear layers.

        Returns:
            Number of projections unpatched.
        """
        if self._model is None or self._patched_count == 0:
            return 0

        count = self._unpatch_model()
        self._current_adapter = None
        self._patched_count = 0
        self._projections = {}

        logger.info("Unpatched %d projections", count)
        return count

    def generate(self, prompt: str, max_tokens: int = 512, **kwargs: Any) -> str:
        """Generate text with the patched model.

        Args:
            prompt: Input text.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional generation parameters.

        Returns:
            Generated text string.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded")
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")

        if _BACKEND == "mlx":
            from mlx_lm import generate as mlx_generate
            return mlx_generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                **kwargs,
            )
        elif _BACKEND == "torch":
            inputs = self._tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    **kwargs,
                )
            # Decode only the new tokens
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            return self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        else:
            raise RuntimeError("No backend available")

    # ------------------------------------------------------------------
    # Internal patching
    # ------------------------------------------------------------------

    def _get_layers(self) -> Any:
        """Navigate model hierarchy to find the transformer layers list."""
        model = self._model
        if hasattr(model, "language_model"):
            model = model.language_model
        if hasattr(model, "model"):
            model = model.model
        if hasattr(model, "layers"):
            return model.layers
        raise AttributeError(
            "Cannot find layers in model — expected model.layers, "
            "model.model.layers, or model.language_model.model.layers"
        )

    def _patch_model(self) -> int:
        """Patch linear projections with MoE-LoRA wrappers.

        Returns:
            Number of projections patched.
        """
        layers = self._get_layers()
        patched = 0

        for (layer_idx, sub_name, proj_name), proj_state in self._projections.items():
            if layer_idx >= len(layers):
                logger.warning(
                    "Layer %d not found (model has %d layers), skipping",
                    layer_idx, len(layers),
                )
                continue

            layer = layers[layer_idx]
            sub = getattr(layer, sub_name, None)
            if sub is None:
                logger.warning(
                    "Sub-module %s not found in layer %d, skipping",
                    sub_name, layer_idx,
                )
                continue

            base_linear = getattr(sub, proj_name, None)
            if base_linear is None:
                logger.warning(
                    "Projection %s not found in layer %d.%s, skipping",
                    proj_name, layer_idx, sub_name,
                )
                continue

            # Skip if already patched
            if isinstance(base_linear, _MoELoRAPatchedLinear):
                continue

            wrapped = _MoELoRAPatchedLinear(base_linear, proj_state)
            setattr(sub, proj_name, wrapped)
            patched += 1

        return patched

    def _unpatch_model(self) -> int:
        """Restore original linear layers by unwrapping patches.

        Returns:
            Number of projections unpatched.
        """
        layers = self._get_layers()
        unpatched = 0

        for layer in layers:
            for sub_name in SUB_MODULES:
                sub = getattr(layer, sub_name, None)
                if sub is None:
                    continue
                for proj_name in self._config.target_modules:
                    proj = getattr(sub, proj_name, None)
                    if isinstance(proj, _MoELoRAPatchedLinear):
                        setattr(sub, proj_name, proj._base)
                        unpatched += 1

        return unpatched

    def info(self) -> dict[str, Any]:
        """Return runtime state summary."""
        return {
            "backend": _BACKEND,
            "base_model": self._base_model_path,
            "adapter": self._current_adapter,
            "patched_projections": self._patched_count,
            "config": {
                "alpha": self._config.alpha,
                "rank": self._config.rank,
                "top_k": self._config.top_k,
                "num_experts": self._config.num_experts,
                "scaling": self._config.scaling,
            },
        }

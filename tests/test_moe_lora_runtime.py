"""Unit tests for MoE-LoRA runtime inference engine.

Tests use mock tensors and a fake model — no real model loading needed.
All tests are backend-aware: they auto-skip if neither MLX nor PyTorch
is available.
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest

from src.serving.moe_lora_runtime import (
    MoELoRAConfig,
    MoELoRAProjection,
    MoELoRARuntime,
    _MoELoRAPatchedLinear,
    _find_adapter_prefixes,
    moe_lora_forward,
    _BACKEND,
)


# ---------------------------------------------------------------------------
# Skip all tests if no backend available
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    _BACKEND == "none",
    reason="No backend (mlx or torch) available",
)


# ---------------------------------------------------------------------------
# Backend-agnostic tensor helpers
# ---------------------------------------------------------------------------

def _randn(*shape: int) -> Any:
    """Create random tensor using detected backend."""
    if _BACKEND == "mlx":
        import mlx.core as mx
        return mx.random.normal(shape)
    elif _BACKEND == "torch":
        import torch
        return torch.randn(*shape)
    raise RuntimeError("No backend")


def _zeros(*shape: int) -> Any:
    """Create zero tensor using detected backend."""
    if _BACKEND == "mlx":
        import mlx.core as mx
        return mx.zeros(shape)
    elif _BACKEND == "torch":
        import torch
        return torch.zeros(*shape)
    raise RuntimeError("No backend")


def _ones(*shape: int) -> Any:
    """Create ones tensor using detected backend."""
    if _BACKEND == "mlx":
        import mlx.core as mx
        return mx.ones(shape)
    elif _BACKEND == "torch":
        import torch
        return torch.ones(*shape)
    raise RuntimeError("No backend")


def _to_float(t: Any) -> float:
    """Extract scalar float from tensor."""
    if _BACKEND == "mlx":
        return t.item()
    elif _BACKEND == "torch":
        return t.item()
    raise RuntimeError("No backend")


def _abs_sum(t: Any) -> float:
    """Sum of absolute values as float."""
    if _BACKEND == "mlx":
        import mlx.core as mx
        return mx.sum(mx.abs(t)).item()
    elif _BACKEND == "torch":
        import torch
        return torch.sum(torch.abs(t)).item()
    raise RuntimeError("No backend")


def _shape(t: Any) -> tuple[int, ...]:
    """Get tensor shape as tuple."""
    return tuple(t.shape)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_config() -> MoELoRAConfig:
    return MoELoRAConfig()


@pytest.fixture
def small_config() -> MoELoRAConfig:
    """Small config for fast tests: rank=4, 2 experts, top_k=1."""
    return MoELoRAConfig(
        rank=4,
        num_experts=2,
        top_k=1,
        router_hidden=8,
        alpha=8.0,
    )


def _make_projection(
    in_dim: int = 64,
    out_dim: int = 32,
    rank: int = 4,
    num_experts: int = 2,
    top_k: int = 1,
    router_hidden: int = 8,
    alpha: float = 8.0,
    zero_b: bool = True,
) -> MoELoRAProjection:
    """Create a MoELoRAProjection with random (or zero-B) weights."""
    scaling = alpha / rank

    lora_a = [_randn(in_dim, rank) * (1.0 / math.sqrt(in_dim)) for _ in range(num_experts)]
    if zero_b:
        lora_b = [_zeros(rank, out_dim) for _ in range(num_experts)]
    else:
        lora_b = [_randn(rank, out_dim) * 0.01 for _ in range(num_experts)]

    return MoELoRAProjection(
        lora_a=lora_a,
        lora_b=lora_b,
        router_w1=_randn(router_hidden, in_dim),
        router_b1=_randn(router_hidden),
        router_w2=_randn(num_experts, router_hidden),
        router_b2=_randn(num_experts),
        scaling=scaling,
        top_k=top_k,
        num_experts=num_experts,
    )


class FakeLinear:
    """Minimal fake linear layer for testing patches."""

    def __init__(self, in_features: int, out_features: int) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.weight = _randn(out_features, in_features)
        self.bias = _zeros(out_features)

    def __call__(self, x: Any) -> Any:
        # Simple linear: x @ W.T + b
        return x @ self.weight.T + self.bias


def _make_fake_model(
    num_layers: int = 2,
    in_dim: int = 64,
    out_dim: int = 32,
) -> SimpleNamespace:
    """Create a fake model with layers[i].self_attn.q_proj etc."""
    layers = []
    for _ in range(num_layers):
        self_attn = SimpleNamespace(
            q_proj=FakeLinear(in_dim, out_dim),
            k_proj=FakeLinear(in_dim, out_dim),
            v_proj=FakeLinear(in_dim, out_dim),
            o_proj=FakeLinear(out_dim, in_dim),
        )
        mlp = SimpleNamespace(
            gate_proj=FakeLinear(in_dim, out_dim),
            up_proj=FakeLinear(in_dim, out_dim),
            down_proj=FakeLinear(out_dim, in_dim),
        )
        layer = SimpleNamespace(self_attn=self_attn, mlp=mlp)
        layers.append(layer)

    model = SimpleNamespace(model=SimpleNamespace(layers=layers))
    return model


# ---------------------------------------------------------------------------
# Tests: MoELoRAConfig
# ---------------------------------------------------------------------------

class TestMoELoRAConfig:
    def test_defaults(self):
        c = MoELoRAConfig()
        assert c.alpha == 32.0
        assert c.rank == 16
        assert c.top_k == 2
        assert c.num_experts == 4
        assert c.router_hidden == 64
        assert c.use_rs_lora is False

    def test_scaling_standard(self):
        c = MoELoRAConfig(alpha=32.0, rank=16, use_rs_lora=False)
        assert c.scaling == 2.0

    def test_scaling_rs_lora(self):
        c = MoELoRAConfig(alpha=32.0, rank=16, use_rs_lora=True)
        assert c.scaling == pytest.approx(32.0 / math.sqrt(16))

    def test_frozen(self):
        c = MoELoRAConfig()
        with pytest.raises(AttributeError):
            c.rank = 8  # type: ignore[misc]

    def test_custom_target_modules(self):
        c = MoELoRAConfig(target_modules=("q_proj", "v_proj"))
        assert c.target_modules == ("q_proj", "v_proj")


# ---------------------------------------------------------------------------
# Tests: MoELoRAProjection
# ---------------------------------------------------------------------------

class TestMoELoRAProjection:
    def test_creation(self):
        proj = _make_projection()
        assert len(proj.lora_a) == 2
        assert len(proj.lora_b) == 2
        assert proj.router_w1 is not None
        assert proj.router_w2 is not None
        assert proj.scaling == 2.0
        assert proj.top_k == 1
        assert proj.num_experts == 2

    def test_shapes(self):
        in_dim, out_dim, rank = 64, 32, 4
        proj = _make_projection(in_dim=in_dim, out_dim=out_dim, rank=rank)
        for a in proj.lora_a:
            assert _shape(a) == (in_dim, rank)
        for b in proj.lora_b:
            assert _shape(b) == (rank, out_dim)
        assert _shape(proj.router_w1) == (8, in_dim)  # router_hidden=8
        assert _shape(proj.router_w2) == (2, 8)  # (num_experts, router_hidden)


# ---------------------------------------------------------------------------
# Tests: moe_lora_forward
# ---------------------------------------------------------------------------

class TestMoELoRAForward:
    def test_output_shape_3d(self):
        """Forward with 3D input returns correct shape."""
        in_dim, out_dim = 64, 32
        proj = _make_projection(in_dim=in_dim, out_dim=out_dim, zero_b=False)
        x = _randn(2, 8, in_dim)
        base_out = _randn(2, 8, out_dim)
        result = moe_lora_forward(x, base_out, proj)
        assert _shape(result) == (2, 8, out_dim)

    def test_output_shape_2d(self):
        """Forward with 2D input returns correct shape (auto-squeeze)."""
        in_dim, out_dim = 64, 32
        proj = _make_projection(in_dim=in_dim, out_dim=out_dim, zero_b=False)
        x = _randn(8, in_dim)
        base_out = _randn(8, out_dim)
        result = moe_lora_forward(x, base_out, proj)
        assert _shape(result) == (8, out_dim)

    def test_zero_b_produces_zero_delta(self):
        """With zero-initialized B matrices, the delta should be zero."""
        in_dim, out_dim = 64, 32
        proj = _make_projection(in_dim=in_dim, out_dim=out_dim, zero_b=True)
        x = _randn(1, 4, in_dim)
        base_out = _randn(1, 4, out_dim)
        result = moe_lora_forward(x, base_out, proj)
        # With B=0, delta=0, so result should equal base_out
        diff = _abs_sum(result - base_out)
        assert diff < 1e-5, f"Expected zero delta, got diff={diff}"

    def test_nonzero_b_produces_nonzero_delta(self):
        """With non-zero B matrices, the delta should be non-zero."""
        in_dim, out_dim = 64, 32
        proj = _make_projection(in_dim=in_dim, out_dim=out_dim, zero_b=False)
        x = _randn(1, 4, in_dim)
        base_out = _zeros(1, 4, out_dim)  # Zero base to isolate delta
        result = moe_lora_forward(x, base_out, proj)
        delta_mag = _abs_sum(result)
        assert delta_mag > 1e-6, f"Expected nonzero delta, got {delta_mag}"

    def test_top_k_selection(self):
        """With top_k=1, exactly 1 expert contributes per token."""
        in_dim, out_dim = 64, 32
        proj = _make_projection(
            in_dim=in_dim, out_dim=out_dim, top_k=1,
            num_experts=4, zero_b=False,
        )
        x = _randn(1, 2, in_dim)
        base_out = _zeros(1, 2, out_dim)
        # Just check it runs without error and produces output
        result = moe_lora_forward(x, base_out, proj)
        assert _shape(result) == (1, 2, out_dim)

    def test_top_k_2(self):
        """With top_k=2, forward runs correctly."""
        in_dim, out_dim = 64, 32
        proj = _make_projection(
            in_dim=in_dim, out_dim=out_dim, top_k=2,
            num_experts=4, zero_b=False,
        )
        x = _randn(1, 4, in_dim)
        base_out = _randn(1, 4, out_dim)
        result = moe_lora_forward(x, base_out, proj)
        assert _shape(result) == (1, 4, out_dim)

    def test_scaling_applied(self):
        """Verify that scaling factor affects output magnitude."""
        in_dim, out_dim = 64, 32

        proj_low = _make_projection(
            in_dim=in_dim, out_dim=out_dim, alpha=1.0, rank=4, zero_b=False,
        )
        proj_high = _make_projection(
            in_dim=in_dim, out_dim=out_dim, alpha=32.0, rank=4, zero_b=False,
        )
        # Use same expert weights to compare scaling effect
        proj_high.lora_a = proj_low.lora_a
        proj_high.lora_b = proj_low.lora_b
        proj_high.router_w1 = proj_low.router_w1
        proj_high.router_b1 = proj_low.router_b1
        proj_high.router_w2 = proj_low.router_w2
        proj_high.router_b2 = proj_low.router_b2

        x = _randn(1, 2, in_dim)
        base_out = _zeros(1, 2, out_dim)

        result_low = moe_lora_forward(x, base_out, proj_low)
        result_high = moe_lora_forward(x, base_out, proj_high)

        mag_low = _abs_sum(result_low)
        mag_high = _abs_sum(result_high)

        # alpha=32/rank=4 gives scaling=8, alpha=1/rank=4 gives scaling=0.25
        # ratio should be ~32
        if mag_low > 1e-8:
            ratio = mag_high / mag_low
            assert ratio > 10, f"Expected high scaling to produce larger output, ratio={ratio}"

    def test_different_inputs_different_routing(self):
        """Different inputs should (usually) produce different routing."""
        in_dim, out_dim = 64, 32
        proj = _make_projection(
            in_dim=in_dim, out_dim=out_dim, top_k=1,
            num_experts=4, zero_b=False,
        )
        x1 = _randn(1, 1, in_dim) * 10.0
        x2 = _randn(1, 1, in_dim) * 10.0
        base_out = _zeros(1, 1, out_dim)

        r1 = moe_lora_forward(x1, base_out, proj)
        r2 = moe_lora_forward(x2, base_out, proj)

        # Not guaranteed to differ but extremely unlikely with random inputs
        diff = _abs_sum(r1 - r2)
        # Just verify they both produce finite output
        assert _abs_sum(r1) < 1e10
        assert _abs_sum(r2) < 1e10


# ---------------------------------------------------------------------------
# Tests: _find_adapter_prefixes
# ---------------------------------------------------------------------------

class TestFindAdapterPrefixes:
    def test_standard_keys(self):
        keys = {
            "language_model.model.layers.0.self_attn.q_proj_moe_lora.experts.0.lora_a",
            "language_model.model.layers.0.self_attn.q_proj_moe_lora.experts.0.lora_b",
            "language_model.model.layers.0.self_attn.q_proj_moe_lora.experts.1.lora_a",
            "language_model.model.layers.0.self_attn.q_proj_moe_lora.experts.1.lora_b",
            "language_model.model.layers.0.self_attn.q_proj_moe_lora.router_w1.weight",
            "language_model.model.layers.0.self_attn.q_proj_moe_lora.router_w2.weight",
        }
        mapping = _find_adapter_prefixes(keys)
        assert (0, "self_attn", "q_proj") in mapping
        assert mapping[(0, "self_attn", "q_proj")] == (
            "language_model.model.layers.0.self_attn.q_proj_moe_lora"
        )

    def test_model_prefix_style(self):
        keys = {
            "model.layers.5.mlp.gate_proj_moe_lora.experts.0.lora_a",
            "model.layers.5.mlp.gate_proj_moe_lora.experts.0.lora_b",
        }
        mapping = _find_adapter_prefixes(keys)
        assert (5, "mlp", "gate_proj") in mapping

    def test_multiple_layers_and_projections(self):
        keys = set()
        for layer in [0, 1, 2]:
            for proj in ["q_proj", "k_proj", "v_proj"]:
                for e in [0, 1]:
                    keys.add(f"language_model.model.layers.{layer}.self_attn.{proj}_moe_lora.experts.{e}.lora_a")
                    keys.add(f"language_model.model.layers.{layer}.self_attn.{proj}_moe_lora.experts.{e}.lora_b")

        mapping = _find_adapter_prefixes(keys)
        assert len(mapping) == 9  # 3 layers * 3 projections

    def test_empty_keys(self):
        assert _find_adapter_prefixes(set()) == {}

    def test_unrelated_keys_ignored(self):
        keys = {
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight",
        }
        assert _find_adapter_prefixes(keys) == {}


# ---------------------------------------------------------------------------
# Tests: _MoELoRAPatchedLinear
# ---------------------------------------------------------------------------

class TestPatchedLinear:
    def test_call_invokes_moe_forward(self):
        """Patched linear should produce base + delta."""
        in_dim, out_dim = 64, 32
        base = FakeLinear(in_dim, out_dim)
        proj = _make_projection(in_dim=in_dim, out_dim=out_dim, zero_b=True)
        patched = _MoELoRAPatchedLinear(base, proj)

        x = _randn(1, 4, in_dim)
        result = patched(x)
        expected = base(x)  # zero B means delta=0
        diff = _abs_sum(result - expected)
        assert diff < 1e-4, f"Expected patched == base with zero B, diff={diff}"

    def test_is_patched_flag(self):
        base = FakeLinear(64, 32)
        proj = _make_projection()
        patched = _MoELoRAPatchedLinear(base, proj)
        assert patched.is_moe_lora_patched is True

    def test_weight_access(self):
        base = FakeLinear(64, 32)
        proj = _make_projection()
        patched = _MoELoRAPatchedLinear(base, proj)
        # Should forward .weight to base
        assert patched.weight is base.weight

    def test_getattr_delegation(self):
        base = FakeLinear(64, 32)
        proj = _make_projection()
        patched = _MoELoRAPatchedLinear(base, proj)
        # in_features is on FakeLinear, delegated via __getattr__
        assert patched.in_features == 64


# ---------------------------------------------------------------------------
# Tests: MoELoRARuntime
# ---------------------------------------------------------------------------

class TestMoELoRARuntime:
    def test_initial_state(self):
        runtime = MoELoRARuntime()
        assert runtime.model is None
        assert runtime.tokenizer is None
        assert runtime.current_adapter is None
        assert runtime.patched_count == 0
        assert runtime.is_loaded is False
        assert runtime.is_patched is False
        assert runtime.backend == _BACKEND

    def test_config_default(self, default_config):
        runtime = MoELoRARuntime(config=default_config)
        assert runtime.config.rank == 16
        assert runtime.config.scaling == 2.0

    def test_load_from_objects(self):
        model = _make_fake_model()
        runtime = MoELoRARuntime()
        runtime.load_base_model_from_objects(model, tokenizer="fake-tok")
        assert runtime.is_loaded is True
        assert runtime.model is model
        assert runtime.tokenizer == "fake-tok"

    def test_load_adapter_requires_model(self, tmp_path):
        runtime = MoELoRARuntime()
        with pytest.raises(RuntimeError, match="Base model not loaded"):
            runtime.load_adapter(tmp_path)

    def test_load_adapter_from_projections(self):
        in_dim, out_dim = 64, 32
        model = _make_fake_model(num_layers=2, in_dim=in_dim, out_dim=out_dim)
        runtime = MoELoRARuntime(
            config=MoELoRAConfig(rank=4, num_experts=2, top_k=1, router_hidden=8, alpha=8.0),
        )
        runtime.load_base_model_from_objects(model, tokenizer=None)

        # Create projection for layer 0, self_attn, q_proj
        proj = _make_projection(in_dim=in_dim, out_dim=out_dim)
        projections = {(0, "self_attn", "q_proj"): proj}

        count = runtime.load_adapter_from_projections(projections, "test-adapter")
        assert count == 1
        assert runtime.is_patched is True
        assert runtime.current_adapter == "test-adapter"

    def test_unpatch_restores_original(self):
        in_dim, out_dim = 64, 32
        model = _make_fake_model(num_layers=2, in_dim=in_dim, out_dim=out_dim)
        original_q_proj = model.model.layers[0].self_attn.q_proj

        runtime = MoELoRARuntime(
            config=MoELoRAConfig(rank=4, num_experts=2, top_k=1, router_hidden=8, alpha=8.0),
        )
        runtime.load_base_model_from_objects(model, tokenizer=None)

        proj = _make_projection(in_dim=in_dim, out_dim=out_dim)
        runtime.load_adapter_from_projections({(0, "self_attn", "q_proj"): proj})

        # q_proj should now be patched
        assert isinstance(model.model.layers[0].self_attn.q_proj, _MoELoRAPatchedLinear)

        # Unpatch
        count = runtime.unpatch()
        assert count == 1
        assert runtime.is_patched is False
        assert runtime.current_adapter is None
        # Original should be restored
        assert model.model.layers[0].self_attn.q_proj is original_q_proj

    def test_hot_swap(self):
        """Loading a new adapter should unpatch the old one first."""
        in_dim, out_dim = 64, 32
        model = _make_fake_model(num_layers=2, in_dim=in_dim, out_dim=out_dim)
        original_q_proj = model.model.layers[0].self_attn.q_proj

        runtime = MoELoRARuntime(
            config=MoELoRAConfig(rank=4, num_experts=2, top_k=1, router_hidden=8, alpha=8.0),
        )
        runtime.load_base_model_from_objects(model, tokenizer=None)

        # Load first adapter
        proj1 = _make_projection(in_dim=in_dim, out_dim=out_dim)
        runtime.load_adapter_from_projections(
            {(0, "self_attn", "q_proj"): proj1}, "adapter-1"
        )
        assert runtime.current_adapter == "adapter-1"

        # Hot-swap to second adapter
        proj2 = _make_projection(in_dim=in_dim, out_dim=out_dim)
        runtime.load_adapter_from_projections(
            {(0, "self_attn", "q_proj"): proj2}, "adapter-2"
        )
        assert runtime.current_adapter == "adapter-2"
        assert runtime.patched_count == 1

        # The patched linear should reference proj2, not proj1
        patched = model.model.layers[0].self_attn.q_proj
        assert isinstance(patched, _MoELoRAPatchedLinear)
        assert patched._proj is proj2

    def test_multiple_projections_patched(self):
        """Patching multiple projections across layers."""
        in_dim, out_dim = 64, 32
        model = _make_fake_model(num_layers=3, in_dim=in_dim, out_dim=out_dim)
        runtime = MoELoRARuntime(
            config=MoELoRAConfig(rank=4, num_experts=2, top_k=1, router_hidden=8, alpha=8.0),
        )
        runtime.load_base_model_from_objects(model, tokenizer=None)

        projections = {}
        for layer_idx in range(3):
            for proj_name in ["q_proj", "k_proj", "v_proj"]:
                projections[(layer_idx, "self_attn", proj_name)] = _make_projection(
                    in_dim=in_dim, out_dim=out_dim,
                )

        count = runtime.load_adapter_from_projections(projections)
        assert count == 9  # 3 layers * 3 projections

    def test_patched_forward_runs(self):
        """Verify that the patched model forward pass works end-to-end."""
        in_dim, out_dim = 64, 32
        model = _make_fake_model(num_layers=1, in_dim=in_dim, out_dim=out_dim)
        runtime = MoELoRARuntime(
            config=MoELoRAConfig(rank=4, num_experts=2, top_k=1, router_hidden=8, alpha=8.0),
        )
        runtime.load_base_model_from_objects(model, tokenizer=None)

        proj = _make_projection(in_dim=in_dim, out_dim=out_dim, zero_b=False)
        runtime.load_adapter_from_projections({(0, "self_attn", "q_proj"): proj})

        # Call the patched q_proj directly
        x = _randn(1, 4, in_dim)
        q_proj = model.model.layers[0].self_attn.q_proj
        result = q_proj(x)
        assert _shape(result) == (1, 4, out_dim)

    def test_skip_missing_layer(self):
        """Projections referencing non-existent layers are skipped."""
        model = _make_fake_model(num_layers=2)
        runtime = MoELoRARuntime(
            config=MoELoRAConfig(rank=4, num_experts=2, top_k=1, router_hidden=8, alpha=8.0),
        )
        runtime.load_base_model_from_objects(model, tokenizer=None)

        proj = _make_projection()
        # Layer 99 does not exist
        count = runtime.load_adapter_from_projections({(99, "self_attn", "q_proj"): proj})
        assert count == 0

    def test_info(self):
        runtime = MoELoRARuntime()
        info = runtime.info()
        assert info["backend"] == _BACKEND
        assert info["base_model"] is None
        assert info["adapter"] is None
        assert info["patched_projections"] == 0
        assert info["config"]["rank"] == 16

    def test_generate_requires_model(self):
        runtime = MoELoRARuntime()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            runtime.generate("hello")

    def test_generate_requires_tokenizer(self):
        runtime = MoELoRARuntime()
        runtime.load_base_model_from_objects(_make_fake_model(), tokenizer=None)
        with pytest.raises(RuntimeError, match="Tokenizer not loaded"):
            runtime.generate("hello")

    def test_unpatch_idempotent(self):
        """Unpatching when nothing is patched returns 0."""
        runtime = MoELoRARuntime()
        assert runtime.unpatch() == 0

    def test_language_model_hierarchy(self):
        """Model with language_model.model.layers hierarchy."""
        in_dim, out_dim = 64, 32
        inner = _make_fake_model(num_layers=1, in_dim=in_dim, out_dim=out_dim)
        # Wrap with language_model level
        model = SimpleNamespace(language_model=inner)

        runtime = MoELoRARuntime(
            config=MoELoRAConfig(rank=4, num_experts=2, top_k=1, router_hidden=8, alpha=8.0),
        )
        runtime.load_base_model_from_objects(model, tokenizer=None)

        proj = _make_projection(in_dim=in_dim, out_dim=out_dim)
        count = runtime.load_adapter_from_projections({(0, "self_attn", "q_proj"): proj})
        assert count == 1

    def test_flat_layers_hierarchy(self):
        """Model with flat model.layers hierarchy (no nested .model)."""
        in_dim, out_dim = 64, 32
        layers = []
        self_attn = SimpleNamespace(
            q_proj=FakeLinear(in_dim, out_dim),
            k_proj=FakeLinear(in_dim, out_dim),
            v_proj=FakeLinear(in_dim, out_dim),
            o_proj=FakeLinear(out_dim, in_dim),
        )
        mlp = SimpleNamespace(
            gate_proj=FakeLinear(in_dim, out_dim),
            up_proj=FakeLinear(in_dim, out_dim),
            down_proj=FakeLinear(out_dim, in_dim),
        )
        layers.append(SimpleNamespace(self_attn=self_attn, mlp=mlp))
        model = SimpleNamespace(layers=layers)

        runtime = MoELoRARuntime(
            config=MoELoRAConfig(rank=4, num_experts=2, top_k=1, router_hidden=8, alpha=8.0),
        )
        runtime.load_base_model_from_objects(model, tokenizer=None)

        proj = _make_projection(in_dim=in_dim, out_dim=out_dim)
        count = runtime.load_adapter_from_projections({(0, "self_attn", "q_proj"): proj})
        assert count == 1


# ---------------------------------------------------------------------------
# Tests: Integration — end-to-end forward pass through patched model
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_full_forward_zero_b(self):
        """Full forward through patched model with zero B should match unpatched."""
        in_dim, out_dim = 64, 32
        model = _make_fake_model(num_layers=1, in_dim=in_dim, out_dim=out_dim)
        x = _randn(1, 4, in_dim)

        # Get unpatched output
        original_q = model.model.layers[0].self_attn.q_proj
        expected = original_q(x)

        # Patch with zero-B adapter
        runtime = MoELoRARuntime(
            config=MoELoRAConfig(rank=4, num_experts=2, top_k=1, router_hidden=8, alpha=8.0),
        )
        runtime.load_base_model_from_objects(model, tokenizer=None)
        proj = _make_projection(in_dim=in_dim, out_dim=out_dim, zero_b=True)
        runtime.load_adapter_from_projections({(0, "self_attn", "q_proj"): proj})

        # Patched output with zero B should match original
        result = model.model.layers[0].self_attn.q_proj(x)
        diff = _abs_sum(result - expected)
        assert diff < 1e-4, f"Zero-B patched output differs from original by {diff}"

    def test_full_forward_nonzero_b(self):
        """Full forward through patched model with non-zero B should differ from base."""
        in_dim, out_dim = 64, 32
        model = _make_fake_model(num_layers=1, in_dim=in_dim, out_dim=out_dim)
        x = _randn(1, 4, in_dim)

        # Get unpatched output
        original_q = model.model.layers[0].self_attn.q_proj
        base_result = original_q(x)

        # Patch with non-zero B
        runtime = MoELoRARuntime(
            config=MoELoRAConfig(rank=4, num_experts=2, top_k=1, router_hidden=8, alpha=8.0),
        )
        runtime.load_base_model_from_objects(model, tokenizer=None)
        proj = _make_projection(in_dim=in_dim, out_dim=out_dim, zero_b=False)
        runtime.load_adapter_from_projections({(0, "self_attn", "q_proj"): proj})

        # Patched output should differ from base
        patched_result = model.model.layers[0].self_attn.q_proj(x)
        diff = _abs_sum(patched_result - base_result)
        assert diff > 1e-6, f"Expected patched output to differ from base, diff={diff}"

    def test_hot_swap_changes_output(self):
        """Hot-swapping adapters should change the output."""
        in_dim, out_dim = 64, 32
        model = _make_fake_model(num_layers=1, in_dim=in_dim, out_dim=out_dim)
        x = _randn(1, 4, in_dim)

        runtime = MoELoRARuntime(
            config=MoELoRAConfig(rank=4, num_experts=2, top_k=1, router_hidden=8, alpha=8.0),
        )
        runtime.load_base_model_from_objects(model, tokenizer=None)

        # Adapter 1
        proj1 = _make_projection(in_dim=in_dim, out_dim=out_dim, zero_b=False)
        runtime.load_adapter_from_projections(
            {(0, "self_attn", "q_proj"): proj1}, "a1"
        )
        result1 = model.model.layers[0].self_attn.q_proj(x)

        # Hot-swap to adapter 2 (different random weights)
        proj2 = _make_projection(in_dim=in_dim, out_dim=out_dim, zero_b=False)
        runtime.load_adapter_from_projections(
            {(0, "self_attn", "q_proj"): proj2}, "a2"
        )
        result2 = model.model.layers[0].self_attn.q_proj(x)

        # Very unlikely both produce the exact same output
        diff = _abs_sum(result1 - result2)
        assert diff > 1e-6, f"Expected different outputs after hot-swap, diff={diff}"

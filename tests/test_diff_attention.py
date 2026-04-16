"""Tests for differential attention module (all mocked, no real model)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from src.base.diff_attention import (  # noqa: E402
    DiffAttentionConfig,
    DiffAttentionWrapper,
    DifferentialAttention,
    apply_diff_attention,
    init_lambda,
)


# -------------------------------------------------------------------
# DifferentialAttention core
# -------------------------------------------------------------------


class TestDifferentialAttention:
    def test_output_shape(self) -> None:
        attn = DifferentialAttention(d_model=768, num_heads=12)
        out = attn(torch.randn(2, 16, 768))
        assert out.shape == (2, 16, 768)

    def test_lambda_is_learnable(self) -> None:
        attn = DifferentialAttention(d_model=768, num_heads=12)
        assert attn.lambda_param.requires_grad is True

    def test_lambda_grad_flows(self) -> None:
        """Verify gradient flows through lambda_param."""
        attn = DifferentialAttention(d_model=256, num_heads=4)
        x = torch.randn(1, 8, 256, requires_grad=False)
        out = attn(x)
        loss = out.sum()
        loss.backward()
        assert attn.lambda_param.grad is not None
        assert attn.lambda_param.grad.abs().sum() > 0

    def test_init_lambda_scales_with_depth(self) -> None:
        assert init_lambda(0, 13, 0.8) < init_lambda(12, 13, 0.8)

    def test_init_lambda_values(self) -> None:
        val = init_lambda(0, 13, 0.8)
        assert abs(val - 0.8 * (1 / 13)) < 1e-6

    def test_nonzero_output(self) -> None:
        torch.manual_seed(42)
        attn = DifferentialAttention(d_model=256, num_heads=4)
        assert attn(torch.randn(1, 8, 256)).abs().sum() > 0


# -------------------------------------------------------------------
# DiffAttentionConfig
# -------------------------------------------------------------------


class TestDiffAttentionConfig:
    def test_frozen(self) -> None:
        cfg = DiffAttentionConfig(d_model=256, num_heads=4)
        with pytest.raises(AttributeError):
            cfg.d_model = 512  # type: ignore[misc]

    def test_defaults(self) -> None:
        cfg = DiffAttentionConfig(d_model=256, num_heads=4)
        assert cfg.num_layers == 13
        assert cfg.reinit_lambda == 0.8


# -------------------------------------------------------------------
# DiffAttentionWrapper
# -------------------------------------------------------------------


def _make_fake_attn(d_model: int = 256, num_heads: int = 4) -> nn.Module:
    """Build a minimal mock attention module with Q/K/V/O projections."""
    attn = nn.Module()
    attn.q_proj = nn.Linear(d_model, d_model, bias=False)
    attn.k_proj = nn.Linear(d_model, d_model, bias=False)
    attn.v_proj = nn.Linear(d_model, d_model, bias=False)
    attn.o_proj = nn.Linear(d_model, d_model, bias=False)
    attn.num_heads = num_heads
    return attn


class TestDiffAttentionWrapper:
    def test_forward_returns_tuple(self) -> None:
        original = _make_fake_attn()
        wrapper = DiffAttentionWrapper(original, d_model=256, num_heads=4)
        x = torch.randn(1, 8, 256)
        result = wrapper(x)
        assert isinstance(result, tuple)
        assert result[0].shape == (1, 8, 256)

    def test_preserves_bf16_dtype(self) -> None:
        original = _make_fake_attn()
        wrapper = DiffAttentionWrapper(original, d_model=256, num_heads=4)
        x = torch.randn(1, 8, 256, dtype=torch.bfloat16)
        result = wrapper(x)
        assert result[0].dtype == torch.bfloat16

    def test_preserves_float32_dtype(self) -> None:
        original = _make_fake_attn()
        wrapper = DiffAttentionWrapper(original, d_model=256, num_heads=4)
        x = torch.randn(1, 8, 256, dtype=torch.float32)
        result = wrapper(x)
        assert result[0].dtype == torch.float32

    def test_warm_start_copies_weights(self) -> None:
        original = _make_fake_attn()
        wrapper = DiffAttentionWrapper(original, d_model=256, num_heads=4)
        wrapper._warm_start_projections()
        # Q1 should match original Q exactly
        assert torch.equal(
            wrapper.diff_attn.q1_proj.weight.data,
            original.q_proj.weight.data,
        )
        # Q2 should be close but not identical (perturbation)
        diff = (wrapper.diff_attn.q2_proj.weight.data - original.q_proj.weight.data).abs()
        assert diff.max() > 0  # perturbation applied
        assert diff.max() < 0.1  # but small

    def test_keeps_original_attn_for_rollback(self) -> None:
        original = _make_fake_attn()
        wrapper = DiffAttentionWrapper(original, d_model=256, num_heads=4)
        assert wrapper.original_attn is original

    def test_accepts_extra_kwargs(self) -> None:
        """HF attention layers pass extra kwargs; wrapper must not crash."""
        original = _make_fake_attn()
        wrapper = DiffAttentionWrapper(original, d_model=256, num_heads=4)
        x = torch.randn(1, 4, 256)
        result = wrapper(x, attention_mask=None, position_ids=None)
        assert result[0].shape == (1, 4, 256)


# -------------------------------------------------------------------
# apply_diff_attention
# -------------------------------------------------------------------


def _make_fake_model(
    num_layers: int = 6,
    d_model: int = 256,
    num_heads: int = 4,
    gated_indices: set[int] | None = None,
) -> nn.Module:
    """Build a minimal mock model matching HF structure.

    Layers in *gated_indices* get a ``gate`` attribute to simulate
    GatedDeltaNet layers that should NOT be patched.
    """
    gated = gated_indices or set()
    layers = nn.ModuleList()
    for i in range(num_layers):
        layer = nn.Module()
        attn = _make_fake_attn(d_model, num_heads)
        if i in gated:
            attn.gate = nn.Linear(d_model, d_model, bias=False)
        layer.self_attn = attn
        layers.append(layer)

    inner = nn.Module()
    inner.layers = layers

    model = nn.Module()
    model.model = inner
    return model


class TestApplyDiffAttention:
    def test_patches_specified_layers(self) -> None:
        model = _make_fake_model(num_layers=6)
        patched = apply_diff_attention(model, [0, 2, 4])
        assert patched == [0, 2, 4]
        for i in [0, 2, 4]:
            assert isinstance(model.model.layers[i].self_attn, DiffAttentionWrapper)

    def test_leaves_unpatched_layers_untouched(self) -> None:
        model = _make_fake_model(num_layers=6)
        apply_diff_attention(model, [0, 2])
        for i in [1, 3, 4, 5]:
            assert not isinstance(model.model.layers[i].self_attn, DiffAttentionWrapper)

    def test_skips_out_of_range_indices(self) -> None:
        model = _make_fake_model(num_layers=4)
        patched = apply_diff_attention(model, [0, 1, 99])
        assert patched == [0, 1]

    def test_returns_empty_on_no_valid_indices(self) -> None:
        model = _make_fake_model(num_layers=4)
        patched = apply_diff_attention(model, [10, 20])
        assert patched == []

    def test_respects_config_override(self) -> None:
        model = _make_fake_model(num_layers=4)
        cfg = DiffAttentionConfig(d_model=256, num_heads=4, reinit_lambda=0.5)
        apply_diff_attention(model, [0], config=cfg)
        wrapper = model.model.layers[0].self_attn
        # Lambda should be initialised with 0.5 * (1/13) since config.num_layers=13
        expected = 0.5 * (1 / 13)
        actual = wrapper.diff_attn.lambda_param[0].item()
        assert abs(actual - expected) < 1e-5

    def test_patched_layer_forward_works(self) -> None:
        model = _make_fake_model(num_layers=4, d_model=128, num_heads=4)
        apply_diff_attention(model, [0])
        wrapper = model.model.layers[0].self_attn
        x = torch.randn(1, 8, 128)
        out = wrapper(x)
        assert out[0].shape == (1, 8, 128)

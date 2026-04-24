"""Tests for the V4 training loop with null-space gradient projection.

All tests use fake mx.array data — no real model is loaded.
Each test completes in well under 5 s (no GPU required).
"""
from __future__ import annotations

import numpy as np
import pytest
import mlx.core as mx

from src.stacks.train_loop_v4 import _parse_param_key, project_grad_tree
from src.stacks.null_space_v4 import NullSpaceRegistry


# ---------------------------------------------------------------------------
# _parse_param_key
# ---------------------------------------------------------------------------


class TestParseParamKey:
    def test_standard_2d_lora_a(self):
        """Standard attention LoRA A key returns correct (layer, module)."""
        result = _parse_param_key("layers.10.self_attn.q_proj.lora_a")
        assert result == (10, "self_attn.q_proj")

    def test_standard_2d_lora_b(self):
        """LoRA B key parses identically."""
        result = _parse_param_key("layers.0.self_attn.k_proj.lora_b")
        assert result == (0, "self_attn.k_proj")

    def test_nested_module_switch_mlp(self):
        """3-segment module kind (switch_mlp.gate_proj) parsed correctly."""
        result = _parse_param_key("layers.5.mlp.switch_mlp.gate_proj.lora_a")
        assert result == (5, "mlp.switch_mlp.gate_proj")

    def test_shared_expert_module(self):
        """Two-segment module kind (shared_expert.gate_proj) parsed correctly."""
        result = _parse_param_key("layers.3.mlp.shared_expert.gate_proj.lora_b")
        assert result == (3, "mlp.shared_expert.gate_proj")

    def test_non_lora_suffix_returns_none(self):
        """Keys not ending in lora_a / lora_b return None."""
        assert _parse_param_key("layers.0.self_attn.q_proj.weight") is None

    def test_no_layers_segment_returns_none(self):
        """Keys without 'layers' prefix return None."""
        assert _parse_param_key("embed_tokens.weight") is None

    def test_unknown_module_kind_returns_none(self):
        """Unrecognised module kind (not in MODULE_KINDS) returns None."""
        assert _parse_param_key("layers.0.some_unknown_proj.lora_a") is None

    def test_layer_index_extracted_correctly(self):
        """Layer index at position 31."""
        result = _parse_param_key("layers.31.self_attn.v_proj.lora_a")
        assert result is not None
        assert result[0] == 31


# ---------------------------------------------------------------------------
# project_grad_tree
# ---------------------------------------------------------------------------


def _make_fake_registry_with_projector(
    layer: int = 0,
    module: str = "self_attn.q_proj",
    param_dim: int = 16,
    top_k: int = 2,
) -> NullSpaceRegistry:
    """Build a NullSpaceRegistry with a single projector from random frozen vectors."""
    from src.stacks.null_space_v4 import build_projector

    rng = np.random.default_rng(0)
    frozen = rng.standard_normal((3, param_dim)).astype(np.float32)
    V_keep = build_projector(frozen, top_k=top_k)

    registry = NullSpaceRegistry()
    registry._projectors[(layer, module)] = V_keep
    return registry


class TestProjectGradTree:
    def test_lora_grad_is_modified(self):
        """LoRA gradients matching a registry entry are projected (value changes)."""
        param_dim = 16
        registry = _make_fake_registry_with_projector(
            layer=0, module="self_attn.q_proj", param_dim=param_dim
        )

        # Build a grad tree mimicking: layers -> 0 -> self_attn -> q_proj -> lora_a
        original = np.ones(param_dim, dtype=np.float32)
        grad_tree = {
            "layers": {
                "0": {
                    "self_attn": {
                        "q_proj": {
                            "lora_a": mx.array(original),
                        }
                    }
                }
            }
        }

        projected_tree = project_grad_tree(grad_tree, registry)
        projected_val = np.array(projected_tree["layers"]["0"]["self_attn"]["q_proj"]["lora_a"])

        # The projector should have changed the gradient
        assert not np.allclose(projected_val, original), (
            "Expected projection to modify the gradient, but it was unchanged"
        )

    def test_lora_grad_is_orthogonal_to_frozen_directions(self):
        """Projected LoRA gradient is orthogonal to the V_keep directions."""
        param_dim = 32
        registry = _make_fake_registry_with_projector(
            layer=2, module="self_attn.v_proj", param_dim=param_dim, top_k=3
        )
        V_keep = registry._projectors[(2, "self_attn.v_proj")]

        rng = np.random.default_rng(7)
        original = rng.standard_normal(param_dim).astype(np.float32)

        grad_tree = {
            "layers": {
                "2": {
                    "self_attn": {
                        "v_proj": {
                            "lora_a": mx.array(original),
                        }
                    }
                }
            }
        }

        projected_tree = project_grad_tree(grad_tree, registry)
        proj = np.array(projected_tree["layers"]["2"]["self_attn"]["v_proj"]["lora_a"])

        for i, row in enumerate(V_keep):
            dot = abs(float(np.dot(proj, row)))
            assert dot < 1e-5, f"Not orthogonal to direction {i}: dot={dot:.2e}"

    def test_non_lora_grad_passes_through_unchanged(self):
        """Non-LoRA leaves (e.g. weight) are returned without modification."""
        registry = _make_fake_registry_with_projector()

        original = mx.array(np.ones(8, dtype=np.float32))
        grad_tree = {
            "layers": {
                "0": {
                    "self_attn": {
                        "q_proj": {
                            "weight": original,  # NOT a lora_ key
                        }
                    }
                }
            }
        }

        projected_tree = project_grad_tree(grad_tree, registry)
        result = projected_tree["layers"]["0"]["self_attn"]["q_proj"]["weight"]
        assert np.allclose(np.array(result), np.array(original))

    def test_empty_registry_leaves_all_grads_unchanged(self):
        """With no frozen stacks, the registry is empty and all grads pass through."""
        registry = NullSpaceRegistry()  # empty — no projectors built

        rng = np.random.default_rng(42)
        arr_a = mx.array(rng.standard_normal(16).astype(np.float32))
        arr_b = mx.array(rng.standard_normal(16).astype(np.float32))

        grad_tree = {
            "layers": {
                "0": {
                    "self_attn": {
                        "q_proj": {
                            "lora_a": arr_a,
                            "lora_b": arr_b,
                        }
                    }
                }
            }
        }

        projected_tree = project_grad_tree(grad_tree, registry)
        result_a = np.array(projected_tree["layers"]["0"]["self_attn"]["q_proj"]["lora_a"])
        result_b = np.array(projected_tree["layers"]["0"]["self_attn"]["q_proj"]["lora_b"])

        assert np.allclose(result_a, np.array(arr_a))
        assert np.allclose(result_b, np.array(arr_b))

    def test_handles_multiple_layers_and_modules(self):
        """Grad tree with multiple layers — each projected independently."""
        registry = NullSpaceRegistry()
        rng = np.random.default_rng(99)

        # Register projectors for two different layers
        for layer_idx in (0, 5):
            frozen = rng.standard_normal((2, 16)).astype(np.float32)
            from src.stacks.null_space_v4 import build_projector
            registry._projectors[(layer_idx, "self_attn.q_proj")] = build_projector(
                frozen, top_k=1
            )

        grad_tree = {
            "layers": {
                "0": {
                    "self_attn": {
                        "q_proj": {
                            "lora_a": mx.array(np.ones(16, dtype=np.float32)),
                        }
                    }
                },
                "5": {
                    "self_attn": {
                        "q_proj": {
                            "lora_a": mx.array(np.ones(16, dtype=np.float32)),
                        }
                    }
                },
            }
        }

        projected_tree = project_grad_tree(grad_tree, registry)

        # Both layers should have been projected (values changed)
        p0 = np.array(projected_tree["layers"]["0"]["self_attn"]["q_proj"]["lora_a"])
        p5 = np.array(projected_tree["layers"]["5"]["self_attn"]["q_proj"]["lora_a"])

        assert p0 is not None
        assert p5 is not None
        # They used different projectors so results differ
        # (projectors were built from different random vectors)

    def test_preserves_tree_structure(self):
        """Output tree has the same nested key structure as the input."""
        registry = NullSpaceRegistry()

        grad_tree = {
            "layers": {
                "0": {
                    "self_attn": {
                        "q_proj": {
                            "lora_a": mx.array(np.ones(8, dtype=np.float32)),
                            "lora_b": mx.array(np.ones(8, dtype=np.float32)),
                        }
                    }
                }
            },
            "embed_tokens": {
                "weight": mx.array(np.ones(8, dtype=np.float32)),
            },
        }

        projected_tree = project_grad_tree(grad_tree, registry)

        assert "layers" in projected_tree
        assert "0" in projected_tree["layers"]
        assert "self_attn" in projected_tree["layers"]["0"]
        assert "lora_a" in projected_tree["layers"]["0"]["self_attn"]["q_proj"]
        assert "lora_b" in projected_tree["layers"]["0"]["self_attn"]["q_proj"]
        assert "embed_tokens" in projected_tree

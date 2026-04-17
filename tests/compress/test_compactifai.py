"""Tests for Phase XIII CompactifAI + QTHA."""
from __future__ import annotations

import pytest
from pathlib import Path
from src.compress.compactifai import compress_model, CompressionResult
from src.stacks.qtha import QTHAConfig, estimate_qtha_params
from src.routing.tn_router import TNRouterConfig, estimate_tn_router_params


class TestCompactifAI:
    def test_compress_model_framework(self, tmp_path):
        (tmp_path / "input").mkdir()
        (tmp_path / "input" / "model.safetensors").write_bytes(b"\x00" * 1024)
        result = compress_model(str(tmp_path / "input"), str(tmp_path / "output"), bond_dim=32)
        assert isinstance(result, CompressionResult)
        assert result.bond_dim == 32
        assert (tmp_path / "output").exists()


class TestQTHA:
    def test_config_defaults(self):
        cfg = QTHAConfig()
        assert cfg.bond_dim == 8
        assert "q_proj" in cfg.target_modules

    def test_param_estimate(self):
        params = estimate_qtha_params(hidden_dim=3072, bond_dim=4, num_layers=13)
        assert params < 700_000  # ~640K for 13 full-attn layers, bond_dim=4

    def test_param_scales_with_bond_dim(self):
        small = estimate_qtha_params(3072, 4, 40)
        large = estimate_qtha_params(3072, 16, 40)
        assert large > small


class TestTNRouter:
    def test_config_defaults(self):
        cfg = TNRouterConfig()
        assert cfg.num_domains == 35

    def test_param_estimate(self):
        cfg = TNRouterConfig()
        params = estimate_tn_router_params(cfg)
        assert params < 200_000  # very lightweight

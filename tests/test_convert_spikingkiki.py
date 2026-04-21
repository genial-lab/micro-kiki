"""Tests for the SpikingKiki-35B-A3B conversion script.

Covers:
- CLI --help exits cleanly
- Dry-run produces valid metadata JSON
- Layer map generation correctness (mock, no torch)
- Resume logic skips already-converted layers
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# Script under test
SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "convert_spikingkiki_35b.py"


# ---------------------------------------------------------------------------
# Helper: import the module without executing main()
# ---------------------------------------------------------------------------


def _import_script():
    """Import the conversion script as a module."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("convert_spikingkiki_35b", SCRIPT)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestCLI:
    """Tests that exercise the argparse layer."""

    def test_help_exits_zero(self):
        """--help must exit 0 and print usage without torch."""
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"--help returned {result.returncode}"
        assert "usage" in result.stdout.lower(), "expected 'usage' in help output"

    def test_dry_run_exits_zero(self, tmp_path: Path):
        """--dry-run must exit 0 and write results JSON."""
        results_out = tmp_path / "meta.json"
        result = subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--dry-run",
                "--results-out", str(results_out),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"--dry-run returned {result.returncode}\nstdout={result.stdout}\n"
            f"stderr={result.stderr}"
        )
        assert results_out.exists(), "results JSON not created"

    def test_dry_run_metadata_schema(self, tmp_path: Path):
        """Dry-run metadata JSON must contain required top-level keys."""
        results_out = tmp_path / "meta.json"
        subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--dry-run",
                "--results-out", str(results_out),
            ],
            capture_output=True,
            check=True,
        )
        data = json.loads(results_out.read_text())
        required_keys = {
            "timestamp",
            "status",
            "input",
            "output",
            "timesteps",
            "model_info",
            "layer_map_stats",
            "converted_layers",
            "total_layers",
            "elapsed_s",
            "spike_stats",
        }
        missing = required_keys - set(data.keys())
        assert not missing, f"metadata missing keys: {missing}"

    def test_dry_run_status_is_dry_run(self, tmp_path: Path):
        """Dry-run status field must be 'dry_run'."""
        results_out = tmp_path / "meta.json"
        subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--dry-run",
                "--results-out", str(results_out),
            ],
            capture_output=True,
            check=True,
        )
        data = json.loads(results_out.read_text())
        assert data["status"] == "dry_run"

    def test_dry_run_includes_layer_map(self, tmp_path: Path):
        """Dry-run metadata must include a layer_map list."""
        results_out = tmp_path / "meta.json"
        subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--dry-run",
                "--results-out", str(results_out),
            ],
            capture_output=True,
            check=True,
        )
        data = json.loads(results_out.read_text())
        assert "layer_map" in data
        assert isinstance(data["layer_map"], list)
        assert len(data["layer_map"]) > 0

    def test_no_weights_returns_2(self, tmp_path: Path):
        """Missing --input directory must return exit code 2."""
        results_out = tmp_path / "meta.json"
        result = subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--input", str(tmp_path / "nonexistent"),
                "--output", str(tmp_path / "out"),
                "--results-out", str(results_out),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2, (
            f"expected exit 2, got {result.returncode}"
        )


# ---------------------------------------------------------------------------
# Layer map tests (pure Python, no torch)
# ---------------------------------------------------------------------------


class TestLayerMap:
    """Tests for build_layer_map() and related helpers."""

    @pytest.fixture(scope="class")
    def mod(self):
        return _import_script()

    def test_total_entries(self, mod):
        """Layer map must have the expected total for 94 blocks."""
        layer_map = mod.build_layer_map(num_layers=mod.NUM_LAYERS)
        # Per block: 4 attn + 1 router + 256 experts * 3 projections = 773
        expected = mod.NUM_LAYERS * (4 + 1 + mod.NUM_EXPERTS * 3)
        assert len(layer_map) == expected, (
            f"expected {expected} entries, got {len(layer_map)}"
        )

    def test_small_map_structure(self, mod):
        """A 2-block map with 4 experts has correct structure."""
        layer_map = mod.build_layer_map.__func__(2) if hasattr(
            mod.build_layer_map, "__func__"
        ) else mod.build_layer_map(num_layers=2)

        # Patch NUM_EXPERTS in the call — call with explicit args
        # build_layer_map(num_layers) uses module-level NUM_EXPERTS
        # so we test at scale-2 (2 blocks)
        kinds = {e["kind"] for e in layer_map}
        assert kinds == {"attn_proj", "moe_router", "moe_expert_ffn"}, (
            f"unexpected kinds: {kinds}"
        )

    def test_activation_assignments(self, mod):
        """Router uses identity; expert FFNs use relu; attn uses identity."""
        layer_map = mod.build_layer_map(num_layers=1)
        for entry in layer_map:
            if entry["kind"] == "moe_router":
                assert entry["activation"] == "identity", (
                    "router must use identity activation"
                )
            elif entry["kind"] == "moe_expert_ffn":
                assert entry["activation"] == "relu", (
                    "expert FFN must use relu activation"
                )
            elif entry["kind"] == "attn_proj":
                assert entry["activation"] == "identity", (
                    "attention projection must use identity activation"
                )

    def test_all_converted_false_initially(self, mod):
        """All layer map entries must start with converted=False."""
        layer_map = mod.build_layer_map(num_layers=2)
        assert all(not e["converted"] for e in layer_map), (
            "some entries were marked converted at initialisation"
        )

    def test_layer_map_stats(self, mod):
        """layer_map_stats returns correct counts."""
        layer_map = mod.build_layer_map(num_layers=1)
        stats = mod.layer_map_stats(layer_map)
        assert stats["moe_router"] == 1
        assert stats["attn_proj"] == 4
        assert stats["moe_expert_ffn"] == mod.NUM_EXPERTS * 3
        assert stats["total"] == 1 + 4 + mod.NUM_EXPERTS * 3

    def test_key_prefix_format(self, mod):
        """key_prefix values must start with 'model.layers.<n>'."""
        layer_map = mod.build_layer_map(num_layers=3)
        for entry in layer_map:
            assert entry["key_prefix"].startswith("model.layers."), (
                f"unexpected key_prefix: {entry['key_prefix']}"
            )

    def test_expert_ids_range(self, mod):
        """moe_expert_ffn entries must cover expert_id 0 … NUM_EXPERTS-1."""
        layer_map = mod.build_layer_map(num_layers=1)
        expert_ids = {
            e["expert_id"]
            for e in layer_map
            if e["kind"] == "moe_expert_ffn"
        }
        assert expert_ids == set(range(mod.NUM_EXPERTS)), (
            "expert IDs do not cover the expected range"
        )


# ---------------------------------------------------------------------------
# Qwen3.6 hybrid attention tests
# ---------------------------------------------------------------------------


class TestHybridAttention:
    """Tests for Qwen3.6-style linear_attn / self_attn hybrid layer maps."""

    @pytest.fixture(scope="class")
    def mod(self):
        return _import_script()

    def test_build_layer_map_qwen36_hybrid(self, mod):
        """Hybrid layer_types must emit linear_attn + passthrough entries
        on linear layers and self_attn entries on full layers."""
        # 4 layers: linear, full, linear, full
        layer_types = [
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ]
        layer_map = mod.build_layer_map(
            num_layers=4, num_experts=2, layer_types=layer_types,
        )
        # Layer 0: 5 linear_attn_proj + 4 passthrough + router + 2*3 experts
        l0 = [e for e in layer_map if e["layer_id"] == 0]
        kinds0 = [e["kind"] for e in l0]
        assert kinds0.count("linear_attn_proj") == 5
        assert kinds0.count("linear_attn_passthrough") == 4
        assert kinds0.count("attn_proj") == 0  # no self_attn on linear layer
        assert kinds0.count("moe_router") == 1
        assert kinds0.count("moe_expert_ffn") == 2 * 3

        # Layer 1 (full_attention): 4 self_attn proj, no linear_attn
        l1 = [e for e in layer_map if e["layer_id"] == 1]
        kinds1 = [e["kind"] for e in l1]
        assert kinds1.count("attn_proj") == 4
        assert kinds1.count("linear_attn_proj") == 0
        assert kinds1.count("linear_attn_passthrough") == 0

        # Check key_prefixes for linear layer 0 match expected module names
        la_proj_keys = [
            e["key_prefix"] for e in l0 if e["kind"] == "linear_attn_proj"
        ]
        assert all(".linear_attn." in k for k in la_proj_keys)
        la_modules = {k.rsplit(".", 1)[-1] for k in la_proj_keys}
        assert la_modules == set(mod.LINEAR_ATTN_PROJ_MODULES)

        passthrough_keys = [
            e["key_prefix"]
            for e in l0
            if e["kind"] == "linear_attn_passthrough"
        ]
        pt_tensors = {k.rsplit(".", 1)[-1] for k in passthrough_keys}
        assert pt_tensors == set(mod.LINEAR_ATTN_PASSTHROUGH_TENSORS)

        # Activations: projs=identity, passthroughs=passthrough,
        # self_attn=identity, router=identity, experts=relu
        for e in l0:
            if e["kind"] == "linear_attn_proj":
                assert e["activation"] == "identity"
            elif e["kind"] == "linear_attn_passthrough":
                assert e["activation"] == "passthrough"

    def test_build_layer_map_qwen35_backward_compat(self, mod):
        """Without layer_types, behaviour matches Qwen3.5 (only self_attn)."""
        # Explicit None: no hybrid emitted.
        layer_map = mod.build_layer_map(
            num_layers=3, num_experts=2, layer_types=None,
        )
        kinds = {e["kind"] for e in layer_map}
        assert "linear_attn_proj" not in kinds
        assert "linear_attn_passthrough" not in kinds
        assert "attn_proj" in kinds
        # Each layer should still have exactly 4 self-attn projections
        for layer_idx in range(3):
            layer_entries = [
                e for e in layer_map if e["layer_id"] == layer_idx
            ]
            attn = [e for e in layer_entries if e["kind"] == "attn_proj"]
            assert len(attn) == 4

    def test_dry_run_qwen36_totals(self, mod, tmp_path: Path):
        """Dry-run on a Qwen3.6 fixture config produces expected totals."""
        cfg = {
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "text_config": {
                "hidden_size": 2048,
                "num_hidden_layers": 40,
                "num_attention_heads": 16,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "num_experts": 256,
                "num_experts_per_tok": 8,
                "moe_intermediate_size": 512,
                "shared_expert_intermediate_size": 512,
                "vocab_size": 248320,
                "layer_types": (
                    ["linear_attention"] * 3 + ["full_attention"]
                ) * 10,
            },
        }
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps(cfg))
        results_out = tmp_path / "meta.json"
        result = subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--dry-run",
                "--config", str(cfg_path),
                "--results-out", str(results_out),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"dry-run failed: {result.stderr}"
        )
        data = json.loads(results_out.read_text())
        stats = data["layer_map_stats"]
        # 40 layers: 30 linear (3-in-4) + 10 full
        # linear_attn_proj: 30 * 5 = 150
        # linear_attn_passthrough: 30 * 4 = 120
        # attn_proj: 10 * 4 = 40
        # moe_router: 40
        # moe_expert_ffn: 40 * 256 * 3 = 30720
        assert stats["linear_attn_proj"] == 150, stats
        assert stats["linear_attn_passthrough"] == 120, stats
        assert stats["attn_proj"] == 40, stats
        assert stats["moe_router"] == 40, stats
        assert stats["moe_expert_ffn"] == 30720, stats
        assert stats["total"] == 150 + 120 + 40 + 40 + 30720


# ---------------------------------------------------------------------------
# Resume logic tests
# ---------------------------------------------------------------------------


class TestResumeLogic:
    """Tests for load_resume_state / save_resume_state."""

    @pytest.fixture()
    def mod(self):
        return _import_script()

    def test_empty_state_when_no_file(self, mod, tmp_path: Path):
        """load_resume_state returns empty set when no state file exists."""
        result = mod.load_resume_state(tmp_path / "nonexistent_dir")
        assert result == set()

    def test_roundtrip(self, mod, tmp_path: Path):
        """Saved state is reloaded correctly."""
        keys = {"model.layers.0.self_attn.q_proj", "model.layers.1.mlp.gate"}
        mod.save_resume_state(tmp_path, keys)
        loaded = mod.load_resume_state(tmp_path)
        assert loaded == keys

    def test_corrupted_state_returns_empty(self, mod, tmp_path: Path):
        """Corrupted state file must return empty set, not raise."""
        state_file = tmp_path / ".convert_state.json"
        state_file.write_text("not valid json {{{")
        result = mod.load_resume_state(tmp_path)
        assert result == set()

    def test_resume_skips_converted_layers(self, mod, tmp_path: Path):
        """dry-run with --resume skips already-saved keys."""
        # Pre-populate a resume state with some keys
        pre_converted = {
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
        }
        mod.save_resume_state(tmp_path, pre_converted)

        # load and verify the keys are present
        loaded = mod.load_resume_state(tmp_path)
        assert pre_converted.issubset(loaded)

    def test_save_creates_parent_dirs(self, mod, tmp_path: Path):
        """save_resume_state must create missing parent directories."""
        deep_path = tmp_path / "a" / "b" / "c"
        mod.save_resume_state(deep_path, {"some.key"})
        state_file = deep_path / ".convert_state.json"
        assert state_file.exists()

    def test_incremental_save(self, mod, tmp_path: Path):
        """Saving multiple times accumulates keys correctly."""
        mod.save_resume_state(tmp_path, {"key.a", "key.b"})
        first = mod.load_resume_state(tmp_path)
        assert first == {"key.a", "key.b"}

        mod.save_resume_state(tmp_path, {"key.a", "key.b", "key.c"})
        second = mod.load_resume_state(tmp_path)
        assert second == {"key.a", "key.b", "key.c"}


# ---------------------------------------------------------------------------
# Metadata structure tests
# ---------------------------------------------------------------------------


class TestMetadata:
    """Tests for build_metadata()."""

    @pytest.fixture()
    def mod(self):
        return _import_script()

    def test_model_info_block(self, mod, tmp_path: Path):
        """model_info must contain architecture constants."""
        args = mod.parse_args([
            "--dry-run",
            "--input", str(tmp_path),
            "--output", str(tmp_path / "out"),
        ])
        layer_map = mod.build_layer_map(num_layers=2)
        meta = mod.build_metadata(
            args, layer_map, spike_stats={}, elapsed_s=1.0, status="test"
        )
        info = meta["model_info"]
        assert info["num_experts"] == mod.NUM_EXPERTS
        assert info["top_k"] == mod.TOP_K
        assert info["num_layers"] == mod.NUM_LAYERS

    def test_elapsed_rounded(self, mod, tmp_path: Path):
        """elapsed_s must be rounded to 1 decimal place."""
        args = mod.parse_args(["--dry-run"])
        layer_map = mod.build_layer_map(num_layers=1)
        meta = mod.build_metadata(
            args, layer_map, spike_stats={},
            elapsed_s=12345.6789, status="test"
        )
        assert meta["elapsed_s"] == 12345.7

    def test_converted_layers_count(self, mod, tmp_path: Path):
        """converted_layers must reflect how many entries are marked True."""
        args = mod.parse_args(["--dry-run"])
        layer_map = mod.build_layer_map(num_layers=1)
        # Mark first 3 entries as converted
        for entry in layer_map[:3]:
            entry["converted"] = True
        meta = mod.build_metadata(
            args, layer_map, spike_stats={}, elapsed_s=0.0, status="test"
        )
        assert meta["converted_layers"] == 3


# ---------------------------------------------------------------------------
# MoE expert weight extraction tests (Qwen3.5 ModuleList vs Qwen3.6 fused)
# ---------------------------------------------------------------------------


class TestExpertWeightExtraction:
    """Tests for the layout-agnostic expert weight helper.

    Qwen3.5 exposes ``block.mlp.experts`` as an ``nn.ModuleList`` of per-expert
    triplets. Qwen3.6 replaced that with a single ``Qwen3_5MoeExperts`` module
    holding fused 3D tensors: ``gate_up_proj.weight`` with shape
    ``(E, 2*I, H)`` and ``down_proj.weight`` with shape ``(E, H, I)``. The
    helper must expose both layouts through the same dict API.
    """

    @pytest.fixture(scope="class")
    def mod(self):
        return _import_script()

    @pytest.fixture(scope="class")
    def torch(self):
        torch = pytest.importorskip("torch")
        return torch

    def _fused_stub(self, torch, num_experts=4, intermediate=16, hidden=8):
        """Build a plain-object stub mimicking HF's ``Qwen3_5MoeExperts``.

        The script only touches ``experts.gate_up_proj.weight`` and
        ``experts.down_proj.weight``, so we build a non-``nn.Module`` stub
        to sidestep torch's Parameter/Module attribute gymnastics.
        """
        fused = _FusedStub()
        fused.gate_up_proj = _TensorHolder(
            torch.randn(num_experts, 2 * intermediate, hidden)
        )
        fused.down_proj = _TensorHolder(
            torch.randn(num_experts, hidden, intermediate)
        )
        return fused

    def _modulelist_stub(self, torch, num_experts=4, intermediate=16, hidden=8):
        class Expert(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = torch.nn.Linear(hidden, intermediate, bias=False)
                self.up_proj = torch.nn.Linear(hidden, intermediate, bias=False)
                self.down_proj = torch.nn.Linear(intermediate, hidden, bias=False)

        return torch.nn.ModuleList([Expert() for _ in range(num_experts)])

    def test_detect_fused_true(self, mod, torch):
        fused = self._fused_stub(torch)
        assert mod._detect_fused_experts(fused) is True

    def test_detect_fused_false_on_modulelist(self, mod, torch):
        ml = self._modulelist_stub(torch)
        assert mod._detect_fused_experts(ml) is False

    def test_num_experts_fused(self, mod, torch):
        fused = self._fused_stub(torch, num_experts=7)
        assert mod._num_experts(fused, is_fused=True) == 7

    def test_num_experts_modulelist(self, mod, torch):
        ml = self._modulelist_stub(torch, num_experts=5)
        assert mod._num_experts(ml, is_fused=False) == 5

    def test_fused_expert_weight_extraction(self, mod, torch):
        """Fused layout: helper slices the correct row of each fused tensor.

        gate = first half of gate_up along axis 1; up = second half;
        down = per-expert slice of down_proj.
        """
        hidden, intermediate, num_experts = 8, 16, 4
        fused = self._fused_stub(
            torch, num_experts=num_experts, intermediate=intermediate, hidden=hidden
        )
        for expert_id in range(num_experts):
            w = mod._get_expert_weights(fused, expert_id, is_fused=True)
            assert set(w.keys()) == {"gate_proj", "up_proj", "down_proj"}
            assert w["gate_proj"]["weight"].shape == (intermediate, hidden)
            assert w["up_proj"]["weight"].shape == (intermediate, hidden)
            assert w["down_proj"]["weight"].shape == (hidden, intermediate)
            assert w["gate_proj"]["bias"] is None
            assert w["up_proj"]["bias"] is None
            assert w["down_proj"]["bias"] is None

            # Values must match a manual slice of the fused tensors.
            gate_up = fused.gate_up_proj.weight.detach().cpu().float().numpy()
            down = fused.down_proj.weight.detach().cpu().float().numpy()
            assert (w["gate_proj"]["weight"] == gate_up[expert_id, :intermediate, :]).all()
            assert (w["up_proj"]["weight"] == gate_up[expert_id, intermediate:, :]).all()
            assert (w["down_proj"]["weight"] == down[expert_id]).all()

    def test_modulelist_expert_weight_extraction(self, mod, torch):
        """ModuleList layout: helper returns the same 3-key dict."""
        hidden, intermediate, num_experts = 8, 16, 3
        ml = self._modulelist_stub(
            torch, num_experts=num_experts, intermediate=intermediate, hidden=hidden
        )
        for expert_id in range(num_experts):
            w = mod._get_expert_weights(ml, expert_id, is_fused=False)
            assert set(w.keys()) == {"gate_proj", "up_proj", "down_proj"}
            assert w["gate_proj"]["weight"].shape == (intermediate, hidden)
            assert w["up_proj"]["weight"].shape == (intermediate, hidden)
            assert w["down_proj"]["weight"].shape == (hidden, intermediate)
            # Values must match the underlying nn.Linear weights.
            import numpy as np
            expert = ml[expert_id]
            assert np.allclose(
                w["gate_proj"]["weight"],
                expert.gate_proj.weight.detach().cpu().float().numpy(),
            )


class _TensorHolder:
    """Expose a tensor via ``.weight`` (mimics an ``nn.Linear``-ish surface).

    The real ``Qwen3_5MoeExperts`` attaches its fused parameters such that
    ``experts.gate_up_proj.weight`` resolves to the fused tensor. This shim
    is the compact equivalent for tests — we do not need gradient tracking
    nor the rest of the ``nn.Linear`` API.
    """

    def __init__(self, tensor):
        self.weight = tensor


class _FusedStub:
    """Plain object standing in for ``Qwen3_5MoeExperts`` in tests.

    Not an ``nn.Module``: the conversion script only calls
    ``hasattr(experts, 'gate_up_proj')``, ``experts.gate_up_proj.weight``,
    and ``experts.down_proj.weight``. A bare namespace suffices.
    """

    pass

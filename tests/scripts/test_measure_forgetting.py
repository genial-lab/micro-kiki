"""Tests for scripts/measure_forgetting.py.

Phase 1a covered the angle-only path (status ``angle_only_partial``).
Phase 1b adds the full gate: ``angle<30° AND winrate_drop>0.03``.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# Torch is required by the safetensors-backed fixtures below.
torch = pytest.importorskip("torch")
from safetensors.torch import save_file  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "measure_forgetting.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("measure_forgetting", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    spec.loader.exec_module(mod)
    return mod


def _adapter_tensors(b: torch.Tensor, a: torch.Tensor) -> dict[str, torch.Tensor]:
    """Build a single-layer, single-projection LoRA adapter dict."""
    base = "base_model.model.model.layers.0.self_attn.q_proj"
    return {
        f"{base}.lora_A.weight": a.contiguous(),
        f"{base}.lora_B.weight": b.contiguous(),
    }


def _write_adapter(path: Path, b: torch.Tensor, a: torch.Tensor) -> None:
    save_file(_adapter_tensors(b, a), str(path))


def _write_eval_dataset(path: Path, entries: list[dict[str, str]]) -> None:
    path.write_text(
        "\n".join(json.dumps(e) for e in entries) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Phase 1a regression — angle-only path (back-compat)
# ---------------------------------------------------------------------------


def test_orthogonal_weights_high_angle(tmp_path: Path) -> None:
    mod = _load_script_module()

    # Prior: delta = B1 @ A1 = [[1,0],[0,0]]
    b1 = torch.tensor([[1.0], [0.0]])
    a1 = torch.tensor([[1.0, 0.0]])
    # New: delta = B2 @ A2 = [[0,0],[0,1]] — orthogonal direction
    b2 = torch.tensor([[0.0], [1.0]])
    a2 = torch.tensor([[0.0, 1.0]])

    prior_path = tmp_path / "prior.safetensors"
    new_path = tmp_path / "new.safetensors"
    _write_adapter(prior_path, b1, a1)
    _write_adapter(new_path, b2, a2)

    out_path = tmp_path / "report.json"
    rc = mod.main(
        [
            "--prior-adapter",
            str(prior_path),
            "--new-adapter",
            str(new_path),
            "--output",
            str(out_path),
        ]
    )
    assert rc == 0
    report = json.loads(out_path.read_text())
    assert report["gate_status"] == "angle_only_partial"
    assert report["angle_degrees_mean"] > 85.0, report


def test_parallel_weights_low_angle(tmp_path: Path) -> None:
    mod = _load_script_module()

    # Prior: delta = [[1,0],[0,0]]
    b1 = torch.tensor([[1.0], [0.0]])
    a1 = torch.tensor([[1.0, 0.0]])
    # New: same direction, tiny perturbation only in A/B scale
    b2 = torch.tensor([[1.001], [0.0005]])
    a2 = torch.tensor([[1.0005, 0.0002]])

    prior_path = tmp_path / "prior.safetensors"
    new_path = tmp_path / "new.safetensors"
    _write_adapter(prior_path, b1, a1)
    _write_adapter(new_path, b2, a2)

    out_path = tmp_path / "report.json"
    rc = mod.main(
        [
            "--prior-adapter",
            str(prior_path),
            "--new-adapter",
            str(new_path),
            "--output",
            str(out_path),
        ]
    )
    assert rc == 0
    report = json.loads(out_path.read_text())
    assert report["angle_degrees_mean"] < 10.0, report
    assert report["warning"] == "angle below threshold"


# ---------------------------------------------------------------------------
# Phase 1b — full gate (angle + win-rate, AND-logic rollback)
# ---------------------------------------------------------------------------


def _patch_generate_fn(monkeypatch, callable_: object) -> str:
    """Install ``callable_`` under a fresh module path and return its locator."""
    import types

    module_name = "tests_mock_gen_fn_module"
    mod = types.ModuleType(module_name)
    mod.generate = callable_  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, mod)
    return f"{module_name}:generate"


def test_full_gate_pass_high_angle(tmp_path: Path, monkeypatch) -> None:
    """angle > 30° → gate passes regardless of winrate drop."""
    mod = _load_script_module()

    # Orthogonal-ish deltas → high angle.
    b1 = torch.tensor([[1.0], [0.0]])
    a1 = torch.tensor([[1.0, 0.0]])
    b2 = torch.tensor([[0.0], [1.0]])
    a2 = torch.tensor([[0.0, 1.0]])

    prior_path = tmp_path / "prior.safetensors"
    new_path = tmp_path / "new.safetensors"
    _write_adapter(prior_path, b1, a1)
    _write_adapter(new_path, b2, a2)

    # Eval dataset where generated output never matches references →
    # measured winrate = 0.0 → drop = 0.9 (huge). But angle is high.
    eval_path = tmp_path / "eval.jsonl"
    _write_eval_dataset(
        eval_path,
        [
            {"prompt": "p1", "reference": "EXPECTED_REF_A"},
            {"prompt": "p2", "reference": "EXPECTED_REF_B"},
        ],
    )

    def fake_generate(prompt, adapter):  # noqa: ARG001
        return "nothing matches here"

    locator = _patch_generate_fn(monkeypatch, fake_generate)

    out_path = tmp_path / "report.json"
    rc = mod.main(
        [
            "--prior-adapter",
            str(prior_path),
            "--new-adapter",
            str(new_path),
            "--eval-dataset",
            str(eval_path),
            "--generate-fn-module",
            locator,
            "--winrate-baseline-score",
            "0.9",
            "--output",
            str(out_path),
        ]
    )
    assert rc == 0, "high angle should pass the gate regardless of winrate drop"
    report = json.loads(out_path.read_text())
    assert report["gate_status"] == "pass"
    assert report["angle_degrees_mean"] > 30.0
    assert report["winrate_measured"] == 0.0
    assert report["winrate_drop"] > 0.03


def test_full_gate_fail_both(tmp_path: Path, monkeypatch) -> None:
    """angle < 30° AND winrate_drop > 0.03 → gate fails."""
    mod = _load_script_module()

    # Parallel deltas → very low angle.
    b1 = torch.tensor([[1.0], [0.0]])
    a1 = torch.tensor([[1.0, 0.0]])
    b2 = torch.tensor([[1.001], [0.0005]])
    a2 = torch.tensor([[1.0005, 0.0002]])

    prior_path = tmp_path / "prior.safetensors"
    new_path = tmp_path / "new.safetensors"
    _write_adapter(prior_path, b1, a1)
    _write_adapter(new_path, b2, a2)

    eval_path = tmp_path / "eval.jsonl"
    _write_eval_dataset(
        eval_path,
        [
            {"prompt": "p1", "reference": "EXPECTED_REF_A"},
            {"prompt": "p2", "reference": "EXPECTED_REF_B"},
        ],
    )

    def fake_generate(prompt, adapter):  # noqa: ARG001
        return "nothing matches here"

    locator = _patch_generate_fn(monkeypatch, fake_generate)

    out_path = tmp_path / "report.json"
    rc = mod.main(
        [
            "--prior-adapter",
            str(prior_path),
            "--new-adapter",
            str(new_path),
            "--eval-dataset",
            str(eval_path),
            "--generate-fn-module",
            locator,
            "--winrate-baseline-score",
            "0.9",
            "--output",
            str(out_path),
        ]
    )
    assert rc == 1, "low-angle + high winrate drop must fail the gate"
    report = json.loads(out_path.read_text())
    assert report["gate_status"] == "fail"
    assert report["angle_degrees_mean"] < 30.0
    assert report["winrate_drop"] > 0.03
    assert report["warning"]  # non-empty


def test_full_gate_pass_low_drop(tmp_path: Path, monkeypatch) -> None:
    """angle < 30° BUT winrate_drop < 0.03 → gate passes (AND-logic)."""
    mod = _load_script_module()

    # Parallel deltas → low angle.
    b1 = torch.tensor([[1.0], [0.0]])
    a1 = torch.tensor([[1.0, 0.0]])
    b2 = torch.tensor([[1.001], [0.0005]])
    a2 = torch.tensor([[1.0005, 0.0002]])

    prior_path = tmp_path / "prior.safetensors"
    new_path = tmp_path / "new.safetensors"
    _write_adapter(prior_path, b1, a1)
    _write_adapter(new_path, b2, a2)

    # Generated output matches every reference → winrate = 1.0, drop = -0.01.
    eval_path = tmp_path / "eval.jsonl"
    _write_eval_dataset(
        eval_path,
        [
            {"prompt": "p1", "reference": "match"},
            {"prompt": "p2", "reference": "match"},
        ],
    )

    def fake_generate(prompt, adapter):  # noqa: ARG001
        return "the string match appears here"

    locator = _patch_generate_fn(monkeypatch, fake_generate)

    out_path = tmp_path / "report.json"
    rc = mod.main(
        [
            "--prior-adapter",
            str(prior_path),
            "--new-adapter",
            str(new_path),
            "--eval-dataset",
            str(eval_path),
            "--generate-fn-module",
            locator,
            "--winrate-baseline-score",
            "0.99",
            "--output",
            str(out_path),
        ]
    )
    assert rc == 0, "low angle but stable winrate should pass (AND-logic)"
    report = json.loads(out_path.read_text())
    assert report["gate_status"] == "pass"
    assert report["angle_degrees_mean"] < 30.0
    assert report["winrate_measured"] == 1.0
    assert report["winrate_drop"] <= 0.03


# ---------------------------------------------------------------------------
# MLX-LM-style adapters (no .weight suffix, non-q/k/v/o module paths)
# ---------------------------------------------------------------------------


def _mlx_adapter_tensors(
    module_key: str,
    b: torch.Tensor,
    a: torch.Tensor,
    *,
    layer: int = 0,
) -> dict[str, torch.Tensor]:
    """Build a single-layer MLX-LM adapter dict for ``module_key``.

    MLX-LM keys look like:
        language_model.model.layers.N.<module_key>.lora_a
        language_model.model.layers.N.<module_key>.lora_b
    with NO ``.weight`` suffix and lowercase ``lora_{a,b}``.
    """
    base = f"language_model.model.layers.{layer}.{module_key}"
    # Clone to defeat storage sharing — safetensors rejects aliased tensors.
    return {
        f"{base}.lora_a": a.detach().clone().contiguous(),
        f"{base}.lora_b": b.detach().clone().contiguous(),
    }


def test_mlx_style_keys_matched(tmp_path: Path) -> None:
    """MLX-LM keys (no .weight, lowercase) are picked up by the regex.

    Verifies both the regex matches and that the resulting delta is grouped
    under the expected ``module_key`` — including a non-{q,k,v,o} module
    (``linear_attn.in_proj_a``) that the pre-fix code silently ignored.
    """
    from src.eval.forgetting import _extract_deltas, _parse_lora_key

    # Direct regex check — MLX-style variants. The 4th tuple element is
    # the MoE-LoRA expert index; ``None`` for plain LoRA.
    assert _parse_lora_key(
        "language_model.model.layers.0.linear_attn.in_proj_a.lora_a"
    ) == (
        "language_model.model.layers.0.",
        "linear_attn.in_proj_a",
        "a",
        None,
    )
    assert _parse_lora_key(
        "language_model.model.layers.12.self_attn.q_proj.lora_b"
    ) == (
        "language_model.model.layers.12.",
        "self_attn.q_proj",
        "b",
        None,
    )
    # And PEFT variants still match (back-compat).
    assert _parse_lora_key(
        "base_model.model.layers.3.self_attn.q_proj.lora_A.weight"
    ) == (
        "base_model.model.layers.3.",
        "self_attn.q_proj",
        "a",
        None,
    )
    assert _parse_lora_key(
        "base_model.model.layers.3.self_attn.q_proj.lora_A.default.weight"
    ) == (
        "base_model.model.layers.3.",
        "self_attn.q_proj",
        "a",
        None,
    )
    # Non-LoRA keys return None.
    assert _parse_lora_key("some.random.weight") is None
    assert _parse_lora_key("language_model.model.embed_tokens.weight") is None

    # _extract_deltas on an MLX adapter: `linear_attn.in_proj_a` must appear
    # as a module key (this was NOT captured by the old PEFT-only regex).
    b = torch.tensor([[1.0], [0.0]])
    a = torch.tensor([[1.0, 0.0]])
    tensors = _mlx_adapter_tensors("linear_attn.in_proj_a", b, a, layer=5)
    deltas, per_expert = _extract_deltas(tensors)
    assert "linear_attn.in_proj_a" in deltas
    assert len(deltas["linear_attn.in_proj_a"]) == 1
    # Plain MLX LoRA (no expert structure) → empty per-expert view.
    assert per_expert == {}


def test_moe_lora_key_parsing() -> None:
    """Pre-pivot MoE-LoRA keys (``.experts.<N>.lora_{a,b}``) parse correctly.

    The module_key must strip the ``.experts.N`` segment so per-expert keys
    group under a single module bucket; ``expert_idx`` must be the integer
    extracted from the key.
    """
    from src.eval.forgetting import _parse_lora_key

    # Expert 0 / lora_a.
    assert _parse_lora_key(
        "base_model.model.layers.7.mlp.down_proj_moe_lora.experts.0.lora_a"
    ) == (
        "base_model.model.layers.7.",
        "mlp.down_proj_moe_lora",
        "a",
        0,
    )
    # Expert 3 / lora_b (same module, different expert → same module_key).
    assert _parse_lora_key(
        "base_model.model.layers.7.mlp.down_proj_moe_lora.experts.3.lora_b"
    ) == (
        "base_model.model.layers.7.",
        "mlp.down_proj_moe_lora",
        "b",
        3,
    )
    # PEFT ``.weight`` suffix + MoE-LoRA path.
    assert _parse_lora_key(
        "base_model.model.layers.12.mlp.up_proj_moe_lora.experts.5.lora_A.weight"
    ) == (
        "base_model.model.layers.12.",
        "mlp.up_proj_moe_lora",
        "a",
        5,
    )
    # Router head weights have no ``.lora_{a,b}`` suffix → parser returns None.
    assert _parse_lora_key(
        "base_model.model.layers.7.mlp.router_w1.weight"
    ) is None
    assert _parse_lora_key(
        "base_model.model.layers.7.mlp.router_w2.weight"
    ) is None


def test_moe_lora_delta_averaged() -> None:
    """Per-module MoE-LoRA delta is the mean of per-expert B@A deltas.

    Build a synthetic 2-expert, single-layer adapter where the two experts
    produce distinct deltas; verify ``_extract_deltas`` collapses them to
    the elementwise mean under a single module_key.
    """
    from src.eval.forgetting import _extract_deltas

    # Expert 0: B@A = [[1, 0], [0, 0]]
    a0 = torch.tensor([[1.0, 0.0]])
    b0 = torch.tensor([[1.0], [0.0]])
    # Expert 1: B@A = [[0, 0], [0, 1]] (orthogonal to expert 0)
    a1 = torch.tensor([[0.0, 1.0]])
    b1 = torch.tensor([[0.0], [1.0]])

    prefix = "base_model.model.layers.0.mlp.down_proj_moe_lora"
    tensors: dict[str, torch.Tensor] = {
        f"{prefix}.experts.0.lora_A.weight": a0.contiguous(),
        f"{prefix}.experts.0.lora_B.weight": b0.contiguous(),
        f"{prefix}.experts.1.lora_A.weight": a1.contiguous(),
        f"{prefix}.experts.1.lora_B.weight": b1.contiguous(),
    }

    deltas, per_expert = _extract_deltas(tensors)

    # Both experts must collapse to a single module_key (the .experts.N
    # segment is stripped during parsing).
    assert list(deltas.keys()) == ["mlp.down_proj_moe_lora"], deltas
    # One layer, so one delta matrix under the module_key.
    assert len(deltas["mlp.down_proj_moe_lora"]) == 1
    observed = deltas["mlp.down_proj_moe_lora"][0]
    expected = (b0 @ a0 + b1 @ a1) / 2.0
    assert torch.allclose(observed, expected, atol=1e-6), (observed, expected)
    # Per-expert view exposes both experts under the same module_key,
    # each with its own unaveraged per-layer delta.
    assert list(per_expert.keys()) == ["mlp.down_proj_moe_lora"], per_expert
    assert set(per_expert["mlp.down_proj_moe_lora"].keys()) == {0, 1}
    assert len(per_expert["mlp.down_proj_moe_lora"][0]) == 1
    assert len(per_expert["mlp.down_proj_moe_lora"][1]) == 1
    assert torch.allclose(
        per_expert["mlp.down_proj_moe_lora"][0][0], b0 @ a0, atol=1e-6
    )
    assert torch.allclose(
        per_expert["mlp.down_proj_moe_lora"][1][0], b1 @ a1, atol=1e-6
    )


def test_per_expert_angles_emitted_for_moe(tmp_path: Path) -> None:
    """MoE adapter (pre-pivot ``.experts.<N>`` LoRA) → JSON carries per-expert.

    Build prior + new adapters with 2 experts each under a MoE-LoRA module,
    run the CLI, and verify the JSON report contains
    ``angle_degrees_per_expert`` keyed by module → expert_idx.
    """
    mod = _load_script_module()

    prefix = "base_model.model.layers.0.mlp.down_proj_moe_lora"

    # Prior: expert 0 rank-1 along e_0 e_0^T, expert 1 along e_1 e_1^T.
    prior_tensors: dict[str, torch.Tensor] = {
        f"{prefix}.experts.0.lora_A.weight": torch.tensor(
            [[1.0, 0.0]]
        ).contiguous(),
        f"{prefix}.experts.0.lora_B.weight": torch.tensor(
            [[1.0], [0.0]]
        ).contiguous(),
        f"{prefix}.experts.1.lora_A.weight": torch.tensor(
            [[0.0, 1.0]]
        ).contiguous(),
        f"{prefix}.experts.1.lora_B.weight": torch.tensor(
            [[0.0], [1.0]]
        ).contiguous(),
    }
    # New: swap expert directions so each expert sees a different angle
    # vs its prior (expert 0 → e_0 e_1^T, expert 1 → e_1 e_0^T).
    new_tensors: dict[str, torch.Tensor] = {
        f"{prefix}.experts.0.lora_A.weight": torch.tensor(
            [[0.0, 1.0]]
        ).contiguous(),
        f"{prefix}.experts.0.lora_B.weight": torch.tensor(
            [[1.0], [0.0]]
        ).contiguous(),
        f"{prefix}.experts.1.lora_A.weight": torch.tensor(
            [[1.0, 0.0]]
        ).contiguous(),
        f"{prefix}.experts.1.lora_B.weight": torch.tensor(
            [[0.0], [1.0]]
        ).contiguous(),
    }

    prior_path = tmp_path / "prior.safetensors"
    new_path = tmp_path / "new.safetensors"
    save_file(prior_tensors, str(prior_path))
    save_file(new_tensors, str(new_path))

    out_path = tmp_path / "report.json"
    rc = mod.main(
        [
            "--prior-adapter",
            str(prior_path),
            "--new-adapter",
            str(new_path),
            "--output",
            str(out_path),
        ]
    )
    assert rc == 0
    report = json.loads(out_path.read_text())

    # Field present + keyed by module → expert.
    assert "angle_degrees_per_expert" in report, report
    per_expert_out = report["angle_degrees_per_expert"]
    assert "mlp.down_proj_moe_lora" in per_expert_out, per_expert_out
    experts = per_expert_out["mlp.down_proj_moe_lora"]
    # JSON object keys are strings; both experts from the adapter appear.
    assert set(experts.keys()) == {"0", "1"}, experts
    # Every per-expert angle is a finite float in [0, 180].
    for expert_idx, angle in experts.items():
        assert isinstance(angle, (int, float)), (expert_idx, angle)
        assert 0.0 <= float(angle) <= 180.0, (expert_idx, angle)
    # Legacy averaged view still intact — default unchanged.
    assert "angle_degrees_per_module" in report
    assert "mlp.down_proj_moe_lora" in report["angle_degrees_per_module"]


def test_per_expert_angles_absent_for_plain_lora(tmp_path: Path) -> None:
    """Plain LoRA adapter (no ``.experts.<N>``) → field must be omitted.

    Mirrors ``test_orthogonal_weights_high_angle`` but asserts the new
    optional ``angle_degrees_per_expert`` field is absent so existing
    callers that parse the JSON strictly are unaffected.
    """
    mod = _load_script_module()

    b1 = torch.tensor([[1.0], [0.0]])
    a1 = torch.tensor([[1.0, 0.0]])
    b2 = torch.tensor([[0.0], [1.0]])
    a2 = torch.tensor([[0.0, 1.0]])

    prior_path = tmp_path / "prior.safetensors"
    new_path = tmp_path / "new.safetensors"
    _write_adapter(prior_path, b1, a1)
    _write_adapter(new_path, b2, a2)

    out_path = tmp_path / "report.json"
    rc = mod.main(
        [
            "--prior-adapter",
            str(prior_path),
            "--new-adapter",
            str(new_path),
            "--output",
            str(out_path),
        ]
    )
    assert rc == 0
    report = json.loads(out_path.read_text())

    # No MoE structure in either adapter → field must be omitted.
    assert "angle_degrees_per_expert" not in report, report
    # Averaged per-module view still there (sanity).
    assert "angle_degrees_per_module" in report
    assert "self_attn.q_proj" in report["angle_degrees_per_module"]


def test_non_qkvo_modules_captured(tmp_path: Path) -> None:
    """MLX adapter with MoE + linear-attn modules yields per-module angles.

    Covers modules the old PEFT-only regex ignored:
    ``mlp.switch_mlp.down_proj``, ``linear_attn.in_proj_qkv``,
    ``mlp.shared_expert.gate_proj``. Verifies they appear in the
    per-module breakdown produced by ``measure_forgetting_signal``.
    """
    mod = _load_script_module()

    # Prior vs new: orthogonal per-module directions in 4×4 flattened space.
    # Use distinct per-layer deltas so the stacked column space has full rank
    # (rank-1 replicated columns produce degenerate QR padding that breaks
    # the subspace-angle interpretation).
    # All matrices below are (2,2) so flattened deltas live in R^4.
    def _delta_pair(idx: int, kind: str) -> tuple[torch.Tensor, torch.Tensor]:
        # kind="prior": deltas span the first two canonical axes.
        # kind="new":   deltas span the last two canonical axes (orthogonal).
        if kind == "prior":
            ones_at = [(0, 0), (0, 1)][idx]
        else:
            ones_at = [(1, 0), (1, 1)][idx]
        b = torch.zeros(2, 1)
        b[ones_at[0], 0] = 1.0
        a = torch.zeros(1, 2)
        a[0, ones_at[1]] = 1.0
        return b, a

    prior_tensors: dict[str, torch.Tensor] = {}
    new_tensors: dict[str, torch.Tensor] = {}
    for layer_idx, layer in enumerate((0, 1)):
        for module_key in (
            "mlp.switch_mlp.down_proj",
            "linear_attn.in_proj_qkv",
            "mlp.shared_expert.gate_proj",
        ):
            b_p, a_p = _delta_pair(layer_idx, "prior")
            b_n, a_n = _delta_pair(layer_idx, "new")
            prior_tensors.update(
                _mlx_adapter_tensors(module_key, b_p, a_p, layer=layer)
            )
            new_tensors.update(
                _mlx_adapter_tensors(module_key, b_n, a_n, layer=layer)
            )

    prior_path = tmp_path / "prior.safetensors"
    new_path = tmp_path / "new.safetensors"
    save_file(prior_tensors, str(prior_path))
    save_file(new_tensors, str(new_path))

    out_path = tmp_path / "report.json"
    rc = mod.main(
        [
            "--prior-adapter",
            str(prior_path),
            "--new-adapter",
            str(new_path),
            "--output",
            str(out_path),
        ]
    )
    assert rc == 0
    report = json.loads(out_path.read_text())

    # All three modules must be present in the per-module breakdown —
    # previously none of these would have matched the PEFT-only regex.
    per_module = report["angle_degrees_per_module"]
    assert "mlp.switch_mlp.down_proj" in per_module, per_module
    assert "linear_attn.in_proj_qkv" in per_module, per_module
    assert "mlp.shared_expert.gate_proj" in per_module, per_module
    # Orthogonal A vs B → each per-module angle should be ~90°.
    for module_key, angle in per_module.items():
        assert angle > 85.0, (module_key, angle)

    # JSON must use the new field name; old name must be gone.
    assert "angle_degrees_per_layer" not in report


def test_partial_winrate_flags_rejected(tmp_path: Path) -> None:
    """Providing only some win-rate flags → exit 2 (CLI misuse)."""
    mod = _load_script_module()

    b1 = torch.tensor([[1.0], [0.0]])
    a1 = torch.tensor([[1.0, 0.0]])
    b2 = torch.tensor([[0.0], [1.0]])
    a2 = torch.tensor([[0.0, 1.0]])

    prior_path = tmp_path / "prior.safetensors"
    new_path = tmp_path / "new.safetensors"
    _write_adapter(prior_path, b1, a1)
    _write_adapter(new_path, b2, a2)

    rc = mod.main(
        [
            "--prior-adapter",
            str(prior_path),
            "--new-adapter",
            str(new_path),
            # Only baseline, missing dataset and module → misuse.
            "--winrate-baseline-score",
            "0.8",
        ]
    )
    assert rc == 2


# ---------------------------------------------------------------------------
# Phase 1c — full gate end-to-end via mocked MLX client
# ---------------------------------------------------------------------------


def test_full_gate_via_mocked_mlx_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full gate runs end-to-end through ``src.serving.mlx_client:generate``.

    Patches the MLX client's sync entry point to a deterministic mock so the
    CLI exercises the real importlib → ``_compute_winrate`` path. Result must
    NOT be ``angle_only_partial`` (all win-rate flags are set).
    """
    mod = _load_script_module()

    # Orthogonal deltas → high angle → gate passes regardless of winrate.
    b1 = torch.tensor([[1.0], [0.0]])
    a1 = torch.tensor([[1.0, 0.0]])
    b2 = torch.tensor([[0.0], [1.0]])
    a2 = torch.tensor([[0.0, 1.0]])

    prior_path = tmp_path / "prior.safetensors"
    new_path = tmp_path / "new.safetensors"
    _write_adapter(prior_path, b1, a1)
    _write_adapter(new_path, b2, a2)

    eval_path = tmp_path / "eval.jsonl"
    _write_eval_dataset(
        eval_path,
        [
            {"prompt": "p1", "reference": "deterministic"},
            {"prompt": "p2", "reference": "deterministic"},
        ],
    )

    # Patch mlx_client.generate (the sync module-level entry point) so the
    # CLI's ``--generate-fn-module src.serving.mlx_client:generate`` path is
    # exercised without touching the real async httpx stack.
    import src.serving.mlx_client as mlx_client

    calls: list[tuple[str, object]] = []

    def _mock_generate(prompt, adapter=None, **_kwargs):
        calls.append((prompt, adapter))
        return "response containing deterministic token"

    monkeypatch.setattr(mlx_client, "generate", _mock_generate)

    out_path = tmp_path / "report.json"
    rc = mod.main(
        [
            "--prior-adapter",
            str(prior_path),
            "--new-adapter",
            str(new_path),
            "--eval-dataset",
            str(eval_path),
            "--generate-fn-module",
            "src.serving.mlx_client:generate",
            "--winrate-baseline-score",
            "0.9",
            "--output",
            str(out_path),
        ]
    )
    assert rc == 0
    report = json.loads(out_path.read_text())
    # Critical: full gate ran — no partial status.
    assert report["gate_status_aggregate"] != "angle_only_partial", report
    assert report["gate_status_aggregate"] == "pass"
    # And the mock was actually hit (2 prompts × 1 adapter = 2 calls).
    assert len(calls) == 2, calls
    # Adapter kwarg propagated as the new-adapter path.
    assert all(str(new_path) == str(adapter) for _, adapter in calls), calls


# ---------------------------------------------------------------------------
# Phase 2 — pluggable scorer (LLM-judge path)
# ---------------------------------------------------------------------------


def test_full_gate_with_judge_scorer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--scorer-module wires a mocked LLM-judge scorer through the full gate.

    Parallel deltas give a low angle, but the mocked judge returns a score
    of 1.0 per prompt → winrate = 1.0, drop <= 0.03 → gate passes via
    AND-logic. This verifies the judge-based scorer propagates through to
    ``gate_status`` (i.e. the pluggable scorer replaces containment).
    """
    import types

    mod = _load_script_module()

    # Parallel deltas → low angle.
    b1 = torch.tensor([[1.0], [0.0]])
    a1 = torch.tensor([[1.0, 0.0]])
    b2 = torch.tensor([[1.001], [0.0005]])
    a2 = torch.tensor([[1.0005, 0.0002]])

    prior_path = tmp_path / "prior.safetensors"
    new_path = tmp_path / "new.safetensors"
    _write_adapter(prior_path, b1, a1)
    _write_adapter(new_path, b2, a2)

    # Reference is deliberately absent from the response — containment would
    # return 0.0 and fail. The judge scorer returns 1.0, so the gate should
    # pass, proving the scorer is actually being used.
    eval_path = tmp_path / "eval.jsonl"
    _write_eval_dataset(
        eval_path,
        [
            {"prompt": "p1", "reference": "EXPECTED_MARKER_A"},
            {"prompt": "p2", "reference": "EXPECTED_MARKER_B"},
        ],
    )

    def fake_generate(prompt, adapter):  # noqa: ARG001
        return "response without the marker"

    gen_locator = _patch_generate_fn(monkeypatch, fake_generate)

    # Install a JudgeScorer-style async callable under a module locator.
    judge_calls: list[tuple[str, str, str]] = []

    async def fake_judge_scorer(prompt, reference, response):
        judge_calls.append((prompt, reference, response))
        return 1.0

    scorer_mod_name = "tests_mock_scorer_module"
    scorer_mod = types.ModuleType(scorer_mod_name)
    scorer_mod.scorer = fake_judge_scorer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, scorer_mod_name, scorer_mod)

    out_path = tmp_path / "report.json"
    rc = mod.main(
        [
            "--prior-adapter",
            str(prior_path),
            "--new-adapter",
            str(new_path),
            "--eval-dataset",
            str(eval_path),
            "--generate-fn-module",
            gen_locator,
            "--winrate-baseline-score",
            "0.99",
            "--scorer-module",
            f"{scorer_mod_name}:scorer",
            "--output",
            str(out_path),
        ]
    )
    assert rc == 0, "judge-scored winrate=1.0 should pass the gate"
    report = json.loads(out_path.read_text())

    # Judge-based winrate = 1.0 even though containment would have been 0.
    assert report["winrate_measured"] == 1.0, report
    assert report["gate_status"] == "pass"
    assert report["angle_degrees_mean"] < 30.0
    # Judge was called once per prompt, with the right shape.
    assert len(judge_calls) == 2, judge_calls
    for prompt, reference, response in judge_calls:
        assert prompt in {"p1", "p2"}
        assert reference.startswith("EXPECTED_MARKER_")
        assert response == "response without the marker"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

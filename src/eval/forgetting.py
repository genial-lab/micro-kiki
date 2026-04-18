"""Forgetting check framework: win-rate delta + gradient subspace angle.

Run after each stack training to detect interference with prior stacks.
Reference: arxiv 2603.02224 (Subspace Geometry, 2026), OPLoRA 2510.13003.

Provides:
- GradientSubspaceAnalyzer: SVD-based gradient subspace angle measurement
- ForgettingEvaluator: orchestrates win-rate + angle checks across stacks
- ForgettingReport: immutable result dataclass
- measure_forgetting_signal(): shared helper (angle + optional win-rate)
  used by scripts/measure_forgetting.py (CLI) and check_all_previous()

Usable standalone (python -m src.eval.forgetting) or imported by
src.ralph.forgetting_auto.ForgettingChecker.
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, NamedTuple

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForgettingReport:
    """Result of a single forgetting check (one prior stack vs new stack)."""

    stack_id: str
    new_stack_id: str
    angle: float
    winrate_base: float
    winrate_adapted: float
    winrate_drop: float
    passed: bool
    should_rollback: bool


# ---------------------------------------------------------------------------
# Gradient subspace analyzer
# ---------------------------------------------------------------------------


class GradientSubspaceAnalyzer:
    """Measures geometric angle between gradient subspaces via SVD."""

    def compute_angle(
        self,
        base_grads: torch.Tensor,
        adapted_grads: torch.Tensor,
    ) -> float:
        """Geometric angle in degrees between two gradient subspaces.

        Uses QR decomposition to get orthonormal bases, then SVD of the
        cross-product to find the principal angle.

        Args:
            base_grads: gradient matrix from base model (n_params, n_samples)
            adapted_grads: gradient matrix from adapted model (n_params, n_samples)

        Returns:
            Angle in degrees. 90° = orthogonal (no interference), 0° = identical.
        """
        import torch

        q_base, _ = torch.linalg.qr(base_grads.float())
        q_adapted, _ = torch.linalg.qr(adapted_grads.float())

        # Principal angle via SVD of Q_base^T @ Q_adapted
        cross = q_base.T @ q_adapted
        singular_values = torch.linalg.svdvals(cross)
        # Clamp for numerical stability
        max_sv = singular_values[0].clamp(max=1.0)
        angle_rad = torch.acos(max_sv)
        return math.degrees(angle_rad.item())

    def collect_gradients(
        self,
        model: object,
        eval_data: list[dict],
        n_samples: int = 100,
    ) -> torch.Tensor:
        """Collect gradient matrix from a small eval set.

        Runs forward+backward on up to n_samples examples, collecting
        gradients of all trainable parameters into a matrix.

        Args:
            model: a PyTorch model (with .parameters())
            eval_data: list of dicts with at least an 'input_ids' key
            n_samples: max number of samples to use

        Returns:
            Gradient matrix of shape (n_params, min(n_samples, len(eval_data)))
        """
        import torch

        model.eval()
        samples = eval_data[:n_samples]
        grad_columns = []

        for sample in samples:
            model.zero_grad()
            input_ids = sample["input_ids"]
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids)
            input_ids = input_ids.unsqueeze(0).to(next(model.parameters()).device)

            outputs = model(input_ids, labels=input_ids)
            outputs.loss.backward()

            grads = []
            for p in model.parameters():
                if p.grad is not None:
                    grads.append(p.grad.detach().flatten())
            grad_columns.append(torch.cat(grads))

        # (n_params, n_samples)
        return torch.stack(grad_columns, dim=1)


# ---------------------------------------------------------------------------
# Thresholds (locked by brainstorm-oplora.md)
# ---------------------------------------------------------------------------

ANGLE_THRESHOLD = 30.0
WINRATE_DROP_THRESHOLD = 0.03

# Per-module canary: the shared_expert_gate has a rank-1 delta shape
# (Qwen3.5-35B-A3B: ``(hidden, r) @ (r, 1)`` → ``(hidden, 1)``). Its
# geometry is structurally constrained — low angles there are an
# artifact of the shape, not a meaningful forgetting signal.
# Empirical evidence: post-pivot 5-adapter sweep (2026-04-18) showed
# this module hits 17.3° on python↔typescript while every other
# module stays > 42°. See docs/training/forgetting-gate.md.
DEFAULT_PER_MODULE_IGNORE = frozenset({"mlp.shared_expert_gate"})


class GateDecision(NamedTuple):
    """Detailed AND-gate verdict with per-axis booleans for message building."""

    failed: bool
    angle_bad: bool
    delta_bad: bool


class PerModuleGateDecision(NamedTuple):
    """Per-module AND-gate verdict.

    ``failed`` is True iff at least one non-ignored module has
    ``angle < angle_threshold`` AND the supplied ``winrate_drop``
    exceeds ``winrate_drop_threshold``. When ``winrate_drop`` is
    ``None`` the gate runs in partial/angle-only mode: ``failed`` is
    always False but ``offending_modules`` still lists modules whose
    angle alone is below the threshold (informational).
    """

    failed: bool
    offending_modules: list[str]
    min_angle_module: str
    min_angle_value: float


def apply_and_gate_detailed(
    angle: float,
    winrate_drop: float,
    angle_threshold: float = ANGLE_THRESHOLD,
    winrate_drop_threshold: float = WINRATE_DROP_THRESHOLD,
) -> GateDecision:
    """Return the detailed AND-gate decision.

    Single source of truth for the AND-gate: rollback iff ``angle < angle_threshold``
    AND ``winrate_drop > winrate_drop_threshold``. Also exposes per-axis booleans
    so callers can build diagnostic messages without re-deriving them.
    """
    angle_bad = angle < angle_threshold
    delta_bad = winrate_drop > winrate_drop_threshold
    return GateDecision(failed=angle_bad and delta_bad, angle_bad=angle_bad, delta_bad=delta_bad)


def apply_and_gate(
    angle: float,
    winrate_drop: float,
    angle_threshold: float = ANGLE_THRESHOLD,
    winrate_drop_threshold: float = WINRATE_DROP_THRESHOLD,
) -> bool:
    """Return True when the forgetting gate trips (rollback).

    Thin bool wrapper over :func:`apply_and_gate_detailed`. Shared by
    ``measure_forgetting_signal``, ``ForgettingEvaluator``, and
    ``src.ralph.forgetting_auto.ForgettingChecker``.
    """
    return apply_and_gate_detailed(
        angle, winrate_drop, angle_threshold, winrate_drop_threshold
    ).failed


def apply_per_module_gate(
    per_module_angles: dict[str, float],
    winrate_drop: float | None,
    angle_threshold: float = ANGLE_THRESHOLD,
    winrate_drop_threshold: float = WINRATE_DROP_THRESHOLD,
    ignore_modules: set[str] | frozenset[str] | None = None,
) -> PerModuleGateDecision:
    """Per-module forgetting gate.

    Fails iff ANY non-ignored module has ``angle < angle_threshold`` AND
    the aggregate ``winrate_drop`` exceeds ``winrate_drop_threshold``.
    This is the fine-grained complement of :func:`apply_and_gate`: the
    aggregate gate uses the mean across modules and masks individual
    divergences, whereas this gate flags the single worst offender.

    The AND-logic with the win-rate drop matches the project invariant
    (top-level ``CLAUDE.md``): "rollback if angle < 30° AND win-rate
    drop > 0.03". A low module angle alone is not a rollback signal —
    the win-rate drop is still required for the gate to fail.

    When ``winrate_drop`` is ``None`` (angle-only / partial mode), the
    gate never fails but ``offending_modules`` still lists modules
    whose angle alone falls below the threshold — callers may surface
    this informationally.

    Args:
        per_module_angles: mapping ``module_key -> angle_degrees``,
            typically from :func:`compute_angles`.
        winrate_drop: baseline − measured win-rate. ``None`` for
            angle-only partial mode.
        angle_threshold: degrees below which a module's angle is
            considered unsafe (default 30.0).
        winrate_drop_threshold: fractional drop above which win-rate
            signals regression (default 0.03).
        ignore_modules: module keys to skip (not considered for gate
            or offender/min reporting). Defaults to
            :data:`DEFAULT_PER_MODULE_IGNORE` which excludes
            ``mlp.shared_expert_gate`` — see its docstring for the
            rank-1 delta rationale. Pass an empty ``set()`` to consider
            every module (strictest mode). Pass ``None`` for the
            default.

    Returns:
        :class:`PerModuleGateDecision` with ``failed``,
        ``offending_modules`` (modules with angle below threshold,
        empty list if none or if ``per_module_angles`` is empty),
        ``min_angle_module`` (the lowest-angle non-ignored module,
        ``""`` if no modules), and ``min_angle_value`` (the
        corresponding angle, ``nan`` if no modules).
    """
    import math

    if ignore_modules is None:
        ignore_modules = DEFAULT_PER_MODULE_IGNORE

    considered = {
        k: float(v) for k, v in per_module_angles.items() if k not in ignore_modules
    }
    if not considered:
        return PerModuleGateDecision(
            failed=False,
            offending_modules=[],
            min_angle_module="",
            min_angle_value=float("nan"),
        )

    min_module = min(considered, key=considered.__getitem__)
    min_value = considered[min_module]
    offending = sorted(k for k, v in considered.items() if v < angle_threshold)

    # AND-logic: a low angle is a rollback signal only when paired with
    # a win-rate drop above threshold. In partial/angle-only mode
    # (``winrate_drop is None``) we never fail — the caller is expected
    # to treat ``offending_modules`` as informational.
    delta_bad = (
        winrate_drop is not None and winrate_drop > winrate_drop_threshold
    )
    failed = bool(offending) and delta_bad
    if not math.isfinite(min_value):
        min_value = float("nan")

    return PerModuleGateDecision(
        failed=failed,
        offending_modules=offending,
        min_angle_module=min_module,
        min_angle_value=float(min_value),
    )


# ---------------------------------------------------------------------------
# Low-level helpers: LoRA delta extraction + angle computation
# (shared by scripts/measure_forgetting.py CLI and measure_forgetting_signal)
# ---------------------------------------------------------------------------

# Legacy: PEFT-style q/k/v/o-only grouping. Kept for backward-compat imports
# but no longer used to filter — the gate now groups by the actual module key
# discovered in the adapter (see ``_parse_lora_key``).
PROJ_GROUPS = ("q_proj", "k_proj", "v_proj", "o_proj")

# Match the ``.lora_{a,b}`` suffix with all known adapter flavours:
#   PEFT:    ``.lora_A.weight`` / ``.lora_A.default.weight``
#   MLX-LM:  ``.lora_a`` (no ``.weight`` suffix, lowercase)
_LORA_SUFFIX = re.compile(
    r"\.lora_(?P<ab>[AaBb])(?:\.default)?(?:\.weight)?$"
)

# Used to split ``<layer_prefix>.layers.<N>.<module_key>`` so the same
# ``module_key`` (e.g. ``self_attn.q_proj``) groups across layers.
_LAYER_SPLIT = re.compile(r"layers\.\d+\.")


def _parse_lora_key(key: str) -> tuple[str, str, str] | None:
    """Extract ``(layer_prefix, module_key, ab)`` from a LoRA adapter tensor name.

    Supports both PEFT (``base_model.model.layers.N.self_attn.q_proj.lora_A.weight``)
    and MLX-LM (``language_model.model.layers.N.linear_attn.in_proj_qkv.lora_a``)
    conventions. Returns ``None`` if the key is not a LoRA tensor.

    - ``layer_prefix``: everything up to and including ``layers.<N>.`` — used
      as the unique identifier that pairs lora_A with lora_B of the same layer.
      When there is no ``layers.<N>.`` segment, we fall back to the parent of
      the module path (still unique per layer in practice).
    - ``module_key``: the module path relative to the layer (e.g.
      ``self_attn.q_proj``, ``linear_attn.in_proj_qkv``, ``mlp.switch_mlp.down_proj``).
    - ``ab``: lower-cased ``"a"`` or ``"b"``.
    """
    m = _LORA_SUFFIX.search(key)
    if not m:
        return None
    ab = m.group("ab").lower()
    body = key[: m.start()]
    layer_m = _LAYER_SPLIT.search(body)
    if layer_m is not None:
        layer_prefix = body[: layer_m.end()]
        module_key = body[layer_m.end():]
    else:
        # No ``layers.<N>.`` segment — treat everything except the final
        # module name as the prefix. This is rare but keeps pairing correct
        # for adapters that don't follow the decoder-layers convention.
        parent, _, leaf = body.rpartition(".")
        if not leaf:
            return None
        layer_prefix = parent
        module_key = leaf
    if not module_key:
        return None
    return layer_prefix, module_key, ab


def _load_tensors(path: Path) -> dict[str, "torch.Tensor"]:
    """Load a LoRA adapter safetensors file."""
    from safetensors.torch import load_file

    logger.debug("loading safetensors: %s", path)
    return load_file(str(path))


def _compose_delta(
    a: "torch.Tensor", b: "torch.Tensor"
) -> "torch.Tensor | None":
    """Compose a LoRA A/B pair into ``Δ = B·A`` handling both conventions.

    Orientations supported:

    - **PEFT** (``peft.LoraConfig``): ``A`` is ``(r, in_features)`` and
      ``B`` is ``(out_features, r)``. Delta is ``B @ A``.
    - **MLX-LM**: ``A`` is ``(in_features, r)`` and ``B`` is
      ``(r, out_features)``. Delta is ``A @ B`` (equivalent to
      ``B.T @ A.T`` in PEFT shape). Detected by matching inner dim on the
      opposite axes.
    - **MLX-LM MoE switch_mlp**: A is ``(n_experts, r, in_features)`` or
      ``(n_experts, in_features, r)`` and B is the matching 3D tensor.
      Expert dim is folded into the "layer" axis so each expert contributes
      an independent per-layer delta to the subspace.

    Returns ``None`` if the shapes are not compatible under any known
    convention.
    """
    import torch

    a_f = a.float()
    b_f = b.float()

    # 3D (MoE switch_mlp): stack per-expert deltas.
    if a_f.ndim == 3 and b_f.ndim == 3 and a_f.shape[0] == b_f.shape[0]:
        deltas: list[torch.Tensor] = []
        for i in range(a_f.shape[0]):
            d = _compose_delta(a_f[i], b_f[i])
            if d is None:
                return None
            deltas.append(d)
        # Return a single tensor shape (n_experts, out, in); caller flattens.
        return torch.stack(deltas, dim=0)

    if a_f.ndim != 2 or b_f.ndim != 2:
        return None

    # PEFT convention: A=(r,in), B=(out,r), delta = B @ A.
    if b_f.shape[1] == a_f.shape[0]:
        return b_f @ a_f
    # MLX convention: A=(in,r), B=(r,out), delta = A @ B.
    if a_f.shape[1] == b_f.shape[0]:
        return a_f @ b_f
    return None


def _extract_deltas(
    tensors: dict[str, "torch.Tensor"],
) -> dict[str, list["torch.Tensor"]]:
    """Group LoRA A/B pairs by module-kind and compute per-layer deltas.

    Returns a mapping ``module_key -> [delta_layer_0, delta_layer_1, ...]``.
    ``module_key`` is the module path relative to the decoder layer (e.g.
    ``self_attn.q_proj``, ``linear_attn.in_proj_qkv``,
    ``mlp.switch_mlp.down_proj``) so the same kind of module groups across
    layers even when tensor names differ in layer index or top-level prefix.

    Supports PEFT (``.lora_A.weight`` / ``.lora_A.default.weight``) and
    MLX-LM (``.lora_a`` with no ``.weight`` suffix) conventions, including
    3D MoE tensors (``switch_mlp.*`` with per-expert stacks) — see
    :func:`_compose_delta`.

    Layers with only A or only B (malformed) are skipped with a warning.
    """
    a_matrices: dict[tuple[str, str], "torch.Tensor"] = {}
    b_matrices: dict[tuple[str, str], "torch.Tensor"] = {}

    for key, tensor in tensors.items():
        parsed = _parse_lora_key(key)
        if parsed is None:
            continue
        layer_prefix, module_key, ab = parsed
        bucket = a_matrices if ab == "a" else b_matrices
        bucket[(layer_prefix, module_key)] = tensor

    deltas: dict[str, list[Any]] = defaultdict(list)
    for (layer_prefix, module_key), a in a_matrices.items():
        b = b_matrices.get((layer_prefix, module_key))
        if b is None:
            logger.warning(
                "missing lora_B for %s%s; skipping", layer_prefix, module_key
            )
            continue
        delta = _compose_delta(a, b)
        if delta is None:
            logger.warning(
                "incompatible A/B shapes for %s%s (A=%s, B=%s); skipping",
                layer_prefix,
                module_key,
                tuple(a.shape),
                tuple(b.shape),
            )
            continue
        # 3D MoE delta (n_experts, out, in): averaging over experts keeps
        # memory bounded on models with hundreds of experts (Qwen3.5-A3B has
        # 256). The mean-expert delta is a reasonable per-layer stand-in for
        # the gate's angle measurement — per-expert forgetting would need
        # its own metric.
        if delta.ndim == 3:
            deltas[module_key].append(delta.mean(dim=0))
        else:
            deltas[module_key].append(delta)
    for (layer_prefix, module_key) in b_matrices.keys():
        if (layer_prefix, module_key) not in a_matrices:
            logger.warning(
                "missing lora_A for %s%s; skipping", layer_prefix, module_key
            )
    return deltas


def _stack_group(deltas: Iterable["torch.Tensor"]) -> "torch.Tensor":
    """Flatten each layer's delta to a column and stack → (n_params, n_layers)."""
    import torch

    cols = [d.reshape(-1) for d in deltas]
    return torch.stack(cols, dim=1)


def compute_angles(
    prior_tensors: dict[str, "torch.Tensor"],
    new_tensors: dict[str, "torch.Tensor"],
    analyzer: GradientSubspaceAnalyzer | None = None,
) -> dict[str, float]:
    """Return a dict of per-module angles (degrees).

    Uses ``GradientSubspaceAnalyzer.compute_angle`` treating each module
    group's stacked per-layer deltas as the "gradient" matrix (columns =
    samples, rows = params). Module keys are discovered dynamically from
    the adapter contents (see ``_extract_deltas``); this covers all
    attention projections (``self_attn.{q,k,v,o}_proj``), linear-attention
    blocks (``linear_attn.{in_proj_a,in_proj_b,in_proj_qkv,in_proj_z,out_proj}``),
    MoE gates and shared/switch expert projections found in modern MoE
    adapters (e.g. Qwen3.5-35B-A3B trained via MLX-LM).
    """
    analyzer = analyzer or GradientSubspaceAnalyzer()
    prior_deltas = _extract_deltas(prior_tensors)
    new_deltas = _extract_deltas(new_tensors)

    # Intersect module keys present in both adapters — a missing module on
    # one side cannot contribute to an angle measurement.
    shared_keys = sorted(set(prior_deltas) & set(new_deltas))

    angles: dict[str, float] = {}
    for module_key in shared_keys:
        p_list = prior_deltas.get(module_key, [])
        n_list = new_deltas.get(module_key, [])
        if not p_list or not n_list:
            logger.debug("no deltas for %s; skipping", module_key)
            continue
        p_mat = _stack_group(p_list)
        n_mat = _stack_group(n_list)
        # Row-align if layer counts differ (truncate to min shared rows).
        if p_mat.shape[0] != n_mat.shape[0]:
            min_rows = min(p_mat.shape[0], n_mat.shape[0])
            p_mat = p_mat[:min_rows]
            n_mat = n_mat[:min_rows]
        # Column-align too: subspace angle requires matching sample count;
        # truncate to min shared layers when layer counts differ.
        if p_mat.shape[1] != n_mat.shape[1]:
            min_cols = min(p_mat.shape[1], n_mat.shape[1])
            p_mat = p_mat[:, :min_cols]
            n_mat = n_mat[:, :min_cols]
        angle = analyzer.compute_angle(p_mat, n_mat)
        angles[module_key] = angle
        logger.debug("%s: angle = %.3f°", module_key, angle)
    return angles


# ---------------------------------------------------------------------------
# Win-rate scoring
# ---------------------------------------------------------------------------


def _load_eval_dataset(path: Path) -> list[dict[str, str]]:
    """Load a JSONL file of ``{prompt, reference}`` entries.

    Matches the shape used by ``src/eval/stack_eval.py`` (``prompt`` key).
    The ``reference`` key is optional; entries without it are scored only
    against the prompt itself (used as fallback reference).
    """
    items: list[dict[str, str]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items


def _containment_score(output: str, reference: str) -> float:
    """Simple similarity score in [0, 1] based on token containment.

    - 1.0 if reference is non-empty and fully contained in output
    - fraction of reference tokens appearing in output otherwise
    - 0.0 if reference is empty
    """
    if not reference:
        return 0.0
    out = output.lower()
    ref = reference.lower()
    if ref in out:
        return 1.0
    ref_tokens = [t for t in re.split(r"\s+", ref) if t]
    if not ref_tokens:
        return 0.0
    hits = sum(1 for t in ref_tokens if t in out)
    return hits / len(ref_tokens)


def _resolve_generate_fn(module_path: str) -> Callable[..., str]:
    """Resolve ``module.path:callable`` (or ``module.path.callable``) to a callable.

    Raises:
        ImportError: on failure to import module
        AttributeError: if the attribute doesn't exist
    """
    if ":" in module_path:
        mod_name, attr = module_path.split(":", 1)
    else:
        mod_name, _, attr = module_path.rpartition(".")
        if not mod_name:
            raise ValueError(
                f"Cannot resolve callable from {module_path!r}; "
                "use 'pkg.module:callable' form."
            )
    module = importlib.import_module(mod_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise AttributeError(
            f"module {mod_name!r} has no attribute {attr!r}"
        ) from exc


def _compute_winrate(
    dataset: list[dict[str, str]],
    generate_fn: Callable[..., str],
    adapter_path: Path,
    threshold: float = 0.5,
) -> float:
    """Run ``generate_fn`` over prompts and compute the fraction passing threshold.

    For each entry, we call ``generate_fn(prompt=..., adapter=adapter_path)``
    and score containment of ``reference`` in the response. The win-rate is
    the fraction of prompts with score ≥ ``threshold``.

    If ``generate_fn`` does not accept an ``adapter`` keyword, it is called
    positionally with the prompt alone (useful for mocked closures in tests).
    """
    if not dataset:
        return 0.0

    wins = 0
    for entry in dataset:
        prompt = entry.get("prompt", "")
        reference = entry.get("reference", prompt)
        try:
            output = generate_fn(prompt=prompt, adapter=adapter_path)
        except TypeError:
            # Mock closures that only take the prompt.
            output = generate_fn(prompt)
        score = _containment_score(output, reference)
        if score >= threshold:
            wins += 1
    return wins / len(dataset)


# ---------------------------------------------------------------------------
# measure_forgetting_signal — shared helper (CLI + check_all_previous)
# ---------------------------------------------------------------------------


def measure_forgetting_signal(
    prior_adapter_path: Path,
    new_adapter_path: Path,
    eval_dataset: Path | str | None = None,
    generate_fn: Callable[..., str] | str | None = None,
    winrate_baseline: float | None = None,
    *,
    angle_threshold: float = ANGLE_THRESHOLD,
    winrate_drop_threshold: float = WINRATE_DROP_THRESHOLD,
    winrate_score_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute forgetting signal (angle + optional win-rate) between two adapters.

    This is the shared math used by ``scripts/measure_forgetting.py`` (CLI)
    and by ``ForgettingEvaluator.check_all_previous`` (library path).

    Args:
        prior_adapter_path: Path to prior stack ``adapter_model.safetensors``.
        new_adapter_path: Path to newly-trained adapter.
        eval_dataset: Optional path to a JSONL file with ``{prompt, reference}``
            entries. Required to compute the win-rate half of the gate.
        generate_fn: Either a callable ``fn(prompt=..., adapter=...) -> str``
            or a string ``"pkg.module:callable"`` resolved via ``importlib``.
            Required to compute win-rate.
        winrate_baseline: Reference win-rate from the prior adapter. Required
            to compute ``winrate_drop``.
        angle_threshold: Degrees below which the angle is considered unsafe.
        winrate_drop_threshold: Fractional drop above which win-rate signals
            regression.
        winrate_score_threshold: Per-prompt score cutoff used when counting
            wins (passed through to ``_compute_winrate``).

    Returns:
        Dict with the JSON-ready fields:
          - ``angle_degrees_mean`` (float or NaN)
          - ``angle_degrees_per_module`` (dict[str, float])
          - ``winrate_measured`` (float or None)
          - ``winrate_baseline`` (float or None)
          - ``winrate_drop`` (float or None)
          - ``gate_status`` ("pass" | "fail" | "angle_only_partial") —
            legacy aggregate-only status, kept for back-compat.
          - ``gate_status_aggregate`` (same values) — the coarse gate
            (mean angle across modules).
          - ``gate_status_per_module`` (same values) — the fine-grained
            gate (any single module < threshold).
          - ``offending_modules`` (list[str]) — modules with angle
            below threshold under the per-module gate (``mlp.
            shared_expert_gate`` is ignored by default; see
            :data:`DEFAULT_PER_MODULE_IGNORE`).
          - ``min_angle_module`` (str) — lowest-angle non-ignored
            module, ``""`` if none.
          - ``min_angle_value`` (float) — lowest angle, ``nan`` if no
            modules.
          - ``warning`` (str or None)
          - ``note`` (str, context)
    """
    prior_tensors = _load_tensors(Path(prior_adapter_path))
    new_tensors = _load_tensors(Path(new_adapter_path))
    angles = compute_angles(prior_tensors, new_tensors)

    if not angles:
        return {
            "angle_degrees_mean": float("nan"),
            "angle_degrees_per_module": {},
            "winrate_measured": None,
            "winrate_baseline": None,
            "winrate_drop": None,
            "warning": "no matching LoRA layers found in either adapter",
            "gate_status": "angle_only_partial",
            "gate_status_aggregate": "angle_only_partial",
            "gate_status_per_module": "angle_only_partial",
            "offending_modules": [],
            "min_angle_module": "",
            "min_angle_value": float("nan"),
            "note": "Win-rate half not measured; run paired eval for full gate.",
        }

    import numpy as np

    mean_angle = float(np.mean(list(angles.values())))
    per_module = {k: float(v) for k, v in angles.items()}

    # Angle-only path (phase 1a behaviour): no win-rate inputs → partial gate.
    winrate_inputs_provided = (
        eval_dataset is not None
        and generate_fn is not None
        and winrate_baseline is not None
    )
    if not winrate_inputs_provided:
        # Per-module partial view: drop=None means never fail, but still
        # surface the lowest module for diagnostics.
        pm_partial = apply_per_module_gate(
            per_module,
            winrate_drop=None,
            angle_threshold=angle_threshold,
            winrate_drop_threshold=winrate_drop_threshold,
        )
        warning: str | None = None
        if mean_angle < angle_threshold:
            warning = "angle below threshold"
        return {
            "angle_degrees_mean": mean_angle,
            "angle_degrees_per_module": per_module,
            "winrate_measured": None,
            "winrate_baseline": None,
            "winrate_drop": None,
            "warning": warning,
            "gate_status": "angle_only_partial",
            "gate_status_aggregate": "angle_only_partial",
            "gate_status_per_module": "angle_only_partial",
            "offending_modules": list(pm_partial.offending_modules),
            "min_angle_module": pm_partial.min_angle_module,
            "min_angle_value": pm_partial.min_angle_value,
            "note": "Win-rate half not measured; run paired eval for full gate.",
        }

    # Full gate path (phase 1b): compute win-rate, apply AND-logic rollback.
    if isinstance(generate_fn, str):
        gen_callable = _resolve_generate_fn(generate_fn)
    else:
        gen_callable = generate_fn

    dataset = _load_eval_dataset(Path(eval_dataset))
    measured = _compute_winrate(
        dataset=dataset,
        generate_fn=gen_callable,
        adapter_path=Path(new_adapter_path),
        threshold=winrate_score_threshold,
    )
    baseline = float(winrate_baseline)
    drop = baseline - measured

    decision = apply_and_gate_detailed(
        mean_angle, drop, angle_threshold, winrate_drop_threshold
    )
    pm_decision = apply_per_module_gate(
        per_module,
        winrate_drop=drop,
        angle_threshold=angle_threshold,
        winrate_drop_threshold=winrate_drop_threshold,
    )
    gate_status_aggregate = "fail" if decision.failed else "pass"
    gate_status_per_module = "fail" if pm_decision.failed else "pass"
    warning_bits = []
    if decision.angle_bad:
        warning_bits.append(f"angle {mean_angle:.2f}° < {angle_threshold}°")
    if decision.delta_bad:
        warning_bits.append(f"winrate_drop {drop:.3f} > {winrate_drop_threshold}")
    if pm_decision.failed and pm_decision.offending_modules:
        warning_bits.append(
            "per-module: " + ",".join(pm_decision.offending_modules)
        )
    warning = "; ".join(warning_bits) or None

    return {
        "angle_degrees_mean": mean_angle,
        "angle_degrees_per_module": per_module,
        "winrate_measured": float(measured),
        "winrate_baseline": baseline,
        "winrate_drop": float(drop),
        "warning": warning,
        # Legacy aggregate-only status (unchanged behaviour).
        "gate_status": gate_status_aggregate,
        "gate_status_aggregate": gate_status_aggregate,
        "gate_status_per_module": gate_status_per_module,
        "offending_modules": list(pm_decision.offending_modules),
        "min_angle_module": pm_decision.min_angle_module,
        "min_angle_value": pm_decision.min_angle_value,
        "note": (
            "Full gate: aggregate (mean angle) AND per-module (any "
            "non-ignored module < 30°) both enforced with winrate_drop>0.03."
        ),
    }


# ---------------------------------------------------------------------------
# Forgetting evaluator
# ---------------------------------------------------------------------------


class ForgettingEvaluator:
    """Orchestrates forgetting checks using StackEvaluator + GradientSubspaceAnalyzer."""

    def __init__(
        self,
        stack_evaluator: object,
        analyzer: GradientSubspaceAnalyzer | None = None,
        angle_threshold: float = ANGLE_THRESHOLD,
        winrate_drop_threshold: float = WINRATE_DROP_THRESHOLD,
    ) -> None:
        self._evaluator = stack_evaluator
        self._analyzer = analyzer or GradientSubspaceAnalyzer()
        self._angle_threshold = angle_threshold
        self._winrate_drop_threshold = winrate_drop_threshold

    def _make_report(
        self,
        stack_id: str,
        new_stack_id: str,
        angle: float,
        winrate_base: float,
        winrate_adapted: float,
    ) -> ForgettingReport:
        winrate_drop = winrate_base - winrate_adapted
        should_rollback = apply_and_gate(
            angle,
            winrate_drop,
            self._angle_threshold,
            self._winrate_drop_threshold,
        )
        return ForgettingReport(
            stack_id=stack_id,
            new_stack_id=new_stack_id,
            angle=angle,
            winrate_base=winrate_base,
            winrate_adapted=winrate_adapted,
            winrate_drop=winrate_drop,
            passed=not should_rollback,
            should_rollback=should_rollback,
        )

    def check_stack(
        self,
        stack_id: str,
        new_stack_id: str,
        eval_data_path: Path,
        angle: float | None = None,
        winrate_base: float | None = None,
        winrate_adapted: float | None = None,
    ) -> ForgettingReport:
        """Check if new_stack causes forgetting on stack_id.

        When angle / winrate values are provided directly, uses those
        (useful for testing and when measurements come from external
        tooling). Otherwise delegates to the analyzer and evaluator.

        Args:
            stack_id: previously-trained stack to check regression on
            new_stack_id: the newly-trained stack
            eval_data_path: path to JSONL eval data for stack_id
            angle: pre-computed gradient subspace angle (optional)
            winrate_base: pre-computed baseline win-rate (optional)
            winrate_adapted: pre-computed adapted win-rate (optional)

        Returns:
            ForgettingReport with pass/rollback decision
        """
        if angle is None or winrate_base is None or winrate_adapted is None:
            raise ValueError(
                "Direct measurement mode requires angle, winrate_base, "
                "and winrate_adapted. Automatic measurement requires a "
                "running model — use scripts/run_forgetting.sh instead."
            )

        report = self._make_report(
            stack_id=stack_id,
            new_stack_id=new_stack_id,
            angle=angle,
            winrate_base=winrate_base,
            winrate_adapted=winrate_adapted,
        )

        if report.should_rollback:
            logger.warning(
                "FORGETTING on %s after %s: angle=%.1f° (<%.1f°), "
                "wr_drop=%.3f (>%.3f) -> ROLLBACK",
                stack_id,
                new_stack_id,
                report.angle,
                self._angle_threshold,
                report.winrate_drop,
                self._winrate_drop_threshold,
            )
        else:
            logger.info(
                "Forgetting check %s vs %s: angle=%.1f°, wr_drop=%.3f -> PASS",
                stack_id,
                new_stack_id,
                report.angle,
                report.winrate_drop,
            )

        return report

    def check_all_previous(
        self,
        trained_stacks: list[str],
        new_stack_id: str,
        eval_data_dir: Path | None = None,
        results: list[dict] | None = None,
        *,
        adapter_path_fn: Callable[[str], Path] | None = None,
        new_adapter_path: Path | None = None,
        generate_fn: Callable[..., str] | str | None = None,
        winrate_baselines: dict[str, float] | None = None,
    ) -> list[ForgettingReport]:
        """Run forgetting check for each previously-trained stack.

        Two modes:

        1. **Pre-computed** (legacy): pass ``results=[...]`` with dicts
           containing ``stack_id``, ``angle``, ``winrate_base``,
           ``winrate_adapted``. Each entry becomes one report.

        2. **Automatic** (phase 1b): pass ``adapter_path_fn`` (maps
           ``stack_id`` → safetensors path) and ``new_adapter_path``;
           optionally ``generate_fn`` + ``winrate_baselines`` +
           ``eval_data_dir`` to also compute the win-rate half.
           For each prior stack, ``measure_forgetting_signal`` is called
           and the resulting dict is folded into a ``ForgettingReport``.

        Args:
            trained_stacks: list of stack IDs trained before new_stack_id
            new_stack_id: the newly-trained stack
            eval_data_dir: directory containing {stack_id}.jsonl files
            results: pre-computed list of dicts (mode 1)
            adapter_path_fn: (mode 2) ``stack_id -> Path`` resolver
            new_adapter_path: (mode 2) path to the new adapter's safetensors
            generate_fn: (mode 2) optional win-rate generation callable
            winrate_baselines: (mode 2) map ``stack_id -> baseline_winrate``

        Returns:
            List of ForgettingReport, one per prior stack.
        """
        reports: list[ForgettingReport] = []

        if results is not None:
            # Mode 1: pre-computed measurements.
            for entry in results:
                report = self._make_report(
                    stack_id=entry["stack_id"],
                    new_stack_id=new_stack_id,
                    angle=entry["angle"],
                    winrate_base=entry["winrate_base"],
                    winrate_adapted=entry["winrate_adapted"],
                )
                reports.append(report)
        else:
            # Mode 2: automatic measurement via measure_forgetting_signal.
            if adapter_path_fn is None or new_adapter_path is None:
                raise ValueError(
                    "check_all_previous automatic mode requires "
                    "adapter_path_fn and new_adapter_path. For pre-computed "
                    "measurements pass results=[...] instead."
                )

            baselines = winrate_baselines or {}
            eval_dir = Path(eval_data_dir) if eval_data_dir else None

            for stack_id in trained_stacks:
                prior_path = Path(adapter_path_fn(stack_id))
                eval_dataset = (
                    eval_dir / f"{stack_id}.jsonl" if eval_dir else None
                )
                if eval_dataset is not None and not eval_dataset.exists():
                    eval_dataset = None
                baseline = baselines.get(stack_id)

                signal = measure_forgetting_signal(
                    prior_adapter_path=prior_path,
                    new_adapter_path=Path(new_adapter_path),
                    eval_dataset=eval_dataset if generate_fn else None,
                    generate_fn=generate_fn,
                    winrate_baseline=baseline,
                    angle_threshold=self._angle_threshold,
                    winrate_drop_threshold=self._winrate_drop_threshold,
                )

                # Map the signal dict into a ForgettingReport. When win-rate
                # wasn't computed, record zeros so the dataclass stays
                # populated and `passed` is driven solely by the signal's
                # gate_status (angle_only_partial → passed=True).
                wr_measured = signal.get("winrate_measured")
                wr_baseline = signal.get("winrate_baseline")
                wr_drop = signal.get("winrate_drop")
                gate_status_agg = signal.get(
                    "gate_status_aggregate",
                    signal.get("gate_status", "angle_only_partial"),
                )
                gate_status_pm = signal.get(
                    "gate_status_per_module", gate_status_agg
                )

                if wr_measured is None or wr_baseline is None:
                    winrate_base_val = 0.0
                    winrate_adapted_val = 0.0
                    winrate_drop_val = 0.0
                else:
                    winrate_base_val = float(wr_baseline)
                    winrate_adapted_val = float(wr_measured)
                    winrate_drop_val = float(wr_drop or 0.0)

                # Rollback on EITHER aggregate or per-module failure
                # (stricter is safer — matches CLI exit policy).
                should_rollback = (
                    gate_status_agg == "fail" or gate_status_pm == "fail"
                )
                reports.append(
                    ForgettingReport(
                        stack_id=stack_id,
                        new_stack_id=new_stack_id,
                        angle=float(signal.get("angle_degrees_mean", 0.0)),
                        winrate_base=winrate_base_val,
                        winrate_adapted=winrate_adapted_val,
                        winrate_drop=winrate_drop_val,
                        passed=not should_rollback,
                        should_rollback=should_rollback,
                    )
                )

        any_rollback = any(r.should_rollback for r in reports)
        if any_rollback:
            failed = [r.stack_id for r in reports if r.should_rollback]
            logger.warning(
                "ROLLBACK TRIGGERED: %d stack(s) show forgetting: %s",
                len(failed),
                ", ".join(failed),
            )
        else:
            logger.info(
                "All %d forgetting checks passed for %s",
                len(reports),
                new_stack_id,
            )

        return reports


# ---------------------------------------------------------------------------
# Legacy compat functions (used by existing code/tests)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForgettingCheckResult:
    """Legacy result type — prefer ForgettingReport for new code."""

    prior_stack: str
    new_stack: str
    win_rate_delta: float
    gradient_subspace_angle_deg: float
    passed: bool
    reason: str


def check_forgetting(
    win_rate_delta: float,
    angle_deg: float,
    prior_stack: str,
    new_stack: str,
    max_delta: float = 0.03,
    min_angle: float = 30.0,
) -> ForgettingCheckResult:
    """Check if new stack causes forgetting on prior stack.

    Fails only if BOTH conditions are true:
    - win_rate_delta > max_delta (regression on prior stack)
    - gradient_subspace_angle < min_angle (high interference)
    """
    # win_rate_delta is already a drop (baseline - measured), so it maps
    # directly onto apply_and_gate_detailed's ``winrate_drop`` axis.
    failed, angle_bad, delta_bad = apply_and_gate_detailed(
        angle=angle_deg,
        winrate_drop=win_rate_delta,
        angle_threshold=min_angle,
        winrate_drop_threshold=max_delta,
    )

    if failed:
        reason = f"FORGETTING: delta={win_rate_delta:.3f}>{max_delta}, angle={angle_deg:.1f}<{min_angle}"
    elif delta_bad:
        reason = f"Delta high ({win_rate_delta:.3f}) but angle safe ({angle_deg:.1f}°)"
    elif angle_bad:
        reason = f"Angle low ({angle_deg:.1f}°) but delta safe ({win_rate_delta:.3f})"
    else:
        reason = "OK"

    return ForgettingCheckResult(
        prior_stack=prior_stack,
        new_stack=new_stack,
        win_rate_delta=win_rate_delta,
        gradient_subspace_angle_deg=angle_deg,
        passed=not failed,
        reason=reason,
    )


def compute_subspace_angle(lora_a_new: torch.Tensor, lora_a_prior: torch.Tensor) -> float:
    """Compute angle between LoRA update subspaces (legacy wrapper)."""
    analyzer = GradientSubspaceAnalyzer()
    return analyzer.compute_angle(lora_a_new, lora_a_prior)


def save_forgetting_report(
    results: list[ForgettingCheckResult],
    new_stack: str,
    output_dir: str = "results",
) -> Path:
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    path = output_dir_path / f"forgetting-{new_stack}.json"
    data = {
        "new_stack": new_stack,
        "timestamp": datetime.now().isoformat(),
        "all_passed": all(r.passed for r in results),
        "checks": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(data, indent=2))
    logger.info("Forgetting report saved to %s", path)
    return path


# ---------------------------------------------------------------------------
# CLI entry point: python -m src.eval.forgetting
# ---------------------------------------------------------------------------


def main() -> int:
    """CLI entry point for forgetting checks."""
    parser = argparse.ArgumentParser(
        description="Run forgetting check for a new stack against previous stacks",
    )
    parser.add_argument("new_stack_id", help="ID of the newly-trained stack")
    parser.add_argument(
        "--all-previous",
        action="store_true",
        help="Check against all previously-trained stacks",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        help="JSON file with pre-computed measurements",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for output reports",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.results_file:
        logger.error("--results-file is required (automatic measurement not yet wired)")
        return 1

    measurements = json.loads(args.results_file.read_text())
    if not isinstance(measurements, list):
        measurements = measurements.get("checks", [measurements])

    evaluator = ForgettingEvaluator(
        stack_evaluator=None,
        analyzer=GradientSubspaceAnalyzer(),
    )

    reports = evaluator.check_all_previous(
        trained_stacks=[],
        new_stack_id=args.new_stack_id,
        results=[
            {
                "stack_id": m.get("stack_id", m.get("prior_stack", "unknown")),
                "angle": m.get("angle", m.get("gradient_subspace_angle_deg", 0.0)),
                "winrate_base": m.get("winrate_base", 0.0),
                "winrate_adapted": m.get("winrate_adapted", 0.0),
            }
            for m in measurements
        ],
    )

    # Save report
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"forgetting-{args.new_stack_id}.json"
    report_data = {
        "new_stack": args.new_stack_id,
        "timestamp": datetime.now().isoformat(),
        "all_passed": all(r.passed for r in reports),
        "reports": [asdict(r) for r in reports],
    }
    report_path.write_text(json.dumps(report_data, indent=2))
    logger.info("Report saved to %s", report_path)

    any_rollback = any(r.should_rollback for r in reports)
    if any_rollback:
        logger.warning("ROLLBACK TRIGGERED — exit code 1")
        return 1

    logger.info("All checks passed — exit code 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())

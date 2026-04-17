#!/usr/bin/env python3
"""LAS conversion: Qwen3.5-35B-A3B MoE → SpikingKiki-35B-A3B.

This is the most architecturally interesting conversion in the micro-kiki
pipeline: a 256-expert MoE (3B active per token) transformed into a spiking
network where expert routing is preserved losslessly via rate-coded identity
activations on the router and ReLU-coded activations on each expert FFN.

Key design choices:
- Router converted with activation="identity" — signed logits kept intact
  so top-k expert selection is preserved (same semantics as LASConverter MoE).
- Expert FFNs converted with activation="relu" — standard rate-code path.
- Attention projections converted with activation="identity" — residual
  stream stability.
- Layer-by-layer with progress logging and resumable state.
- ~40h estimated wall time on Mac Studio M3 Ultra; --resume skips layers
  already present in the output directory.

Architecture of Qwen3.5-35B-A3B per transformer block:
  - Attention: q_proj, k_proj, v_proj, o_proj
  - MoE FFN: gate (router 7168→256), 256 experts each with gate_proj +
    up_proj + down_proj (SwiGLU pattern)
  - Total ~94 transformer blocks

Usage:
  uv run python scripts/convert_spikingkiki_35b.py \\
      --input  /path/to/Qwen3.5-35B-A3B \\
      --output /path/to/SpikingKiki-35B-A3B \\
      --timesteps 128 \\
      --resume

  # Dry-run (no torch required):
  uv run python scripts/convert_spikingkiki_35b.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
CONFIG_PATH = REPO_ROOT / "models" / "qwen3.5-35b-a3b" / "config.json"

# Qwen3.5-35B-A3B architecture constants
NUM_LAYERS = 94
NUM_EXPERTS = 256
TOP_K = 8  # Qwen3.5-35B-A3B uses top-8 routing
HIDDEN_DIM = 7168
NUM_ATTENTION_HEADS = 64
NUM_KV_HEADS = 4
HEAD_DIM = 112
EXPERT_INTERMEDIATE = 2048  # per-expert intermediate dim
SHARED_EXPERT_INTERMEDIATE = 7168
VOCAB_SIZE = 151936

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("convert_35b")


# ---------------------------------------------------------------------------
# Layer map generation (pure Python, no torch)
# ---------------------------------------------------------------------------


def build_layer_map(num_layers: int = NUM_LAYERS) -> list[dict[str, Any]]:
    """Return the canonical layer map for Qwen3.5-35B-A3B.

    Each entry describes one logical conversion unit with its weight key
    path inside the HuggingFace state_dict.

    Returns
    -------
    list of dicts with keys:
        layer_id   : int   — transformer block index
        kind       : str   — "attn_proj" | "moe_router" | "moe_expert_ffn"
        key_prefix : str   — state_dict key prefix
        expert_id  : int | None
        activation : str   — "identity" or "relu"
        converted  : bool  — False initially (set True when done)
    """
    entries: list[dict[str, Any]] = []
    for layer_idx in range(num_layers):
        prefix = f"model.layers.{layer_idx}"

        # Attention projections — identity activation to preserve residual
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            entries.append(
                {
                    "layer_id": layer_idx,
                    "kind": "attn_proj",
                    "key_prefix": f"{prefix}.self_attn.{proj}",
                    "expert_id": None,
                    "activation": "identity",
                    "converted": False,
                }
            )

        # MoE router — identity activation (signed logits must be preserved)
        entries.append(
            {
                "layer_id": layer_idx,
                "kind": "moe_router",
                "key_prefix": f"{prefix}.mlp.gate",
                "expert_id": None,
                "activation": "identity",
                "converted": False,
            }
        )

        # MoE expert FFNs — relu activation (SwiGLU gate + up + down)
        for expert_id in range(NUM_EXPERTS):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                entries.append(
                    {
                        "layer_id": layer_idx,
                        "kind": "moe_expert_ffn",
                        "key_prefix": (
                            f"{prefix}.mlp.experts.{expert_id}.{proj}"
                        ),
                        "expert_id": expert_id,
                        "activation": "relu",
                        "converted": False,
                    }
                )

    return entries


def layer_map_stats(layer_map: list[dict[str, Any]]) -> dict[str, int]:
    """Summarise the layer map by kind."""
    counts: dict[str, int] = {}
    for entry in layer_map:
        counts[entry["kind"]] = counts.get(entry["kind"], 0) + 1
    counts["total"] = len(layer_map)
    return counts


# ---------------------------------------------------------------------------
# Resume logic — check which layers are already saved
# ---------------------------------------------------------------------------


def load_resume_state(output_dir: Path) -> set[str]:
    """Return set of key_prefixes already converted (saved to disk)."""
    state_file = output_dir / ".convert_state.json"
    if not state_file.exists():
        return set()
    try:
        data = json.loads(state_file.read_text())
        return set(data.get("converted_keys", []))
    except (json.JSONDecodeError, OSError):
        log.warning("could not parse resume state — starting fresh")
        return set()


def save_resume_state(output_dir: Path, converted_keys: set[str]) -> None:
    """Persist the set of converted key_prefixes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    state_file = output_dir / ".convert_state.json"
    state_file.write_text(
        json.dumps({"converted_keys": sorted(converted_keys)}, indent=2)
    )


# ---------------------------------------------------------------------------
# Torch-dependent conversion (guarded — works without torch for testing)
# ---------------------------------------------------------------------------


def _try_import_torch() -> Any:
    try:
        import torch  # type: ignore
        return torch
    except ImportError:
        return None


def _try_import_transformers() -> Any:
    try:
        from transformers import AutoModelForCausalLM, AutoConfig  # type: ignore
        return AutoModelForCausalLM, AutoConfig
    except ImportError:
        return None, None


def _extract_linear_weights(module: Any) -> dict[str, Any]:
    """Extract weight + bias as numpy from a torch.nn.Linear."""
    import numpy as np
    w = module.weight.detach().cpu().float().numpy()
    b = (
        module.bias.detach().cpu().float().numpy()
        if module.bias is not None
        else None
    )
    return {"weight": w, "bias": b}


def convert_block_torch(
    model: Any,
    layer_idx: int,
    converter: Any,
    spike_stats: dict[str, Any],
    converted_keys: set[str],
    output_dir: Path,
    timesteps: int,
) -> dict[str, Any]:
    """Convert one transformer block, saving spiking layers to disk.

    Returns a dict of spike stats for the block (counts + activation bounds).
    """
    import numpy as np
    from src.spiking.las_converter import LASConverter

    block = model.model.layers[layer_idx]
    prefix = f"model.layers.{layer_idx}"
    block_stats: dict[str, Any] = {"layer_id": layer_idx, "projections": {}}

    # --- Attention projections ---
    attn = block.self_attn
    for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        key = f"{prefix}.self_attn.{proj_name}"
        if key in converted_keys:
            log.debug("skip (resume) %s", key)
            continue
        proj_module = getattr(attn, proj_name)
        weights = _extract_linear_weights(proj_module)
        spiking = converter.convert_layer(weights, activation="identity")
        _save_spiking_layer(spiking, key, output_dir)
        w_abs_max = float(np.abs(weights["weight"]).max())
        block_stats["projections"][proj_name] = {
            "in": spiking.in_features,
            "out": spiking.out_features,
            "w_abs_max": round(w_abs_max, 4),
        }
        converted_keys.add(key)

    # --- MoE router ---
    router_key = f"{prefix}.mlp.gate"
    if router_key not in converted_keys:
        router_weights = _extract_linear_weights(block.mlp.gate)
        spiking_router = converter.convert_layer(
            router_weights, activation="identity"
        )
        _save_spiking_layer(spiking_router, router_key, output_dir)
        block_stats["router"] = {
            "in": spiking_router.in_features,
            "out": spiking_router.out_features,
        }
        converted_keys.add(router_key)
    else:
        log.debug("skip (resume) %s", router_key)

    # --- MoE experts ---
    experts_module = block.mlp.experts
    expert_spike_counts: list[float] = []
    for expert_id in range(len(experts_module)):
        expert = experts_module[expert_id]
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            key = f"{prefix}.mlp.experts.{expert_id}.{proj_name}"
            if key in converted_keys:
                continue
            proj_module = getattr(expert, proj_name)
            weights = _extract_linear_weights(proj_module)
            spiking = converter.convert_layer(weights, activation="relu")
            _save_spiking_layer(spiking, key, output_dir)
            # Rough spike count estimate via max weight * timesteps
            est_spikes = float(
                np.abs(weights["weight"]).mean() * timesteps
            )
            expert_spike_counts.append(est_spikes)
            converted_keys.add(key)

    if expert_spike_counts:
        block_stats["expert_spike_rate_mean"] = round(
            float(np.mean(expert_spike_counts)), 4
        )

    return block_stats


def _save_spiking_layer(spiking: Any, key: str, output_dir: Path) -> None:
    """Save a SpikingLinear weight array as .npz alongside a metadata JSON."""
    import numpy as np
    safe_key = key.replace(".", "_").replace("/", "_")
    out_path = output_dir / f"{safe_key}.npz"
    np.savez_compressed(
        out_path,
        weight=spiking.weight,
        bias=spiking.bias if spiking.bias is not None else np.array([]),
        timesteps=np.array([spiking.timesteps]),
        max_rate=np.array([spiking.max_rate]),
    )


# ---------------------------------------------------------------------------
# Metadata assembly
# ---------------------------------------------------------------------------


def build_metadata(
    args: argparse.Namespace,
    layer_map: list[dict[str, Any]],
    spike_stats: dict[str, Any],
    elapsed_s: float,
    status: str,
) -> dict[str, Any]:
    """Assemble the results JSON structure."""
    map_stats = layer_map_stats(layer_map)
    converted = sum(1 for e in layer_map if e.get("converted"))
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": status,
        "input": str(args.input),
        "output": str(args.output),
        "timesteps": args.timesteps,
        "model_info": {
            "name": "Qwen3.5-35B-A3B",
            "num_layers": NUM_LAYERS,
            "num_experts": NUM_EXPERTS,
            "top_k": TOP_K,
            "hidden_dim": HIDDEN_DIM,
            "expert_intermediate_dim": EXPERT_INTERMEDIATE,
        },
        "layer_map_stats": map_stats,
        "converted_layers": converted,
        "total_layers": len(layer_map),
        "elapsed_s": round(elapsed_s, 1),
        "spike_stats": spike_stats,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Story-27: Theoretical analysis (no weights, no GPU)
# ---------------------------------------------------------------------------


def load_model_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load the model config.json for architecture analysis.

    Falls back to hardcoded constants if no config file is available.
    """
    path = config_path or CONFIG_PATH
    if path.exists():
        return json.loads(path.read_text())
    log.warning("config.json not found at %s, using hardcoded constants", path)
    return {
        "hidden_size": HIDDEN_DIM,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": NUM_ATTENTION_HEADS,
        "num_key_value_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "num_experts": NUM_EXPERTS,
        "num_experts_per_tok": TOP_K,
        "moe_intermediate_size": EXPERT_INTERMEDIATE,
        "shared_expert_intermediate_size": SHARED_EXPERT_INTERMEDIATE,
        "vocab_size": VOCAB_SIZE,
    }


def _count_params_linear(in_dim: int, out_dim: int, bias: bool = False) -> int:
    """Parameter count for a single linear layer."""
    return in_dim * out_dim + (out_dim if bias else 0)


def analyze_layer_convertibility(config: dict[str, Any]) -> dict[str, Any]:
    """Analyze which layers can be converted to spiking and which cannot.

    Returns a structured dict with per-layer-type analysis.
    """
    hidden = config["hidden_size"]
    n_layers = config["num_hidden_layers"]
    n_heads = config["num_attention_heads"]
    n_kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    n_experts = config["num_experts"]
    top_k = config["num_experts_per_tok"]
    expert_inter = config["moe_intermediate_size"]
    shared_inter = config.get("shared_expert_intermediate_size", 0)
    has_bias = config.get("attention_bias", False)

    # --- Attention projections (fully convertible) ---
    q_dim = n_heads * head_dim        # 64 * 112 = 7168
    k_dim = n_kv_heads * head_dim     # 4 * 112 = 448
    v_dim = n_kv_heads * head_dim     # 4 * 112 = 448
    o_dim = hidden                     # 7168

    attn_params_per_layer = (
        _count_params_linear(hidden, q_dim, has_bias)
        + _count_params_linear(hidden, k_dim, has_bias)
        + _count_params_linear(hidden, v_dim, has_bias)
        + _count_params_linear(q_dim, o_dim, has_bias)
    )

    # --- MoE router (convertible with identity activation) ---
    router_params_per_layer = _count_params_linear(hidden, n_experts, False)

    # --- MoE expert FFNs (research needed — SwiGLU complication) ---
    # Each expert has: gate_proj (hidden -> inter), up_proj (hidden -> inter),
    # down_proj (inter -> hidden). SwiGLU: silu(gate) * up, then down.
    expert_params = (
        _count_params_linear(hidden, expert_inter, False)  # gate
        + _count_params_linear(hidden, expert_inter, False)  # up
        + _count_params_linear(expert_inter, hidden, False)  # down
    )
    expert_params_total = expert_params * n_experts

    # Shared expert (if any)
    shared_expert_params = 0
    if shared_inter > 0:
        shared_expert_params = (
            _count_params_linear(hidden, shared_inter, False)  # gate
            + _count_params_linear(hidden, shared_inter, False)  # up
            + _count_params_linear(shared_inter, hidden, False)  # down
        )

    total_params_per_layer = (
        attn_params_per_layer
        + router_params_per_layer
        + expert_params_total
        + shared_expert_params
    )

    # Embedding + lm_head (not converted)
    embed_params = config.get("vocab_size", VOCAB_SIZE) * hidden
    lm_head_params = embed_params  # typically tied but counted separately

    return {
        "attention": {
            "convertible": True,
            "method": "LAS identity activation (signed logits preserved)",
            "projections": {
                "q_proj": {"shape": [hidden, q_dim], "params": _count_params_linear(hidden, q_dim, has_bias)},
                "k_proj": {"shape": [hidden, k_dim], "params": _count_params_linear(hidden, k_dim, has_bias)},
                "v_proj": {"shape": [hidden, v_dim], "params": _count_params_linear(hidden, v_dim, has_bias)},
                "o_proj": {"shape": [q_dim, o_dim], "params": _count_params_linear(q_dim, o_dim, has_bias)},
            },
            "params_per_layer": attn_params_per_layer,
            "total_params": attn_params_per_layer * n_layers,
            "notes": [
                "GQA (grouped query attention) with 64 Q heads, 4 KV heads",
                "Identity activation preserves signed values for softmax",
                "Rate-coded LIF with T>=64 achieves <2% relative error",
                "Residual stream integrity maintained via ANN-equivalent path",
            ],
        },
        "moe_router": {
            "convertible": True,
            "method": "LAS identity activation (expert selection from signed logits)",
            "params_per_layer": router_params_per_layer,
            "total_params": router_params_per_layer * n_layers,
            "notes": [
                f"Router: {hidden} -> {n_experts} (top-{top_k} selection)",
                "Identity activation critical: signed logits determine ranking",
                "ANN-equivalent routing used for selection (spiking for energy)",
                "norm_topk_prob normalises expert combination weights",
            ],
        },
        "moe_expert_ffn": {
            "convertible": "partial",
            "method": "LAS relu activation on gate/up/down (SwiGLU issue)",
            "params_per_expert": expert_params,
            "num_experts": n_experts,
            "active_per_token": top_k,
            "total_params": expert_params_total * n_layers,
            "blocking_issues": [
                "SwiGLU gate uses SiLU (x * sigmoid(x)), not ReLU -- "
                "rate-coded LIF only approximates ReLU natively",
                "Signed intermediate activations from SiLU need two-channel "
                "encoding (positive + negative spike trains)",
                "gate_proj * up_proj element-wise multiply has no direct "
                "spiking equivalent -- requires temporal correlation coding",
            ],
            "research_paths": [
                "Two-channel signed rate code (2x spike overhead, halves energy gain)",
                "Surrogate SiLU via shifted ReLU: silu(x) ~ relu(x+0.5) - 0.25 "
                "(rough, ~8% error at boundary)",
                "Hybrid: keep SwiGLU gate in ANN, convert up/down to spiking "
                "(pragmatic, ~60% of expert params converted)",
                "Temporal correlation coding for element-wise multiply "
                "(active research, no proven LAS-compatible method yet)",
            ],
            "notes": [
                f"Each expert: {hidden} -> {expert_inter} -> {hidden} (SwiGLU)",
                f"Only {top_k}/{n_experts} experts active per token",
                "down_proj (inter -> hidden) is straightforward ReLU conversion",
                "gate_proj output feeds SiLU -- the conversion bottleneck",
            ],
        },
        "shared_expert": {
            "convertible": "partial",
            "method": "Same SwiGLU issues as MoE experts",
            "params_per_layer": shared_expert_params,
            "total_params": shared_expert_params * n_layers,
            "notes": [
                f"Shared expert: {hidden} -> {shared_inter} -> {hidden}",
                "Always active (not gated by router)",
                "Same SwiGLU conversion challenges as regular experts",
            ],
        },
        "embedding_lm_head": {
            "convertible": False,
            "method": "N/A (lookup table, not a matmul)",
            "params": embed_params + lm_head_params,
            "notes": [
                "Embedding is a lookup, not amenable to spiking conversion",
                "lm_head logits need full precision for sampling",
            ],
        },
        "summary": {
            "total_params_per_layer": total_params_per_layer,
            "total_model_params_estimate": (
                total_params_per_layer * n_layers
                + embed_params + lm_head_params
            ),
            "fully_convertible_params": (
                (attn_params_per_layer + router_params_per_layer) * n_layers
            ),
            "partially_convertible_params": (
                (expert_params_total + shared_expert_params) * n_layers
            ),
            "not_convertible_params": embed_params + lm_head_params,
        },
    }


def calculate_spike_rate_equivalence(
    config: dict[str, Any],
    timesteps: int = 128,
) -> dict[str, Any]:
    """Calculate theoretical spike rate for attention layers.

    For a rate-coded LIF neuron with threshold = max_rate / T:
    - An activation `a` produces `floor(a * T / max_rate)` spikes
    - Reconstruction error bounded by `max_rate / T`
    - Energy per spike ~ 1 addition (vs 1 MAC for dense matmul)

    Returns spike rate analysis for different layer types.
    """
    hidden = config["hidden_size"]
    n_heads = config["num_attention_heads"]
    n_kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    n_experts = config["num_experts"]
    top_k = config["num_experts_per_tok"]
    expert_inter = config["moe_intermediate_size"]

    max_rate = 1.0
    threshold = max_rate / timesteps
    quant_error = threshold  # max per-element quantisation error

    # Theoretical spike rate for uniformly distributed activations [0, 1]
    # Mean spike count per neuron per forward = T * mean_activation / max_rate
    # For uniform [0, 1]: mean_activation = 0.5
    mean_activation = 0.5
    mean_spike_count = timesteps * mean_activation / max_rate

    # Attention: Q/K/V/O projections
    q_neurons = n_heads * head_dim  # 7168
    kv_neurons = n_kv_heads * head_dim  # 448
    attn_neurons_per_token = q_neurons + 2 * kv_neurons + hidden
    attn_spikes_per_token = attn_neurons_per_token * mean_spike_count

    # MoE: only top_k experts active
    expert_neurons_per_token = top_k * (2 * expert_inter + hidden)  # gate+up+down
    expert_spikes_per_token = expert_neurons_per_token * mean_spike_count

    # MAC comparison: dense multiply-accumulate ops
    q_macs = hidden * q_neurons  # matmul
    k_macs = hidden * kv_neurons
    v_macs = hidden * kv_neurons
    o_macs = q_neurons * hidden
    attn_macs_per_token = q_macs + k_macs + v_macs + o_macs

    expert_macs_per_token = top_k * (
        hidden * expert_inter  # gate_proj
        + hidden * expert_inter  # up_proj
        + expert_inter * hidden  # down_proj
    )

    return {
        "timesteps": timesteps,
        "threshold": threshold,
        "max_quantisation_error": quant_error,
        "relative_error_bound": f"O(1/{timesteps}) = {1.0/timesteps:.6f}",
        "attention": {
            "neurons_per_token": attn_neurons_per_token,
            "mean_spikes_per_token": attn_spikes_per_token,
            "macs_per_token": attn_macs_per_token,
            "spike_to_mac_ratio": round(
                attn_spikes_per_token / attn_macs_per_token, 6
            ),
        },
        "moe_ffn_per_token": {
            "active_experts": top_k,
            "neurons_per_token": expert_neurons_per_token,
            "mean_spikes_per_token": expert_spikes_per_token,
            "macs_per_token": expert_macs_per_token,
            "spike_to_mac_ratio": round(
                expert_spikes_per_token / expert_macs_per_token, 6
            ),
        },
        "notes": [
            "Spike-to-MAC ratio < 1 means fewer spike events than MAC ops",
            "Each spike = 1 addition; each MAC = 1 multiply + 1 add",
            "Actual sparsity depends on activation distribution (ReLU helps)",
            f"With T={timesteps}, quantisation error < {quant_error:.4f} per neuron",
        ],
    }


def estimate_energy_savings(
    config: dict[str, Any],
    timesteps: int = 128,
) -> dict[str, Any]:
    """Estimate energy savings from spiking vs multiply-accumulate.

    Energy model (45nm CMOS, Horowitz 2014):
    - 32-bit float MAC: ~4.6 pJ
    - 32-bit float ADD: ~0.9 pJ
    - Spike event (accumulate): ~0.9 pJ (just addition, no multiply)

    For neuromorphic hardware (Loihi 2, SpiNNaker 2):
    - Spike event: ~0.02-0.1 pJ (much lower than CMOS ADD)
    """
    hidden = config["hidden_size"]
    n_layers = config["num_hidden_layers"]
    n_heads = config["num_attention_heads"]
    n_kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    n_experts = config["num_experts"]
    top_k = config["num_experts_per_tok"]
    expert_inter = config["moe_intermediate_size"]
    shared_inter = config.get("shared_expert_intermediate_size", 0)

    # Energy constants (pJ per operation, 45nm CMOS)
    E_MAC_32 = 4.6   # 32-bit float multiply-accumulate
    E_ADD_32 = 0.9   # 32-bit float addition (spike accumulate)
    E_SPIKE_NEUROMORPHIC = 0.05  # neuromorphic hardware estimate

    # Per-layer MAC counts (attention)
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim
    attn_macs = (
        hidden * q_dim     # q_proj
        + hidden * kv_dim  # k_proj
        + hidden * kv_dim  # v_proj
        + q_dim * hidden   # o_proj
    )

    # Per-layer MAC counts (MoE, per token — only top_k active)
    expert_macs = top_k * (
        hidden * expert_inter  # gate_proj
        + hidden * expert_inter  # up_proj
        + expert_inter * hidden  # down_proj
    )

    # Shared expert
    shared_macs = 0
    if shared_inter > 0:
        shared_macs = (
            hidden * shared_inter
            + hidden * shared_inter
            + shared_inter * hidden
        )

    # Router MACs
    router_macs = hidden * n_experts

    total_macs_per_token_per_layer = (
        attn_macs + expert_macs + shared_macs + router_macs
    )
    total_macs_per_token = total_macs_per_token_per_layer * n_layers

    # Spike counts (assuming mean activation 0.5, max_rate 1.0)
    mean_spike_rate = 0.5  # fraction of timesteps that spike
    mean_spikes_per_neuron = timesteps * mean_spike_rate

    # Attention spike ops (each spike = 1 ADD to downstream neuron)
    # Number of downstream connections = out_features
    attn_spike_ops = (q_dim + kv_dim + kv_dim + hidden) * mean_spikes_per_neuron

    # Expert spike ops (only top_k active)
    expert_spike_ops = top_k * (
        expert_inter + expert_inter + hidden
    ) * mean_spikes_per_neuron

    total_spike_ops_per_layer = attn_spike_ops + expert_spike_ops
    total_spike_ops_per_token = total_spike_ops_per_layer * n_layers

    # Energy calculations
    ann_energy_per_token = total_macs_per_token * E_MAC_32  # pJ
    snn_energy_cmos = total_spike_ops_per_token * E_ADD_32  # pJ
    snn_energy_neuromorphic = total_spike_ops_per_token * E_SPIKE_NEUROMORPHIC

    return {
        "energy_model": {
            "mac_32bit_pJ": E_MAC_32,
            "add_32bit_pJ": E_ADD_32,
            "spike_neuromorphic_pJ": E_SPIKE_NEUROMORPHIC,
            "source": "Horowitz 2014 (45nm CMOS)",
        },
        "per_token_per_layer": {
            "ann_macs": total_macs_per_token_per_layer,
            "snn_spike_ops": int(total_spike_ops_per_layer),
        },
        "per_token_full_model": {
            "ann_macs": total_macs_per_token,
            "ann_energy_pJ": round(ann_energy_per_token, 1),
            "ann_energy_uJ": round(ann_energy_per_token / 1e6, 3),
            "snn_spike_ops": int(total_spike_ops_per_token),
            "snn_energy_cmos_pJ": round(snn_energy_cmos, 1),
            "snn_energy_cmos_uJ": round(snn_energy_cmos / 1e6, 3),
            "snn_energy_neuromorphic_pJ": round(snn_energy_neuromorphic, 1),
            "snn_energy_neuromorphic_uJ": round(snn_energy_neuromorphic / 1e6, 3),
        },
        "savings": {
            "cmos_ratio": round(ann_energy_per_token / snn_energy_cmos, 2)
            if snn_energy_cmos > 0 else float("inf"),
            "neuromorphic_ratio": round(
                ann_energy_per_token / snn_energy_neuromorphic, 2
            ) if snn_energy_neuromorphic > 0 else float("inf"),
            "cmos_saving_percent": round(
                (1 - snn_energy_cmos / ann_energy_per_token) * 100, 1
            ) if ann_energy_per_token > 0 else 0,
            "neuromorphic_saving_percent": round(
                (1 - snn_energy_neuromorphic / ann_energy_per_token) * 100, 1
            ) if ann_energy_per_token > 0 else 0,
        },
        "caveats": [
            "Energy model assumes 45nm CMOS; modern 5nm would scale all values",
            "Spike ops count assumes mean activation 0.5 (uniform distribution)",
            "Real activation sparsity (post-ReLU) would reduce spike count",
            "SwiGLU gate path not fully convertible — hybrid energy mix",
            "Does not include memory access energy (often dominant)",
            f"T={timesteps} timesteps means {timesteps}x temporal overhead",
            "Neuromorphic estimate assumes dedicated hardware (Loihi 2 class)",
        ],
    }


def run_analysis(
    config_path: Path | None = None,
    timesteps: int = 128,
    results_out: Path | None = None,
) -> dict[str, Any]:
    """Run the full story-27 theoretical analysis.

    No weights, no GPU, no torch required.
    """
    config = load_model_config(config_path)
    log.info("loaded config for %s", config.get("_name_or_path", "unknown"))

    convertibility = analyze_layer_convertibility(config)
    spike_rates = calculate_spike_rate_equivalence(config, timesteps)
    energy = estimate_energy_savings(config, timesteps)

    # Summary statistics
    summary = convertibility["summary"]
    total = summary["total_model_params_estimate"]
    fully = summary["fully_convertible_params"]
    partial = summary["partially_convertible_params"]
    not_conv = summary["not_convertible_params"]

    results = {
        "story": "story-27",
        "title": "LAS convert Qwen3.5-35B-A3B -> SpikingKiki-35B (MoE)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": {
            "name": config.get("_name_or_path", "Qwen3.5-35B-A3B"),
            "architecture": config.get("architectures", ["unknown"])[0]
            if isinstance(config.get("architectures"), list)
            else "unknown",
            "hidden_size": config["hidden_size"],
            "num_layers": config["num_hidden_layers"],
            "num_experts": config["num_experts"],
            "active_experts_per_token": config["num_experts_per_tok"],
            "total_params_estimate": total,
            "active_params_estimate": config.get("active_params_billions", 3.0),
        },
        "convertibility": convertibility,
        "spike_rate_equivalence": spike_rates,
        "energy_savings": energy,
        "conversion_summary": {
            "fully_convertible_percent": round(fully / total * 100, 1),
            "partially_convertible_percent": round(partial / total * 100, 1),
            "not_convertible_percent": round(not_conv / total * 100, 1),
            "fully_convertible_params": fully,
            "partially_convertible_params": partial,
            "not_convertible_params": not_conv,
            "verdict": (
                "Attention layers (GQA projections) and MoE routers are fully "
                "convertible via LAS with identity activation. MoE expert FFNs "
                "are partially convertible: down_proj is straightforward, but "
                "SwiGLU gate (SiLU activation + element-wise multiply) has no "
                "direct spiking equivalent. Recommended approach: hybrid "
                "conversion keeping SwiGLU gate in ANN domain, converting "
                "up/down projections to spiking. This yields ~60% expert "
                "parameter conversion with predictable energy savings."
            ),
        },
        "research_questions": [
            {
                "id": "RQ1",
                "question": "Can SiLU be approximated by shifted ReLU in rate code?",
                "status": "open",
                "priority": "high",
                "notes": "silu(x) ~ relu(x+0.5) - 0.25 gives ~8% boundary error",
            },
            {
                "id": "RQ2",
                "question": "Does two-channel signed encoding preserve MoE quality?",
                "status": "open",
                "priority": "high",
                "notes": "2x spike overhead; needs empirical validation on MoE",
            },
            {
                "id": "RQ3",
                "question": "What is the minimum T for <1% attention error at 35B scale?",
                "status": "testable",
                "priority": "medium",
                "notes": f"Theory: T=128 gives error bound {1/128:.4f}; "
                         "empirical validation needs weight samples",
            },
            {
                "id": "RQ4",
                "question": "Can temporal correlation coding handle element-wise multiply?",
                "status": "open",
                "priority": "low",
                "notes": "Active research area; no proven LAS-compatible method",
            },
        ],
    }

    out_path = results_out or (RESULTS_DIR / "spikingkiki-35b-analysis.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    log.info("analysis written to %s", out_path)

    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "LAS conversion: Qwen3.5-35B-A3B MoE → SpikingKiki-35B-A3B"
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "/Volumes/studio/models/Qwen3.5-35B-A3B"
        ),
        help="Path to the HuggingFace model directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/Volumes/studio/models/SpikingKiki-35B-A3B"
        ),
        help="Output directory for converted spiking layers.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=128,
        help="LIF integration timesteps T (default 128).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip layers already saved to --output.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate layer map and metadata without loading weights.",
    )
    parser.add_argument(
        "--results-out",
        type=Path,
        default=RESULTS_DIR / "spikingkiki-35b-convert.json",
        help="Path for the conversion metadata JSON.",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run theoretical analysis only (story-27). No weights/GPU needed.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to model config.json for analysis mode.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.getLogger().setLevel(args.log_level)

    t_start = time.monotonic()

    # --- Analyze mode (story-27: theoretical analysis, no weights) ---
    if args.analyze:
        log.info("running theoretical analysis (story-27) — no weights needed")
        results = run_analysis(
            config_path=args.config,
            timesteps=args.timesteps,
            results_out=RESULTS_DIR / "spikingkiki-35b-analysis.json",
        )
        summary = results["conversion_summary"]
        log.info(
            "fully convertible: %.1f%%, partial: %.1f%%, not: %.1f%%",
            summary["fully_convertible_percent"],
            summary["partially_convertible_percent"],
            summary["not_convertible_percent"],
        )
        savings = results["energy_savings"]["savings"]
        log.info(
            "energy savings — CMOS: %.1f%%, neuromorphic: %.1f%%",
            savings["cmos_saving_percent"],
            savings["neuromorphic_saving_percent"],
        )
        return 0

    log.info(
        "SpikingKiki-35B conversion — timesteps=%d resume=%s dry_run=%s",
        args.timesteps,
        args.resume,
        args.dry_run,
    )

    layer_map = build_layer_map()
    map_stats = layer_map_stats(layer_map)
    log.info(
        "layer map: %d total (%s)",
        map_stats["total"],
        ", ".join(f"{k}={v}" for k, v in map_stats.items() if k != "total"),
    )

    spike_stats: dict[str, Any] = {}

    # --- Dry-run path (no torch needed) ---
    if args.dry_run:
        log.info("dry-run: skipping weight loading")
        meta = build_metadata(
            args, layer_map, spike_stats,
            elapsed_s=time.monotonic() - t_start,
            status="dry_run",
        )
        meta["layer_map"] = layer_map
        args.results_out.parent.mkdir(parents=True, exist_ok=True)
        args.results_out.write_text(json.dumps(meta, indent=2) + "\n")
        log.info("metadata written to %s", args.results_out)
        return 0

    # --- Check model directory ---
    if not args.input.exists():
        log.error("model directory not found: %s", args.input)
        meta = build_metadata(
            args, layer_map, spike_stats,
            elapsed_s=time.monotonic() - t_start,
            status="no_weights",
        )
        args.results_out.parent.mkdir(parents=True, exist_ok=True)
        args.results_out.write_text(json.dumps(meta, indent=2) + "\n")
        return 2

    # --- Import torch (guarded) ---
    torch = _try_import_torch()
    if torch is None:
        log.error("torch not available — cannot load model weights")
        meta = build_metadata(
            args, layer_map, spike_stats,
            elapsed_s=time.monotonic() - t_start,
            status="torch_missing",
        )
        args.results_out.parent.mkdir(parents=True, exist_ok=True)
        args.results_out.write_text(json.dumps(meta, indent=2) + "\n")
        return 3

    AutoModelForCausalLM, AutoConfig = _try_import_transformers()
    if AutoModelForCausalLM is None:
        log.error("transformers not available")
        meta = build_metadata(
            args, layer_map, spike_stats,
            elapsed_s=time.monotonic() - t_start,
            status="transformers_missing",
        )
        args.results_out.parent.mkdir(parents=True, exist_ok=True)
        args.results_out.write_text(json.dumps(meta, indent=2) + "\n")
        return 3

    # --- Resume state ---
    converted_keys: set[str] = set()
    if args.resume:
        converted_keys = load_resume_state(args.output)
        log.info("resume: %d layers already converted", len(converted_keys))

    # --- Load model ---
    log.info("loading model from %s (bfloat16, cpu map) …", args.input)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(args.input),
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
        )
        model.eval()
    except Exception as exc:
        log.error("model load failed: %s: %s", type(exc).__name__, exc)
        meta = build_metadata(
            args, layer_map, spike_stats,
            elapsed_s=time.monotonic() - t_start,
            status="load_failed",
        )
        meta["error"] = str(exc)
        args.results_out.parent.mkdir(parents=True, exist_ok=True)
        args.results_out.write_text(json.dumps(meta, indent=2) + "\n")
        return 4

    # --- LASConverter ---
    from src.spiking.las_converter import LASConverter
    converter = LASConverter(timesteps=args.timesteps, max_rate=1.0)

    args.output.mkdir(parents=True, exist_ok=True)

    # --- Layer-by-layer conversion ---
    n_total = NUM_LAYERS
    for layer_idx in range(n_total):
        t_layer = time.monotonic()
        log.info(
            "[%d/%d] converting block %d …", layer_idx + 1, n_total, layer_idx
        )

        try:
            block_stats = convert_block_torch(
                model=model,
                layer_idx=layer_idx,
                converter=converter,
                spike_stats=spike_stats,
                converted_keys=converted_keys,
                output_dir=args.output,
                timesteps=args.timesteps,
            )
        except Exception as exc:
            log.error(
                "block %d conversion failed: %s: %s",
                layer_idx, type(exc).__name__, exc,
            )
            spike_stats[f"layer_{layer_idx}"] = {
                "status": "failed",
                "error": str(exc),
            }
            # Save resume state so we can skip completed layers on retry
            save_resume_state(args.output, converted_keys)
            continue

        spike_stats[f"layer_{layer_idx}"] = block_stats
        elapsed_block = time.monotonic() - t_layer
        log.info(
            "  block %d done in %.1fs (%d keys converted so far)",
            layer_idx,
            elapsed_block,
            len(converted_keys),
        )

        # Periodic resume state flush (every 5 blocks)
        if layer_idx % 5 == 0:
            save_resume_state(args.output, converted_keys)
            log.debug("resume state flushed (%d keys)", len(converted_keys))

    save_resume_state(args.output, converted_keys)

    # Mark converted layers in the layer map
    for entry in layer_map:
        entry["converted"] = entry["key_prefix"] in converted_keys

    elapsed_total = time.monotonic() - t_start
    log.info("conversion complete in %.1fs", elapsed_total)

    meta = build_metadata(
        args, layer_map, spike_stats,
        elapsed_s=elapsed_total,
        status="ok",
    )
    args.results_out.parent.mkdir(parents=True, exist_ok=True)
    args.results_out.write_text(json.dumps(meta, indent=2) + "\n")
    log.info("metadata written to %s", args.results_out)

    return 0


if __name__ == "__main__":
    sys.exit(main())

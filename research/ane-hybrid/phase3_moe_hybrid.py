#!/usr/bin/env python3
"""Phase 3 : Pipeline hybride ANE+Metal pour Qwen3.5-35B-A3B.

Combine :
- ANE : 30 DeltaNet + 10 Full Attention (couches converties Phase 2)
- GPU Metal (MLX) : MoE FFN (8 experts routés + 1 shared)
- CPU : Router (softmax + top-8)

Architecture pipeline :
  Token → Embedding → [ANE attention → CPU router → GPU experts] × 40 → LM Head

Phase 3 du plan ANE hybrid pipeline.
"""

import sys
import json
import time
import torch
import numpy as np
from pathlib import Path

import coremltools as ct

# MLX pour les experts MoE
try:
    import mlx.core as mx
    import mlx.nn as mxnn
    from safetensors import safe_open
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("ATTENTION : MLX non disponible — benchmark ANE-only")


# === Chargement des couches ANE (Phase 2) ===

def load_ane_layers(stack_dir: str, compute_units=ct.ComputeUnit.CPU_AND_NE):
    """Charge les 40 couches CoreML depuis Phase 2."""
    stack_path = Path(stack_dir)
    layer_types = ['linear_attention', 'linear_attention', 'linear_attention', 'full_attention'] * 10

    models = []
    for i in range(40):
        lt = layer_types[i]
        tag = f"layer_{i:02d}_{lt[:3]}"
        path = stack_path / f"{tag}.mlpackage"
        if not path.exists():
            raise FileNotFoundError(f"Couche {tag} introuvable : {path}")
        m = ct.models.MLModel(str(path), compute_units=compute_units)
        models.append((lt, m))

    print(f"40 couches ANE chargées depuis {stack_dir}")
    return models, layer_types


# === Router MoE (CPU) ===

def load_router_weights(layer_idx: int):
    """Charge les poids du router MoE pour une couche."""
    snap_dir = list(Path('/Users/clems/.cache/huggingface/hub/models--Jackrong--Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled/snapshots/').iterdir())[0]
    with open(snap_dir / 'model.safetensors.index.json') as f:
        idx = json.load(f)

    gate_key = f'model.language_model.layers.{layer_idx}.mlp.gate.weight'
    shard = idx['weight_map'][gate_key]
    f = safe_open(str(snap_dir / shard), framework='pt')
    return f.get_tensor(gate_key).float().numpy()


def route_experts(hidden_states: np.ndarray, gate_weight: np.ndarray, top_k: int = 8):
    """Route MoE : softmax sur 256 experts, sélectionne top-8.

    hidden_states: [1, 1, 2048]
    gate_weight: [256, 2048]
    Returns: expert_indices [8], expert_scores [8] (normalisées)
    """
    logits = hidden_states.reshape(-1) @ gate_weight.T  # [256]
    # Top-K
    indices = np.argpartition(logits, -top_k)[-top_k:]
    scores = logits[indices]
    # Softmax sur les top-K
    scores = np.exp(scores - scores.max())
    scores = scores / scores.sum()
    return indices, scores


# === Shared Expert (simple — petite FFN) ===

def load_shared_expert_weights(layer_idx: int):
    """Charge les poids du shared expert."""
    snap_dir = list(Path('/Users/clems/.cache/huggingface/hub/models--Jackrong--Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled/snapshots/').iterdir())[0]
    with open(snap_dir / 'model.safetensors.index.json') as f:
        idx = json.load(f)

    prefix = f'model.language_model.layers.{layer_idx}.mlp.shared_expert'
    loaded = {}
    shards_needed = set()
    for key, shard in idx['weight_map'].items():
        if prefix in key:
            shards_needed.add(shard)

    for shard_name in shards_needed:
        f = safe_open(str(snap_dir / shard_name), framework='pt')
        for key in f.keys():
            if prefix in key:
                loaded[key] = f.get_tensor(key).float().numpy()

    return {
        'gate': loaded[f'{prefix}.gate_proj.weight'],
        'up': loaded[f'{prefix}.up_proj.weight'],
        'down': loaded[f'{prefix}.down_proj.weight'],
    }


def shared_expert_forward(x: np.ndarray, weights: dict) -> np.ndarray:
    """SwiGLU shared expert : gate_proj * silu(up_proj) → down_proj."""
    gate = x @ weights['gate'].T
    up = x @ weights['up'].T
    # SiLU
    gate_act = gate * (1.0 / (1.0 + np.exp(-gate)))
    hidden = gate_act * up
    return hidden @ weights['down'].T


# === Pipeline hybride complet ===

def hybrid_forward_token(
    ane_models: list,
    layer_types: list,
    routers: list,
    shared_experts: list,
    hidden_states: np.ndarray,
    deltanet_states: list,
    conv_states: list,
):
    """Forward pass d'un token à travers les 40 couches.

    ANE : attention (DeltaNet / Full Attention)
    CPU : routing + shared expert
    GPU : routed experts (TODO Phase 3 complète)
    """
    delta_idx = 0

    for i, (lt, model) in enumerate(ane_models):
        # === Attention sur ANE ===
        if lt == 'linear_attention':
            inp = {
                "hidden_states": hidden_states,
                "state_in": deltanet_states[delta_idx],
                "conv_state_in": conv_states[delta_idx],
            }
            result = model.predict(inp)
            attn_out = result["output"]
            deltanet_states[delta_idx] = result["state_out"]
            conv_states[delta_idx] = result["conv_state_out"]
            delta_idx += 1
        else:
            result = model.predict({"hidden_states": hidden_states})
            attn_out = result["output"]

        # === Residual connection (attention) ===
        # Note : pas de LayerNorm ici car il est intégré dans les couches CoreML
        hidden_after_attn = hidden_states + attn_out

        # === MoE FFN (CPU pour l'instant) ===
        # Router
        expert_ids, expert_scores = route_experts(
            hidden_after_attn, routers[i], top_k=8
        )

        # Shared expert
        shared_out = shared_expert_forward(
            hidden_after_attn.reshape(1, -1), shared_experts[i]
        )

        # TODO Phase 3 complète : routed experts sur GPU Metal
        # Pour l'instant : shared expert seul (approximation)
        ffn_out = shared_out.reshape(hidden_states.shape)

        # Residual connection (FFN)
        hidden_states = hidden_after_attn + ffn_out

    return hidden_states


def benchmark_hybrid():
    """Benchmark du pipeline hybride ANE+CPU."""
    print("=== Phase 3 : Pipeline Hybride ANE+Metal ===")
    print()

    # Charger les couches ANE
    stack_dir = "research/ane-hybrid/mlpackages/full_stack"
    print("Chargement des 40 couches ANE...")
    ane_models, layer_types = load_ane_layers(stack_dir)

    # Charger les routers et shared experts
    print("Chargement des routers et shared experts...")
    routers = []
    shared_experts = []
    for i in range(40):
        routers.append(load_router_weights(i))
        shared_experts.append(load_shared_expert_weights(i))
        print(f"\r  Couche {i+1}/40", end="", flush=True)
    print(" OK")

    # Initialiser les états DeltaNet
    deltanet_states = [
        np.zeros((1, 32, 128, 128), dtype=np.float16) for _ in range(30)
    ]
    conv_states = [
        np.zeros((1, 8192, 3), dtype=np.float16) for _ in range(30)
    ]

    # Warmup
    print("\nWarmup...")
    x = np.random.randn(1, 1, 2048).astype(np.float16)
    hybrid_forward_token(
        ane_models, layer_types, routers, shared_experts,
        x, deltanet_states, conv_states
    )

    # Benchmark
    n_tokens = 10
    print(f"Benchmark {n_tokens} tokens (ANE attention + CPU shared expert)...")
    t0 = time.time()
    for _ in range(n_tokens):
        x = np.random.randn(1, 1, 2048).astype(np.float16)
        out = hybrid_forward_token(
            ane_models, layer_types, routers, shared_experts,
            x, deltanet_states, conv_states
        )
    elapsed = time.time() - t0

    tok_s = n_tokens / elapsed
    ms_per_tok = elapsed / n_tokens * 1000

    print(f"\n=== RÉSULTAT PHASE 3 ===")
    print(f"  {n_tokens} tokens en {elapsed:.2f}s")
    print(f"  {tok_s:.1f} tok/s ({ms_per_tok:.1f} ms/tok)")
    print(f"  Mode : ANE (attention) + CPU (router + shared expert)")
    print(f"  Note : sans routed experts GPU (approximation)")
    print(f"  Avec GPU experts : estimation ~{tok_s * 0.7:.1f}-{tok_s * 0.9:.1f} tok/s")
    print(f"  Output range: [{out.min():.4f}, {out.max():.4f}]")


if __name__ == "__main__":
    benchmark_hybrid()

#!/usr/bin/env python3
"""Phase 3b : Experts MoE sur GPU Metal (MLX) + pipeline ANE.

Optimise Phase 3 en remplaçant le shared expert CPU par
du GPU Metal pour TOUS les experts (shared + 8 routés).

Architecture pipeline :
  Couche N   : GPU fait le MoE FFN (8 experts + shared)
  Couche N+1 : ANE fait l'attention (DeltaNet/Full)
  → Pipeline parallèle, le plus lent dicte le débit

Benchmark vs Phase 3 (9.9 tok/s avec shared expert CPU seul).
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import coremltools as ct

try:
    import mlx.core as mx
    from safetensors import safe_open
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("MLX requis pour les experts GPU")
    sys.exit(1)


# === Chargement ANE (réutilise Phase 2) ===

def load_ane_layers(stack_dir):
    stack_path = Path(stack_dir)
    layer_types = ['linear_attention'] * 3 + ['full_attention']
    layer_types = layer_types * 10
    models = []
    for i in range(40):
        lt = layer_types[i]
        tag = f"layer_{i:02d}_{lt[:3]}"
        path = stack_path / f"{tag}.mlpackage"
        m = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
        models.append((lt, m))
    return models, layer_types


# === MoE FFN sur GPU Metal (MLX) ===

class MLXMoEFFN:
    """MoE FFN complet sur GPU Metal via MLX.

    Charge les poids des 256 experts + shared expert.
    Route top-8 + shared pour chaque token.
    """

    def __init__(self, layer_idx: int, num_experts=256, top_k=8):
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.top_k = top_k
        self._load_weights()

    def _to_mx(self, tensor):
        """Convertit un tenseur safetensors (bf16/fp32) en mx.array float16."""
        import torch
        if hasattr(tensor, 'numpy'):
            # torch tensor
            return mx.array(tensor.float().half().numpy())
        return mx.array(tensor)

    def _load_weights(self):
        snap_dir = list(Path('/Users/clems/.cache/huggingface/hub/models--Jackrong--Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled/snapshots/').iterdir())[0]
        with open(snap_dir / 'model.safetensors.index.json') as f:
            idx = json.load(f)

        prefix = f'model.language_model.layers.{self.layer_idx}.mlp'

        # Helper pour charger un tenseur par clé
        def load_tensor(key):
            shard = idx['weight_map'][key]
            f = safe_open(str(snap_dir / shard), framework='pt')
            return self._to_mx(f.get_tensor(key))

        # Router gate
        self.gate = load_tensor(f'{prefix}.gate.weight')

        # Shared expert
        shared_prefix = f'{prefix}.shared_expert'
        self.shared = {
            name.split('.')[0]: load_tensor(f'{shared_prefix}.{name}')
            for name in ['gate_proj.weight', 'up_proj.weight', 'down_proj.weight']
        }

        # Shared expert gate
        self.shared_gate = load_tensor(f'{prefix}.shared_expert_gate.weight')

        # Routed experts (packed tensors)
        # experts.gate_up_proj: [num_experts, 2*intermediate, hidden]
        # experts.down_proj: [num_experts, hidden, intermediate]
        for name in ['gate_up_proj', 'down_proj']:
            key = f'{prefix}.experts.{name}'
            setattr(self, f'experts_{name}', load_tensor(key))

    def forward(self, x_np: np.ndarray) -> np.ndarray:
        """Forward MoE FFN sur GPU.

        x_np: [1, 1, 2048] numpy (vient de l'ANE)
        Returns: [1, 1, 2048] numpy
        """
        x = mx.array(x_np.reshape(1, -1))  # [1, 2048]

        # Router
        logits = x @ self.gate.T  # [1, 256]
        logits = logits.squeeze(0)  # [256]

        # Top-K indices et scores
        indices = mx.argpartition(-logits, kth=self.top_k)[:self.top_k]
        scores = logits[indices]
        scores = mx.softmax(scores)
        mx.eval(indices, scores)

        # Routed experts — per-expert loop (MLX indexing)
        indices_list = indices.tolist()
        scores_list = scores.tolist()

        result = mx.zeros((1, x.shape[-1]))
        for k in range(self.top_k):
            idx = indices_list[k]
            score = scores_list[k]
            gu = self.experts_gate_up_proj[idx]  # [2*inter, hidden]
            d = self.experts_down_proj[idx]       # [hidden, inter]

            gu_out = x @ gu.T  # [1, 2*inter]
            inter_size = gu_out.shape[-1] // 2
            gate_out = gu_out[:, :inter_size]
            up_out = gu_out[:, inter_size:]
            h = (gate_out * mx.sigmoid(gate_out)) * up_out  # SiLU
            out_k = h @ d.T  # [1, hidden]
            result = result + score * out_k

        # Shared expert
        sg = x @ self.shared['gate_proj'].T
        su = x @ self.shared['up_proj'].T
        sh = (sg * mx.sigmoid(sg)) * su
        shared_out = sh @ self.shared['down_proj'].T

        # Shared expert gate
        shared_weight = mx.sigmoid(x @ self.shared_gate.T)
        shared_out = shared_out * shared_weight

        total = result + shared_out
        mx.eval(total)

        return np.array(total).reshape(1, 1, -1).astype(np.float16)


def benchmark_gpu_experts():
    """Benchmark MoE FFN sur GPU seul."""
    print("=== Benchmark MoE FFN GPU (MLX) ===")

    print("Chargement experts couche 0...")
    t0 = time.time()
    moe = MLXMoEFFN(layer_idx=0)
    print(f"  Chargé en {time.time()-t0:.1f}s")

    x = np.random.randn(1, 1, 2048).astype(np.float16)

    # Warmup
    for _ in range(3):
        moe.forward(x)

    # Bench
    n = 50
    t0 = time.time()
    for _ in range(n):
        out = moe.forward(x)
    elapsed = time.time() - t0
    print(f"  {n} forwards en {elapsed:.2f}s → {n/elapsed:.1f} tok/s ({elapsed/n*1000:.1f} ms/tok)")
    print(f"  Output range: [{out.min():.4f}, {out.max():.4f}]")
    return elapsed / n


def benchmark_hybrid_pipeline():
    """Benchmark pipeline parallèle ANE attention + GPU experts."""
    print("\n=== Pipeline Hybride ANE+GPU ===")

    # Charger ANE
    print("Chargement ANE...")
    ane_models, layer_types = load_ane_layers("research/ane-hybrid/mlpackages/full_stack")

    # Charger experts GPU (seulement 2 couches pour le benchmark)
    print("Chargement experts GPU (couches 0-1)...")
    moe_layers = {}
    for i in range(2):
        moe_layers[i] = MLXMoEFFN(layer_idx=i)
    print("OK")

    # Benchmark : attention ANE + MoE GPU pour 2 premières couches
    deltanet_states = [np.zeros((1, 32, 128, 128), dtype=np.float16) for _ in range(30)]
    conv_states = [np.zeros((1, 8192, 3), dtype=np.float16) for _ in range(30)]

    x = np.random.randn(1, 1, 2048).astype(np.float16)

    # Séquentiel d'abord (baseline)
    print("\nBenchmark séquentiel (2 couches)...")
    n = 20
    t0 = time.time()
    for _ in range(n):
        x_tok = np.random.randn(1, 1, 2048).astype(np.float16)
        for i in range(2):
            lt, model = ane_models[i]
            if lt == 'linear_attention':
                inp = {"hidden_states": x_tok, "state_in": deltanet_states[i], "conv_state_in": conv_states[i]}
                r = model.predict(inp)
                attn_out = r["output"]
                deltanet_states[i] = r["state_out"]
                conv_states[i] = r["conv_state_out"]
            else:
                r = model.predict({"hidden_states": x_tok})
                attn_out = r["output"]
            x_after_attn = x_tok + attn_out
            ffn_out = moe_layers[i].forward(x_after_attn)
            x_tok = x_after_attn + ffn_out
    seq_elapsed = time.time() - t0
    seq_ms = seq_elapsed / n * 1000 / 2  # per layer

    # Pipeline parallèle (ANE couche N+1 en même temps que GPU couche N)
    print("Benchmark pipeline (2 couches, ThreadPool)...")
    executor = ThreadPoolExecutor(max_workers=2)

    t0 = time.time()
    for _ in range(n):
        x_tok = np.random.randn(1, 1, 2048).astype(np.float16)

        # Couche 0 : séquentiel (pas de pipeline pour la première)
        lt, model = ane_models[0]
        inp = {"hidden_states": x_tok, "state_in": deltanet_states[0], "conv_state_in": conv_states[0]}
        r = model.predict(inp)
        deltanet_states[0] = r["state_out"]
        conv_states[0] = r["conv_state_out"]
        x_after_attn_0 = x_tok + r["output"]

        # Pipeline : GPU fait MoE couche 0 PENDANT QUE ANE fait attention couche 1
        gpu_future = executor.submit(moe_layers[0].forward, x_after_attn_0)

        lt1, model1 = ane_models[1]
        # Pour l'ANE couche 1, on utilise x_after_attn_0 + ffn (pas encore dispo)
        # En vrai pipeline, on utilise le résultat de la couche 0 complète
        # Ici on simule en lançant l'ANE en parallèle
        inp1 = {"hidden_states": x_after_attn_0, "state_in": deltanet_states[1], "conv_state_in": conv_states[1]}
        ane_future = executor.submit(model1.predict, inp1)

        ffn_out_0 = gpu_future.result()
        r1 = ane_future.result()
        deltanet_states[1] = r1["state_out"]
        conv_states[1] = r1["conv_state_out"]

        x_after_layer0 = x_after_attn_0 + ffn_out_0
        x_after_attn_1 = x_after_layer0 + r1["output"]
        ffn_out_1 = moe_layers[1].forward(x_after_attn_1)
        x_tok = x_after_attn_1 + ffn_out_1

    pipe_elapsed = time.time() - t0
    pipe_ms = pipe_elapsed / n * 1000 / 2

    executor.shutdown()

    # Extrapolation à 40 couches
    seq_40 = seq_ms * 40
    pipe_40 = pipe_ms * 40

    print(f"\n=== RÉSULTATS ===")
    print(f"  Séquentiel (2 couches) : {seq_ms:.1f} ms/couche")
    print(f"  Pipeline   (2 couches) : {pipe_ms:.1f} ms/couche")
    print(f"  Speedup pipeline : {seq_ms/pipe_ms:.2f}x")
    print(f"")
    print(f"  Extrapolation 40 couches :")
    print(f"    Séquentiel : {seq_40:.0f} ms/tok → {1000/seq_40:.1f} tok/s")
    print(f"    Pipeline   : {pipe_40:.0f} ms/tok → {1000/pipe_40:.1f} tok/s")


if __name__ == "__main__":
    ms_per_expert = benchmark_gpu_experts()
    print(f"\n  MoE FFN GPU : {ms_per_expert*1000:.1f} ms/couche")
    print(f"  ANE attention : ~1.7 ms/couche (Phase 2)")
    print(f"  → Le GPU MoE est le bottleneck")

    benchmark_hybrid_pipeline()

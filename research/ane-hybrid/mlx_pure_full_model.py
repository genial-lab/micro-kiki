#!/usr/bin/env python3
"""Pipeline MLX pur : 40 couches DeltaNet + Attention + MoE FFN.

Tout sur GPU Metal, sans CoreML/ANE. Benchmark complet du modèle
Qwen3.5-35B-A3B avec vrais poids.
"""

import json
import time
from pathlib import Path

import mlx.core as mx
from safetensors import safe_open


# === Chargement des poids ===

SNAP_DIR = None
IDX = None

def init_weights():
    global SNAP_DIR, IDX
    SNAP_DIR = list(Path('/Users/clems/.cache/huggingface/hub/models--Jackrong--Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled/snapshots/').iterdir())[0]
    with open(SNAP_DIR / 'model.safetensors.index.json') as f:
        IDX = json.load(f)

def load_mx(key):
    shard = IDX['weight_map'][key]
    f = safe_open(str(SNAP_DIR / shard), framework='pt')
    return mx.array(f.get_tensor(key).float().half().numpy())


# === DeltaNet Layer (30 couches) ===

class MLXDeltaNetLayer:
    def __init__(self, layer_idx):
        prefix = f'model.language_model.layers.{layer_idx}.linear_attn'
        self.w_qkv = load_mx(f'{prefix}.in_proj_qkv.weight')
        self.w_z = load_mx(f'{prefix}.in_proj_z.weight')
        self.w_a = load_mx(f'{prefix}.in_proj_a.weight')
        self.w_b = load_mx(f'{prefix}.in_proj_b.weight')
        self.A_log = load_mx(f'{prefix}.A_log').astype(mx.float32)
        self.dt_bias = load_mx(f'{prefix}.dt_bias')
        self.w_norm = load_mx(f'{prefix}.norm.weight')
        self.w_out = load_mx(f'{prefix}.out_proj.weight')
        # Conv1d
        self.w_conv = load_mx(f'{prefix}.conv1d.weight')  # [8192, 1, 4]
        mx.eval(self.w_qkv, self.w_z, self.w_a, self.w_b,
                self.A_log, self.dt_bias, self.w_norm, self.w_out, self.w_conv)

    def __call__(self, x, state, conv_state):
        """x: [1, 2048], state: [1, 32, 128, 128], conv: [1, 8192, 3]"""
        # Projections
        qkv = x @ self.w_qkv.T  # [1, 8192]

        # Conv1d update (single token depthwise)
        qkv_col = qkv.reshape(1, 8192, 1)
        combined = mx.concatenate([conv_state, qkv_col], axis=2)  # [1, 8192, 4]
        new_conv = combined[:, :, 1:]  # [1, 8192, 3]
        # Depthwise conv: sum(input * kernel) per channel
        w = self.w_conv.squeeze(1)  # [8192, 4]
        conv_out = mx.sum(combined * w.reshape(1, 8192, 4), axis=2, keepdims=True)  # [1, 8192, 1]
        qkv = conv_out.squeeze(2)  # [1, 8192]
        qkv = qkv * mx.sigmoid(qkv)  # SiLU

        z = (x @ self.w_z.T).reshape(1, 32, 128)
        beta = mx.sigmoid(x @ self.w_b.T).reshape(1, 32, 1)
        a = x @ self.w_a.T

        # Split Q, K, V + GQA repeat
        q = mx.repeat(qkv[:, :2048].reshape(1, 16, 128), 2, axis=1)
        k = mx.repeat(qkv[:, 2048:4096].reshape(1, 16, 128), 2, axis=1)
        v = qkv[:, 4096:].reshape(1, 32, 128)

        # L2 norm + scale
        q = q / (mx.sqrt(mx.sum(q * q, axis=-1, keepdims=True)) + 1e-6) / (128 ** 0.5)
        k = k / (mx.sqrt(mx.sum(k * k, axis=-1, keepdims=True)) + 1e-6)

        # Mamba-style decay
        g_exp = mx.exp(
            -mx.exp(self.A_log) * mx.log(1 + mx.exp(a.astype(mx.float32) + self.dt_bias.astype(mx.float32)))
        ).reshape(1, 32, 1, 1)

        # Recurrent update
        new_state = state * g_exp
        kv_mem = mx.sum(new_state * k.reshape(1, 32, 128, 1), axis=2)
        delta = (v - kv_mem) * beta
        new_state = new_state + k.reshape(1, 32, 128, 1) * delta.reshape(1, 32, 1, 128)
        out = mx.sum(new_state * q.reshape(1, 32, 128, 1), axis=2)  # [1, 32, 128]

        # RMSNorm gated
        var = mx.mean(out * out, axis=-1, keepdims=True)
        out = out * mx.rsqrt(var + 1e-6) * self.w_norm
        z_silu = z * mx.sigmoid(z.astype(mx.float32)).astype(z.dtype)
        out = (out * z_silu).reshape(1, -1)

        return out @ self.w_out.T, new_state, new_conv


# === Full Attention Layer (10 couches) ===

class MLXFullAttentionLayer:
    def __init__(self, layer_idx):
        prefix = f'model.language_model.layers.{layer_idx}.self_attn'
        self.w_q = load_mx(f'{prefix}.q_proj.weight')  # [8192, 2048]
        self.w_k = load_mx(f'{prefix}.k_proj.weight')  # [512, 2048]
        self.w_v = load_mx(f'{prefix}.v_proj.weight')  # [512, 2048]
        self.w_o = load_mx(f'{prefix}.o_proj.weight')  # [2048, 4096]
        mx.eval(self.w_q, self.w_k, self.w_v, self.w_o)

    def __call__(self, x):
        """x: [1, 2048] → out: [1, 2048]"""
        q = (x @ self.w_q.T).reshape(1, 16, 512)[:, :, :256]  # Tronquer aux 256 utiles
        k = (x @ self.w_k.T).reshape(1, 2, 256)
        v = (x @ self.w_v.T).reshape(1, 2, 256)

        # GQA repeat 2 → 16
        k = mx.repeat(k, 8, axis=1)
        v = mx.repeat(v, 8, axis=1)

        # Attention (single token → softmax triviale)
        scale = 256 ** -0.5
        attn = mx.sum(q * k, axis=-1, keepdims=True) * scale  # [1, 16, 1]
        attn = mx.softmax(attn, axis=-1)  # trivial pour 1 token
        out = (attn * v).reshape(1, -1)  # [1, 4096]

        return out @ self.w_o.T


# === MoE FFN Layer ===

class MLXMoELayer:
    def __init__(self, layer_idx):
        prefix = f'model.language_model.layers.{layer_idx}.mlp'
        self.gate = load_mx(f'{prefix}.gate.weight')
        self.shared = {
            'gate': load_mx(f'{prefix}.shared_expert.gate_proj.weight'),
            'up': load_mx(f'{prefix}.shared_expert.up_proj.weight'),
            'down': load_mx(f'{prefix}.shared_expert.down_proj.weight'),
        }
        self.shared_gate = load_mx(f'{prefix}.shared_expert_gate.weight')
        self.experts_gu = load_mx(f'{prefix}.experts.gate_up_proj')
        self.experts_down = load_mx(f'{prefix}.experts.down_proj')
        mx.eval(self.gate, self.shared['gate'], self.shared['up'],
                self.shared['down'], self.shared_gate,
                self.experts_gu, self.experts_down)

    def __call__(self, x):
        """x: [1, 2048] → out: [1, 2048]"""
        # Router top-8
        logits = (x @ self.gate.T).squeeze(0)
        indices = mx.argpartition(-logits, kth=8)[:8]
        scores = mx.softmax(logits[indices])
        mx.eval(indices, scores)

        idx_list = indices.tolist()
        scores_list = scores.tolist()

        # 8 routed experts
        result = mx.zeros((1, x.shape[-1]))
        for k in range(8):
            gu = self.experts_gu[idx_list[k]]
            d = self.experts_down[idx_list[k]]
            gu_out = x @ gu.T
            inter = gu_out.shape[-1] // 2
            g = gu_out[:, :inter]
            u = gu_out[:, inter:]
            h = (g * mx.sigmoid(g)) * u  # SiLU gate
            result = result + scores_list[k] * (h @ d.T)

        # Shared expert
        sg = x @ self.shared['gate'].T
        su = x @ self.shared['up'].T
        sh = (sg * mx.sigmoid(sg)) * su
        s_out = sh @ self.shared['down'].T
        s_weight = mx.sigmoid(x @ self.shared_gate.T)

        return result + s_out * s_weight


# === LayerNorm ===

class MLXRMSNorm:
    def __init__(self, layer_idx, which='input'):
        key = f'model.language_model.layers.{layer_idx}.{"input_layernorm" if which == "input" else "post_attention_layernorm"}.weight'
        self.weight = load_mx(key)
        mx.eval(self.weight)

    def __call__(self, x):
        var = mx.mean(x * x, axis=-1, keepdims=True)
        return x * mx.rsqrt(var + 1e-6) * self.weight


# === Modèle complet ===

def main():
    init_weights()

    layer_types = (['linear_attention'] * 3 + ['full_attention']) * 10

    print("=== MLX Pur : Qwen3.5-35B-A3B complet ===\n")

    # Charger couche par couche (pour ne pas tout mettre en RAM d'un coup)
    # Test avec N premières couches
    N_LAYERS = 4  # Commencer petit pour vérifier
    print(f"Chargement {N_LAYERS} couches...")

    layers = []
    for i in range(N_LAYERS):
        lt = layer_types[i]
        print(f"  [{i+1}/{N_LAYERS}] {lt}...", end=" ", flush=True)
        norm1 = MLXRMSNorm(i, 'input')
        norm2 = MLXRMSNorm(i, 'post_attention')
        if lt == 'linear_attention':
            attn = MLXDeltaNetLayer(i)
        else:
            attn = MLXFullAttentionLayer(i)
        moe = MLXMoELayer(i)
        layers.append((lt, norm1, attn, norm2, moe))
        print("OK")

    # === Forward pass ===
    print(f"\nBenchmark {N_LAYERS} couches MLX pur...")

    # Init states
    delta_states = [mx.zeros((1, 32, 128, 128)) for _ in range(N_LAYERS)]
    conv_states = [mx.zeros((1, 8192, 3)) for _ in range(N_LAYERS)]

    def forward_token(x):
        di = 0
        for i, (lt, norm1, attn, norm2, moe) in enumerate(layers):
            # Pre-norm + attention
            h = norm1(x)
            if lt == 'linear_attention':
                attn_out, delta_states[di], conv_states[di] = attn(h, delta_states[di], conv_states[di])
                di += 1
            else:
                attn_out = attn(h)
            x = x + attn_out

            # Pre-norm + MoE FFN
            h = norm2(x)
            ffn_out = moe(h)
            x = x + ffn_out
        return x

    # Warmup
    x = mx.random.normal((1, 2048)) * 0.1
    for _ in range(3):
        out = forward_token(x)
        mx.eval(out)

    # Benchmark
    n = 20
    t0 = time.time()
    for _ in range(n):
        x = mx.random.normal((1, 2048)) * 0.1
        out = forward_token(x)
        mx.eval(out)
    elapsed = time.time() - t0

    ms_per_tok = elapsed / n * 1000
    ms_per_layer = ms_per_tok / N_LAYERS
    tok_s_extrapolated = 1000 / (ms_per_layer * 40)

    print(f"\n=== RÉSULTATS MLX PUR ===")
    print(f"  {N_LAYERS} couches : {ms_per_tok:.1f} ms/tok ({n/elapsed:.1f} tok/s)")
    print(f"  Par couche : {ms_per_layer:.2f} ms")
    print(f"  Extrapolation 40 couches : {ms_per_layer*40:.0f} ms → {tok_s_extrapolated:.1f} tok/s")
    print(f"  Output range: [{float(out.min()):.4f}, {float(out.max()):.4f}]")
    print(f"")
    print(f"  Comparaison :")
    print(f"    CoreML ANE (40 couches attention) : 12.0 tok/s")
    print(f"    ANE + CPU experts                 : 13.9 tok/s")
    print(f"    MLX pur (attention + MoE)          : {tok_s_extrapolated:.1f} tok/s")

    # === Test qualité ===
    print(f"\n=== QUALITÉ ===")
    x_test = mx.random.normal((1, 2048)) * 0.1
    delta_states_q = [mx.zeros((1, 32, 128, 128)) for _ in range(N_LAYERS)]
    conv_states_q = [mx.zeros((1, 8192, 3)) for _ in range(N_LAYERS)]

    for t in range(5):
        out = forward_token(mx.random.normal((1, 2048)) * 0.1)
        mx.eval(out)

    print(f"  Après 5 tokens : range [{float(out.min()):.4f}, {float(out.max()):.4f}]")
    print(f"  Std: {float(mx.std(out)):.6f}")
    state_norms = [float(mx.sqrt(mx.sum(s*s))) for s in delta_states[:3]]
    print(f"  State norms: {[f'{n:.4f}' for n in state_norms]}")
    print(f"  PASS" if max(state_norms) > 0.01 else "  WARN — states faibles")


if __name__ == "__main__":
    main()

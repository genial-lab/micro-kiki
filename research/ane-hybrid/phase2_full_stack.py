#!/usr/bin/env python3
"""Phase 2 : Stack complet 40 couches → CoreML pour ANE.

Convertit les 30 couches DeltaNet + 10 couches Full Attention en CoreML.
Chaque couche = un .mlpackage séparé (mode decode single-token).

Benchmark : temps total pour 1 token à travers les 40 couches sur ANE.
"""

import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from deltanet_real import RealGatedDeltaNet, load_real_weights, l2norm

import coremltools as ct


# === Full Attention Layer (pour les couches 3, 7, 11, ...) ===

class RealFullAttention(nn.Module):
    """Couche Full Attention GQA de Qwen3.5-35B-A3B.

    Note: q_proj=[8192,2048] car head_dim_q=512 (256 rope + 256 nope).
    k_proj=[512,2048], v_proj=[512,2048] avec num_kv_heads=2, head_dim=256.
    o_proj=[2048,4096] car output = num_heads * head_dim = 16 * 256 = 4096...
    Mais en fait output=num_heads*v_head_dim.
    """

    def __init__(self, hidden_size=2048, num_heads=16, num_kv_heads=2, head_dim=256, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        # q_proj produit 8192 = num_heads * (head_dim * 2) — inclut QK-norm components
        self.q_head_dim = 512  # 8192 / 16
        self.q_proj = nn.Linear(hidden_size, num_heads * self.q_head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        # o_proj prend 4096 = num_heads * head_dim (après troncation de Q à head_dim)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(self, x, kv_cache_k=None, kv_cache_v=None, position=0):
        """Single-token decode avec KV cache.

        x: [1, 1, 2048]
        kv_cache_k: [1, num_kv_heads, cache_len, head_dim]
        kv_cache_v: [1, num_kv_heads, cache_len, head_dim]
        """
        B = 1
        q = self.q_proj(x).reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Pas de RoPE ici pour simplifier (Phase 2 = benchmark, Phase 3 = intégration complète)

        # KV cache append
        if kv_cache_k is not None:
            k = torch.cat([kv_cache_k, k], dim=2)
            v = torch.cat([kv_cache_v, v], dim=2)

        # GQA repeat
        repeat = self.num_heads // self.num_kv_heads
        k_expanded = k.repeat_interleave(repeat, dim=1)
        v_expanded = v.repeat_interleave(repeat, dim=1)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (q @ k_expanded.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = attn @ v_expanded

        out = out.transpose(1, 2).reshape(B, 1, -1)
        return self.o_proj(out), k, v


def load_full_attn_weights(model, layer_idx):
    """Charge les poids d'une couche Full Attention depuis HF."""
    from safetensors import safe_open

    snap_dir = list(Path('/Users/clems/.cache/huggingface/hub/models--Jackrong--Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled/snapshots/').iterdir())[0]
    with open(snap_dir / 'model.safetensors.index.json') as f:
        idx = json.load(f)

    prefix = f'model.language_model.layers.{layer_idx}.self_attn'
    shards_needed = set()
    for key, shard in idx['weight_map'].items():
        if prefix in key:
            shards_needed.add(shard)

    loaded = {}
    for shard_name in shards_needed:
        f = safe_open(str(snap_dir / shard_name), framework='pt')
        for key in f.keys():
            if prefix in key:
                loaded[key] = f.get_tensor(key)

    def get(name):
        return loaded[f'{prefix}.{name}']

    model.q_proj.weight.data = get('q_proj.weight').float()
    model.k_proj.weight.data = get('k_proj.weight').float()
    model.v_proj.weight.data = get('v_proj.weight').float()
    model.o_proj.weight.data = get('o_proj.weight').float()
    return model


# === Wrapper decode DeltaNet (traceable) ===

class DeltaNetDecodeTraceable(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x, state, conv_state):
        mixed_qkv = self.m.in_proj_qkv(x).transpose(1, 2)
        z = self.m.in_proj_z(x).reshape(1, 1, -1, 128)
        beta = self.m.in_proj_b(x).sigmoid()
        a = self.m.in_proj_a(x)

        combined = torch.cat([conv_state, mixed_qkv], dim=-1)
        new_conv = combined[:, :, -3:]
        mixed_qkv = F.conv1d(combined, self.m.conv1d.weight, None, padding=0, groups=8192)
        mixed_qkv = F.silu(mixed_qkv[:, :, -1:]).transpose(1, 2)

        q, k, v = torch.split(mixed_qkv, [2048, 2048, 4096], dim=-1)
        q = q.reshape(1, 1, 16, 128).repeat_interleave(2, dim=2)
        k = k.reshape(1, 1, 16, 128).repeat_interleave(2, dim=2)
        v = v.reshape(1, 1, 32, 128)

        g = -self.m.A_log.float().exp() * F.softplus(a.float() + self.m.dt_bias)

        q = l2norm(q).squeeze(1).float() / (128 ** 0.5)
        k = l2norm(k).squeeze(1).float()
        v = v.squeeze(1).float()
        g_t = g.squeeze(1).float().exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta.squeeze(1).float().unsqueeze(-1)

        s = state.float() * g_t
        kv_mem = (s * k.unsqueeze(-1)).sum(dim=-2)
        delta = (v - kv_mem) * beta_t
        new_state = s + k.unsqueeze(-1) * delta.unsqueeze(-2)

        out = (new_state * q.unsqueeze(-1)).sum(dim=-2)
        out_flat = out.reshape(-1, 128).to(x.dtype)
        z_flat = z.reshape(-1, 128)
        out_flat = self.m.norm(out_flat, z_flat)
        out = self.m.out_proj(out_flat.reshape(1, 1, -1))

        return out, new_state.to(x.dtype), new_conv


def convert_deltanet_layer(layer_idx):
    """Convertit une couche DeltaNet en CoreML."""
    model = RealGatedDeltaNet()
    model = load_real_weights(model, layer_idx=layer_idx)
    model.eval()

    wrapper = DeltaNetDecodeTraceable(model)
    wrapper.eval()

    x = torch.randn(1, 1, 2048)
    s = torch.zeros(1, 32, 128, 128)
    c = torch.zeros(1, 8192, 3)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (x, s, c))

    coreml = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=(1, 1, 2048)),
            ct.TensorType(name="state_in", shape=(1, 32, 128, 128)),
            ct.TensorType(name="conv_state_in", shape=(1, 8192, 3)),
        ],
        outputs=[
            ct.TensorType(name="output"),
            ct.TensorType(name="state_out"),
            ct.TensorType(name="conv_state_out"),
        ],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
    )
    return coreml


def convert_full_attn_layer(layer_idx):
    """Convertit une couche Full Attention en CoreML (sans KV cache pour simplifier)."""
    model = RealFullAttention()
    model = load_full_attn_weights(model, layer_idx)
    model.eval()

    # Wrapper simple sans KV cache (pour benchmark de base)
    class AttnTraceable(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            # q_proj=[8192,2048] → [1,1,16,512], on tronque à head_dim=256 pour l'attention
            q_full = self.m.q_proj(x).reshape(1, 1, 16, 512)
            q = q_full[:, :, :, :256].transpose(1, 2)  # Garder les 256 premiers dims
            k = self.m.k_proj(x).reshape(1, 1, 2, 256).transpose(1, 2)
            v = self.m.v_proj(x).reshape(1, 1, 2, 256).transpose(1, 2)
            k_e = k.repeat_interleave(8, dim=1)
            v_e = v.repeat_interleave(8, dim=1)
            attn = F.softmax((q @ k_e.transpose(-2, -1)) * (256 ** -0.5), dim=-1)
            out = (attn @ v_e).transpose(1, 2).reshape(1, 1, -1)
            return self.m.o_proj(out)

    wrapper = AttnTraceable(model)
    wrapper.eval()

    traced = torch.jit.trace(wrapper, torch.randn(1, 1, 2048))
    coreml = ct.convert(
        traced,
        inputs=[ct.TensorType(name="hidden_states", shape=(1, 1, 2048))],
        outputs=[ct.TensorType(name="output")],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
    )
    return coreml


def main():
    output_dir = Path("research/ane-hybrid/mlpackages/full_stack")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Config des couches
    layer_types = [
        'linear_attention', 'linear_attention', 'linear_attention', 'full_attention',
    ] * 10  # 40 couches

    print("=== Phase 2 : Conversion des 40 couches ===")
    print()

    converted = 0
    for i in range(40):
        lt = layer_types[i]
        tag = f"layer_{i:02d}_{lt[:3]}"
        path = output_dir / f"{tag}.mlpackage"

        if path.exists():
            print(f"  [{i+1}/40] {tag} — déjà converti")
            converted += 1
            continue

        try:
            t0 = time.time()
            if lt == 'linear_attention':
                coreml = convert_deltanet_layer(i)
            else:
                coreml = convert_full_attn_layer(i)
            coreml.save(str(path))
            elapsed = time.time() - t0
            converted += 1
            print(f"  [{i+1}/40] {tag} — OK ({elapsed:.1f}s)")
        except Exception as e:
            print(f"  [{i+1}/40] {tag} — ERREUR: {e}")

    print(f"\n{converted}/40 couches converties")

    # === Benchmark full stack ===
    if converted == 40:
        print("\n=== Benchmark full stack (CPU+ANE) ===")

        models = []
        for i in range(40):
            lt = layer_types[i]
            tag = f"layer_{i:02d}_{lt[:3]}"
            path = output_dir / f"{tag}.mlpackage"
            m = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
            models.append((lt, m))

        # Single token through all 40 layers
        x = np.random.randn(1, 1, 2048).astype(np.float16)
        states = [np.zeros((1, 32, 128, 128), dtype=np.float16) for _ in range(30)]
        conv_states = [np.zeros((1, 8192, 3), dtype=np.float16) for _ in range(30)]

        # Warmup
        print("Warmup...")
        delta_idx = 0
        for lt, m in models:
            if lt == 'linear_attention':
                inp = {"hidden_states": x, "state_in": states[delta_idx], "conv_state_in": conv_states[delta_idx]}
                r = m.predict(inp)
                x = r["output"]
                delta_idx += 1
            else:
                r = m.predict({"hidden_states": x})
                x = r["output"]

        # Benchmark
        n_tokens = 20
        print(f"Benchmark {n_tokens} tokens...")
        t0 = time.time()
        for _ in range(n_tokens):
            x = np.random.randn(1, 1, 2048).astype(np.float16)
            delta_idx = 0
            for lt, m in models:
                if lt == 'linear_attention':
                    inp = {"hidden_states": x, "state_in": states[delta_idx], "conv_state_in": conv_states[delta_idx]}
                    r = m.predict(inp)
                    x = r["output"]
                    states[delta_idx] = r["state_out"]
                    conv_states[delta_idx] = r["conv_state_out"]
                    delta_idx += 1
                else:
                    r = m.predict({"hidden_states": x})
                    x = r["output"]
        elapsed = time.time() - t0
        tok_s = n_tokens / elapsed
        ms_per_tok = elapsed / n_tokens * 1000

        print(f"\n=== RÉSULTAT ===")
        print(f"  {n_tokens} tokens en {elapsed:.2f}s")
        print(f"  {tok_s:.1f} tok/s ({ms_per_tok:.1f} ms/tok)")
        print(f"  30 DeltaNet + 10 Full Attention sur ANE")
    else:
        print(f"\nConversion incomplète ({converted}/40). Fix les erreurs puis relancer.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Implémentation exacte du GatedDeltaNet Qwen3.5 + chargement vrais poids.

Copié directement de modeling_qwen3_5_moe.py (transformers) pour garantir
l'équivalence numérique. Phase 1.4 du plan ANE hybrid.
"""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# === Fonctions utilitaires (identiques au code HF) ===

def l2norm(x, dim=-1, eps=1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


class RMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


# === Recurrent decode (single token) — cible ANE ===

def recurrent_gated_delta_rule(query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel=False):
    """Forme récurrente token-par-token. Ops: matmul, exp, sum — compatible CoreML."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(value)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


# === Chunk prefill — cible ANE (parallélisé) ===

def chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=64, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False):
    """Forme chunkwise parallèle. Identique au code HF transformers."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


# === Couche GatedDeltaNet complète (exacte HF) ===

class RealGatedDeltaNet(nn.Module):
    """Reproduction exacte de Qwen3_5MoeGatedDeltaNet."""

    def __init__(self, hidden_size=2048, num_k_heads=16, num_v_heads=32,
                 head_k_dim=128, head_v_dim=128, conv_kernel_size=4, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.key_dim = num_k_heads * head_k_dim      # 2048
        self.value_dim = num_v_heads * head_v_dim     # 4096
        self.conv_dim = self.key_dim * 2 + self.value_dim  # 8192
        self.conv_kernel_size = conv_kernel_size

        # Projections
        self.in_proj_qkv = nn.Linear(hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.in_proj_a = nn.Linear(hidden_size, num_v_heads, bias=False)
        self.in_proj_b = nn.Linear(hidden_size, num_v_heads, bias=False)

        # Conv1d depthwise causale
        self.conv1d = nn.Conv1d(
            self.conv_dim, self.conv_dim, kernel_size=conv_kernel_size,
            groups=self.conv_dim, padding=conv_kernel_size - 1, bias=False
        )

        # Mamba-style decay
        self.A_log = nn.Parameter(torch.log(torch.empty(num_v_heads).uniform_(0, 16)))
        self.dt_bias = nn.Parameter(torch.ones(num_v_heads))

        # Output
        self.norm = RMSNormGated(head_v_dim, eps=eps)
        self.out_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(self, hidden_states, state=None, conv_state=None):
        """
        Args:
            hidden_states: [B, T, hidden_size]
            state: [B, num_v_heads, head_k_dim, head_v_dim] ou None
            conv_state: [B, conv_dim, conv_kernel_size-1] ou None
        Returns:
            output: [B, T, hidden_size]
            new_state: [B, num_v_heads, head_k_dim, head_v_dim]
            new_conv_state: [B, conv_dim, conv_kernel_size-1]
        """
        B, T, _ = hidden_states.shape
        is_decode = (T == 1 and state is not None)

        # Projections
        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)  # [B, conv_dim, T]
        z = self.in_proj_z(hidden_states).reshape(B, T, -1, self.head_v_dim)
        beta = self.in_proj_b(hidden_states).sigmoid()
        a = self.in_proj_a(hidden_states)

        # Conv1d
        if is_decode and conv_state is not None:
            # Single token: update conv state
            combined = torch.cat([conv_state, mixed_qkv], dim=-1)
            new_conv_state = combined[:, :, -(self.conv_kernel_size - 1):]
            mixed_qkv = F.conv1d(
                combined, self.conv1d.weight, self.conv1d.bias,
                padding=0, groups=self.conv_dim
            )
            mixed_qkv = F.silu(mixed_qkv[:, :, -T:])
        else:
            if conv_state is None:
                new_conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - 1 - mixed_qkv.shape[-1], 0))
            else:
                new_conv_state = mixed_qkv[:, :, -(self.conv_kernel_size - 1):]
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :T])

        mixed_qkv = mixed_qkv.transpose(1, 2)

        # Split Q, K, V
        q, k, v = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(B, T, self.num_k_heads, self.head_k_dim)
        k = k.reshape(B, T, self.num_k_heads, self.head_k_dim)
        v = v.reshape(B, T, self.num_v_heads, self.head_v_dim)

        # Mamba-style decay
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        # GQA repeat K heads → V heads
        if self.num_v_heads > self.num_k_heads:
            repeat = self.num_v_heads // self.num_k_heads
            q = q.repeat_interleave(repeat, dim=2)
            k = k.repeat_interleave(repeat, dim=2)

        # Core attention
        if is_decode:
            core_out, new_state = recurrent_gated_delta_rule(
                q, k, v, g=g, beta=beta,
                initial_state=state, output_final_state=True,
                use_qk_l2norm_in_kernel=True
            )
        else:
            core_out, new_state = chunk_gated_delta_rule(
                q, k, v, g=g, beta=beta,
                initial_state=state, output_final_state=True,
                use_qk_l2norm_in_kernel=True
            )

        # Output gating
        core_out = core_out.reshape(-1, self.head_v_dim)
        z_flat = z.reshape(-1, self.head_v_dim)
        core_out = self.norm(core_out, z_flat)
        core_out = core_out.reshape(B, T, -1)
        output = self.out_proj(core_out)

        return output, new_state, new_conv_state


def load_real_weights(model, layer_idx=0):
    """Charge les vrais poids depuis le modèle HF Qwen3.5-35B-A3B-Opus."""
    from safetensors import safe_open

    snap_dir = list(Path('/Users/clems/.cache/huggingface/hub/models--Jackrong--Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled/snapshots/').iterdir())[0]

    # Lire l'index pour trouver les shards
    with open(snap_dir / 'model.safetensors.index.json') as f:
        idx = json.load(f)

    prefix = f'model.language_model.layers.{layer_idx}.linear_attn'
    weight_map = {}
    for key, shard in idx['weight_map'].items():
        if prefix in key:
            weight_map[key] = shard

    # Charger les poids par shard
    loaded = {}
    shards_needed = set(weight_map.values())
    for shard_name in shards_needed:
        shard_path = snap_dir / shard_name
        f = safe_open(str(shard_path), framework='pt')
        for key in f.keys():
            if prefix in key:
                loaded[key] = f.get_tensor(key)

    # Mapper vers notre modèle
    def get(name):
        return loaded[f'{prefix}.{name}']

    model.in_proj_qkv.weight.data = get('in_proj_qkv.weight').float()
    model.in_proj_z.weight.data = get('in_proj_z.weight').float()
    model.in_proj_a.weight.data = get('in_proj_a.weight').float()
    model.in_proj_b.weight.data = get('in_proj_b.weight').float()
    model.conv1d.weight.data = get('conv1d.weight').float()
    model.A_log.data = get('A_log').float()
    model.dt_bias.data = get('dt_bias').float()
    model.norm.weight.data = get('norm.weight').float()
    model.out_proj.weight.data = get('out_proj.weight').float()

    print(f"Poids chargés pour layer {layer_idx}")
    return model


def test_real_weights():
    """Test avec les vrais poids Qwen3.5."""
    print("=== Phase 1.4 : Test DeltaNet avec vrais poids ===")
    print()

    # Créer le modèle avec la config réelle
    model = RealGatedDeltaNet(
        hidden_size=2048,
        num_k_heads=16,
        num_v_heads=32,
        head_k_dim=128,
        head_v_dim=128,
        conv_kernel_size=4,
    )
    model.eval()

    # Charger les vrais poids
    model = load_real_weights(model, layer_idx=0)

    B, T = 1, 128
    x = torch.randn(B, T, 2048)

    print(f"Input: [{B}, {T}, 2048]")

    # Test prefill (chunkwise)
    with torch.no_grad():
        t0 = time.time()
        out_prefill, state, conv_state = model(x)
        t_prefill = time.time() - t0
    print(f"Prefill {T} tokens: {t_prefill*1000:.1f} ms")
    print(f"  Output: {list(out_prefill.shape)}")
    print(f"  State: {list(state.shape)}")
    print(f"  Conv state: {list(conv_state.shape)}")
    print(f"  Output range: [{out_prefill.min().item():.4f}, {out_prefill.max().item():.4f}]")

    # Test decode (recurrent, token par token)
    print()
    decode_input = torch.randn(1, 1, 2048)
    n_decode = 20
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_decode):
            out_decode, state, conv_state = model(decode_input, state=state, conv_state=conv_state)
    t_decode = time.time() - t0
    tok_per_sec = n_decode / t_decode
    print(f"Decode {n_decode} tokens: {t_decode*1000:.1f} ms ({tok_per_sec:.1f} tok/s PyTorch CPU)")
    print(f"  Output: {list(out_decode.shape)}")
    print(f"  Output range: [{out_decode.min().item():.4f}, {out_decode.max().item():.4f}]")

    # Vérifier que les sorties ne sont pas nulles (vrais poids → sorties significatives)
    assert out_prefill.abs().max() > 0.01, "FAIL: Sorties proches de zéro"
    assert out_decode.abs().max() > 0.01, "FAIL: Sorties decode proches de zéro"
    print()
    print("PASS — Sorties non-nulles avec vrais poids")

    return model, state, conv_state


if __name__ == "__main__":
    test_real_weights()

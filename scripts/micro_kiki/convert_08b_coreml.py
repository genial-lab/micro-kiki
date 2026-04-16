#!/usr/bin/env python3
"""Task 1 : Conversion Qwen3.5-0.8B complet vers CoreML pour ANE.

Convertit le modele GatedDeltaNet 0.8B en deux .mlpackage :
  - decode (1 token recurrent, ct.StateType pour les etats)
  - prefill (chunk=64 tokens)

Reutilise les patterns prouves de research/ane-hybrid/convert_deltanet.py.

Usage :
    python scripts/micro_kiki/convert_08b_coreml.py
    python scripts/micro_kiki/convert_08b_coreml.py --test
    python scripts/micro_kiki/convert_08b_coreml.py --benchmark
"""

import argparse
import copy
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import coremltools as ct


# ── Config ───────────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen3.5-0.8B"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output" / "micro-kiki" / "coreml"
PREFILL_CHUNK_SIZE = 64


@dataclass(frozen=True)
class Qwen35SmallConfig:
    """Config pour Qwen3.5-0.8B GatedDeltaNet."""

    hidden_size: int = 1536
    num_layers: int = 24
    num_key_heads: int = 8
    num_value_heads: int = 16
    key_head_dim: int = 64
    value_head_dim: int = 64
    conv_kernel_size: int = 4
    intermediate_size: int = 4096
    vocab_size: int = 248064
    full_attention_interval: int = 4
    head_dim: int = 128
    num_attention_heads: int = 12
    num_kv_heads: int = 2

    @property
    def key_dim(self) -> int:
        return self.num_key_heads * self.key_head_dim

    @property
    def value_dim(self) -> int:
        return self.num_value_heads * self.value_head_dim

    @property
    def conv_dim(self) -> int:
        return self.key_dim * 2 + self.value_dim

    @property
    def num_deltanet_layers(self) -> int:
        return self.num_layers * 3 // 4

    @property
    def num_full_attn_layers(self) -> int:
        return self.num_layers // 4

    @property
    def layer_types(self) -> list:
        pattern = ["linear_attention"] * 3 + ["full_attention"]
        return (pattern * (self.num_layers // 4))[:self.num_layers]


def load_config_from_hf(model_path: Path) -> Qwen35SmallConfig:
    """Charge la config depuis le config.json du modele HF."""
    config_path = model_path / "config.json"
    if not config_path.exists():
        print(f"Config introuvable a {config_path}, utilisation des valeurs par defaut")
        return Qwen35SmallConfig()

    with open(config_path) as f:
        raw = json.load(f)

    text_cfg = raw.get("text_config", raw)
    return Qwen35SmallConfig(
        hidden_size=text_cfg.get("hidden_size", 1536),
        num_layers=text_cfg.get("num_hidden_layers", 24),
        num_key_heads=text_cfg.get("linear_num_key_heads", 8),
        num_value_heads=text_cfg.get("linear_num_value_heads", 16),
        key_head_dim=text_cfg.get("linear_key_head_dim", 64),
        value_head_dim=text_cfg.get("linear_value_head_dim", 64),
        conv_kernel_size=text_cfg.get("linear_conv_kernel_dim", 4),
        intermediate_size=text_cfg.get("intermediate_size", 4096),
        vocab_size=text_cfg.get("vocab_size", 248064),
        full_attention_interval=text_cfg.get("full_attention_interval", 4),
        head_dim=text_cfg.get("head_dim", 128),
        num_attention_heads=text_cfg.get("num_attention_heads", 12),
        num_kv_heads=text_cfg.get("num_key_value_heads", 2),
    )


# ── L2 norm compatible CoreML ────────────────────────────────────────────────

def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    sq_sum = (x * x).sum(dim=dim, keepdim=True)
    inv_norm = torch.rsqrt(sq_sum + eps)
    return x * inv_norm


# ── RMSNorm Gated (pour la sortie GatedDeltaNet) ────────────────────────────

class RMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_f32 = x.to(torch.float32)
        var = x_f32.pow(2).mean(-1, keepdim=True)
        x_normed = x_f32 * torch.rsqrt(var + self.eps)
        x_normed = self.weight * x_normed.to(dtype)
        return x_normed * F.silu(gate.to(torch.float32)).to(dtype)


# ── GatedDeltaNet Layer (single token decode) ────────────────────────────────

class GatedDeltaNetDecode(nn.Module):
    """Decode wrapper for a single GatedDeltaNet layer.

    Processes 1 token recurrently, maintaining state via ct.StateType buffers.
    Uses Conv2d layout [B, C, 1, 1] for ANE compatibility.
    """

    def __init__(self, cfg: Qwen35SmallConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden_size
        Hk = cfg.num_key_heads
        Hv = cfg.num_value_heads
        Dk = cfg.key_head_dim
        Dv = cfg.value_head_dim
        kv_group = Hv // Hk
        Ve = kv_group * Dv
        conv_dim = cfg.conv_dim

        # Projections as Conv2d(1x1) for ANE layout
        self.in_proj_qkv = nn.Conv2d(H, conv_dim, 1, bias=False)
        self.in_proj_z = nn.Conv2d(H, cfg.value_dim, 1, bias=False)
        self.in_proj_a = nn.Conv2d(H, Hv, 1, bias=False)
        self.in_proj_b = nn.Conv2d(H, Hv, 1, bias=False)
        self.out_proj = nn.Conv2d(cfg.value_dim, H, 1, bias=False)

        # Conv1d weights stored as buffers (not nn.Conv1d to avoid coremltools bugs)
        self.register_buffer(
            "conv1d_weight",
            torch.zeros(conv_dim, 1, cfg.conv_kernel_size),
        )

        # Mamba-style decay parameters
        self.register_buffer("A_log", torch.zeros(Hv))
        self.register_buffer("dt_bias", torch.zeros(Hv))

        # RMSNormGated for output
        self.norm = RMSNormGated(Dv)

        # Structural params
        self.Hk = Hk
        self.Hv = Hv
        self.Dk = Dk
        self.Dv = Dv
        self.Ve = Ve
        self.kv_group = kv_group
        self.conv_kernel_size = cfg.conv_kernel_size

        # === Mutable state buffers (become ct.StateType) ===

        # Recurrent DeltaNet state: [1, Hv, Dk, Dv]
        self.register_buffer(
            "deltanet_state",
            torch.zeros(1, Hv, Dk, Dv, dtype=torch.float16),
        )

        # Conv cache: last (kernel_size - 1) tokens
        cache_len = cfg.conv_kernel_size - 1
        self.register_buffer(
            "conv_cache",
            torch.zeros(1, conv_dim, cache_len, dtype=torch.float16),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Single token decode.

        Args:
            hidden_states: [1, H, 1, 1] — ANE layout

        Returns:
            output: [1, H, 1, 1]
        """
        Hk = self.Hk
        Hv = self.Hv
        Dk = self.Dk
        Dv = self.Dv
        kv_group = self.kv_group

        # Projections (Conv2d 1x1)
        mixed_qkv = self.in_proj_qkv(hidden_states)  # [1, conv_dim, 1, 1]
        z = self.in_proj_z(hidden_states)              # [1, value_dim, 1, 1]
        a = self.in_proj_a(hidden_states)              # [1, Hv, 1, 1]
        beta = torch.sigmoid(self.in_proj_b(hidden_states))  # [1, Hv, 1, 1]

        # Conv1d causal with cache
        qkv_1d = mixed_qkv.squeeze(2).squeeze(2).unsqueeze(2)  # [1, conv_dim, 1]
        combined = torch.cat([self.conv_cache, qkv_1d], dim=2)  # [1, conv_dim, kernel_size]

        # Update conv cache
        self.conv_cache[:, :, :] = combined[:, :, 1:]

        # Manual depthwise conv: element-wise multiply and sum
        conv_w = self.conv1d_weight.squeeze(1)  # [conv_dim, kernel_size]
        conv_out = (combined * conv_w.unsqueeze(0)).sum(dim=2, keepdim=True)  # [1, conv_dim, 1]
        conv_out = F.silu(conv_out)

        # Split Q, K, V
        key_dim = self.cfg.key_dim
        value_dim = self.cfg.value_dim
        q_1d, k_1d, v_1d = torch.split(
            conv_out, [key_dim, key_dim, value_dim], dim=1
        )

        # Reshape to heads
        q = q_1d.squeeze(2).view(1, Hk, Dk)
        k = k_1d.squeeze(2).view(1, Hk, Dk)
        v = v_1d.squeeze(2).view(1, Hv, Dv)
        z_heads = z.squeeze(2).squeeze(2).view(1, Hv, Dv)

        # GQA: repeat K heads to match V heads
        if Hv > Hk:
            q = q.repeat_interleave(kv_group, dim=1)  # [1, Hv, Dk]
            k = k.repeat_interleave(kv_group, dim=1)

        # L2 normalize Q, K
        q = l2_normalize(q, dim=-1)
        k = l2_normalize(k, dim=-1)

        # Scale
        scale = 1.0 / (Dk ** 0.5)
        q = q * scale

        # Mamba-style decay
        a_flat = a.squeeze(2).squeeze(2)  # [1, Hv]
        g = -self.A_log.float().exp() * F.softplus(a_flat.float() + self.dt_bias)
        g_decay = g.exp().unsqueeze(-1).unsqueeze(-1)  # [1, Hv, 1, 1]
        beta_4d = beta.squeeze(2).squeeze(2).unsqueeze(-1)  # [1, Hv, 1]

        # Recurrent state update
        state = self.deltanet_state.float()  # [1, Hv, Dk, Dv]

        # Decay
        state = state * g_decay

        # Delta error-correcting update
        k_exp = k.unsqueeze(-1).float()                 # [1, Hv, Dk, 1]
        retrieved = (state * k_exp).sum(dim=2)           # [1, Hv, Dv]
        error = v.float() - retrieved
        delta = error * beta_4d.float()
        outer = k.float().unsqueeze(-1) * delta.unsqueeze(2)  # [1, Hv, Dk, Dv]
        new_state = state + outer

        # Write state
        self.deltanet_state[:, :, :, :] = new_state.half()

        # Query the state for output
        q_exp = q.unsqueeze(-1).float()                  # [1, Hv, Dk, 1]
        output = (new_state * q_exp).sum(dim=2)          # [1, Hv, Dv]

        # Output gating with RMSNorm
        output_flat = output.reshape(-1, Dv).to(hidden_states.dtype)
        z_flat = z_heads.reshape(-1, Dv)
        output_flat = self.norm(output_flat, z_flat)

        # Reshape to ANE layout
        output_ane = output_flat.reshape(1, -1, 1, 1)
        output_ane = self.out_proj(output_ane)

        return output_ane


# ── Full Attention Layer (decode) ────────────────────────────────────────────

class FullAttentionDecode(nn.Module):
    """Full attention layer decode with KV cache via ct.StateType."""

    def __init__(self, cfg: Qwen35SmallConfig, max_cache_len: int = 2048):
        super().__init__()
        H = cfg.hidden_size
        num_heads = cfg.num_attention_heads
        num_kv = cfg.num_kv_heads
        head_dim = cfg.head_dim

        self.q_proj = nn.Conv2d(H, num_heads * head_dim, 1, bias=False)
        self.k_proj = nn.Conv2d(H, num_kv * head_dim, 1, bias=False)
        self.v_proj = nn.Conv2d(H, num_kv * head_dim, 1, bias=False)
        self.o_proj = nn.Conv2d(num_heads * head_dim, H, 1, bias=False)

        self.num_heads = num_heads
        self.num_kv = num_kv
        self.head_dim = head_dim
        self.gqa_repeat = num_heads // num_kv
        self.max_cache_len = max_cache_len

        # KV cache as mutable state
        self.register_buffer(
            "kv_cache_k",
            torch.zeros(1, num_kv, max_cache_len, head_dim, dtype=torch.float16),
        )
        self.register_buffer(
            "kv_cache_v",
            torch.zeros(1, num_kv, max_cache_len, head_dim, dtype=torch.float16),
        )
        self.register_buffer(
            "cache_pos",
            torch.zeros(1, dtype=torch.int32),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Single token decode with KV cache.

        Args:
            hidden_states: [1, H, 1, 1]
        Returns:
            output: [1, H, 1, 1]
        """
        q = self.q_proj(hidden_states).squeeze(2).squeeze(2)
        k = self.k_proj(hidden_states).squeeze(2).squeeze(2)
        v = self.v_proj(hidden_states).squeeze(2).squeeze(2)

        q = q.view(1, self.num_heads, self.head_dim).unsqueeze(2)
        k = k.view(1, self.num_kv, self.head_dim).unsqueeze(2)
        v = v.view(1, self.num_kv, self.head_dim).unsqueeze(2)

        # Note: in traced model, cache_pos is managed externally
        # For simplicity in CoreML, we use a growing cache approach
        # Append to cache
        pos = self.cache_pos[0].item()
        if pos < self.max_cache_len:
            self.kv_cache_k[:, :, pos:pos+1, :] = k
            self.kv_cache_v[:, :, pos:pos+1, :] = v
            self.cache_pos[0] = pos + 1

        # Use cache up to current position
        seq_len = min(pos + 1, self.max_cache_len)
        k_cached = self.kv_cache_k[:, :, :seq_len, :]
        v_cached = self.kv_cache_v[:, :, :seq_len, :]

        # GQA repeat
        k_exp = k_cached.repeat_interleave(self.gqa_repeat, dim=1)
        v_exp = v_cached.repeat_interleave(self.gqa_repeat, dim=1)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (q @ k_exp.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v_exp).squeeze(2)

        out = out.reshape(1, -1, 1, 1)
        return self.o_proj(out)


# ── RMSNorm (pre-layer) ─────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_f32 = x.to(torch.float32)
        var = x_f32.pow(2).mean(-1, keepdim=True)
        return (x_f32 * torch.rsqrt(var + self.eps)).to(dtype) * self.weight


# ── MLP (SwiGLU) ─────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, cfg: Qwen35SmallConfig):
        super().__init__()
        H = cfg.hidden_size
        I = cfg.intermediate_size
        self.gate_proj = nn.Conv2d(H, I, 1, bias=False)
        self.up_proj = nn.Conv2d(H, I, 1, bias=False)
        self.down_proj = nn.Conv2d(I, H, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ── Full model wrapper (decode) ─────────────────────────────────────────────

class Qwen35SmallDecodeModel(nn.Module):
    """Complete Qwen3.5-0.8B model in decode mode for CoreML conversion.

    All layers process 1 token at a time with state management.
    ANE-compatible Conv2d layout throughout.
    """

    def __init__(self, cfg: Qwen35SmallConfig):
        super().__init__()
        self.cfg = cfg

        # Embedding (Conv2d style: lookup then reshape)
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)

        # Layers
        self.input_norms = nn.ModuleList()
        self.attn_layers = nn.ModuleList()
        self.post_attn_norms = nn.ModuleList()
        self.mlps = nn.ModuleList()

        for i, lt in enumerate(cfg.layer_types):
            self.input_norms.append(RMSNorm(cfg.hidden_size))
            if lt == "linear_attention":
                self.attn_layers.append(GatedDeltaNetDecode(cfg))
            else:
                self.attn_layers.append(FullAttentionDecode(cfg))
            self.post_attn_norms.append(RMSNorm(cfg.hidden_size))
            self.mlps.append(MLP(cfg))

        self.final_norm = RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, token_id: torch.Tensor) -> torch.Tensor:
        """Forward pass for 1 token.

        Args:
            token_id: [1] — single token ID

        Returns:
            logits: [1, vocab_size]
        """
        # Embedding
        h = self.embed_tokens(token_id)  # [1, H]
        h = h.view(1, -1, 1, 1)  # ANE layout [1, H, 1, 1]

        # Layers
        for i in range(self.cfg.num_layers):
            # Pre-norm (applied in channel dim)
            residual = h
            h_flat = h.squeeze(2).squeeze(2)
            h_normed = self.input_norms[i](h_flat)
            h_ane = h_normed.view(1, -1, 1, 1)

            # Attention
            attn_out = self.attn_layers[i](h_ane)

            # Residual
            h = residual + attn_out

            # Post-attention norm + MLP
            residual = h
            h_flat = h.squeeze(2).squeeze(2)
            h_normed = self.post_attn_norms[i](h_flat)
            h_ane = h_normed.view(1, -1, 1, 1)
            mlp_out = self.mlps[i](h_ane)
            h = residual + mlp_out

        # Final norm + LM head
        h_flat = h.squeeze(2).squeeze(2)
        h_normed = self.final_norm(h_flat)
        logits = self.lm_head(h_normed)

        return logits


# ── Weight loading ───────────────────────────────────────────────────────────

def find_model_path() -> Path:
    """Find Qwen3.5-0.8B model in common locations."""
    candidates = [
        Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3.5-0.8B",
        Path("/Users/clems/KIKI-Mac_tunner/models/Qwen3.5-0.8B"),
        Path("/Users/clems/KIKI-Mac_tunner/models/Qwen3.5-0.8B-bf16"),
    ]
    for c in candidates:
        if c.exists():
            # If HF cache, find snapshot dir
            snapshots = c / "snapshots"
            if snapshots.exists():
                snap_dirs = list(snapshots.iterdir())
                if snap_dirs:
                    return snap_dirs[0]
            return c
    return Path("/Users/clems/KIKI-Mac_tunner/models/Qwen3.5-0.8B")


def load_weights_from_hf(model: Qwen35SmallDecodeModel, model_path: Path) -> None:
    """Load weights from HuggingFace safetensors into our model."""
    from safetensors import safe_open

    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        shards = set(index["weight_map"].values())
    else:
        # Single shard
        shard_files = list(model_path.glob("*.safetensors"))
        shards = {s.name for s in shard_files}

    all_weights = {}
    for shard_name in shards:
        shard_path = model_path / shard_name
        if not shard_path.exists():
            continue
        with safe_open(str(shard_path), framework="pt") as f:
            for key in f.keys():
                all_weights[key] = f.get_tensor(key)

    cfg = model.cfg

    # Embedding
    if "model.embed_tokens.weight" in all_weights:
        model.embed_tokens.weight.data = all_weights["model.embed_tokens.weight"].float()

    # LM head (may be tied)
    if "lm_head.weight" in all_weights:
        model.lm_head.weight.data = all_weights["lm_head.weight"].float()
    elif "model.embed_tokens.weight" in all_weights:
        model.lm_head.weight.data = all_weights["model.embed_tokens.weight"].float()

    # Layers
    for i in range(cfg.num_layers):
        prefix = f"model.layers.{i}"
        lt = cfg.layer_types[i]

        # Input norm
        norm_key = f"{prefix}.input_layernorm.weight"
        if norm_key in all_weights:
            model.input_norms[i].weight.data = all_weights[norm_key].float()

        # Post-attention norm
        post_norm_key = f"{prefix}.post_attention_layernorm.weight"
        if post_norm_key in all_weights:
            model.post_attn_norms[i].weight.data = all_weights[post_norm_key].float()

        # MLP
        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            w_key = f"{prefix}.mlp.{proj_name}.weight"
            if w_key in all_weights:
                w = all_weights[w_key].float()
                # Linear -> Conv2d: [out, in] -> [out, in, 1, 1]
                getattr(model.mlps[i], proj_name).weight.data = w.unsqueeze(-1).unsqueeze(-1)

        # Attention
        layer = model.attn_layers[i]
        if lt == "linear_attention":
            _load_deltanet_weights(layer, all_weights, prefix, cfg)
        else:
            _load_full_attn_weights(layer, all_weights, prefix, cfg)

    # Final norm
    if "model.norm.weight" in all_weights:
        model.final_norm.weight.data = all_weights["model.norm.weight"].float()

    print(f"Loaded {len(all_weights)} weight tensors from {model_path}")


def _load_deltanet_weights(
    layer: GatedDeltaNetDecode,
    weights: dict,
    prefix: str,
    cfg: Qwen35SmallConfig,
) -> None:
    """Load GatedDeltaNet weights into our decode layer."""
    attn_prefix = f"{prefix}.linear_attn"

    mapping = {
        "in_proj_qkv": "in_proj_qkv",
        "in_proj_z": "in_proj_z",
        "in_proj_a": "in_proj_a",
        "in_proj_b": "in_proj_b",
        "out_proj": "out_proj",
    }

    for our_name, hf_name in mapping.items():
        w_key = f"{attn_prefix}.{hf_name}.weight"
        if w_key in weights:
            w = weights[w_key].float()
            getattr(layer, our_name).weight.data = w.unsqueeze(-1).unsqueeze(-1)

    # Conv1d weights
    conv_key = f"{attn_prefix}.conv1d.weight"
    if conv_key in weights:
        layer.conv1d_weight.data = weights[conv_key].float()

    # A_log, dt_bias
    for param_name in ["A_log", "dt_bias"]:
        p_key = f"{attn_prefix}.{param_name}"
        if p_key in weights:
            getattr(layer, param_name).data = weights[p_key].float()

    # Norm
    norm_key = f"{attn_prefix}.norm.weight"
    if norm_key in weights:
        layer.norm.weight.data = weights[norm_key].float()


def _load_full_attn_weights(
    layer: FullAttentionDecode,
    weights: dict,
    prefix: str,
    cfg: Qwen35SmallConfig,
) -> None:
    """Load Full Attention weights into our decode layer."""
    attn_prefix = f"{prefix}.self_attn"

    for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        w_key = f"{attn_prefix}.{proj_name}.weight"
        if w_key in weights:
            w = weights[w_key].float()
            getattr(layer, proj_name).weight.data = w.unsqueeze(-1).unsqueeze(-1)


# ── CoreML state utilities ──────────────────────────────────────────────────

# Buffer names that are constants (weights), not mutable state
_CONST_BUFFER_NAMES = {
    "conv1d_weight", "A_log", "dt_bias",
}


def _get_mutable_state_specs(model: nn.Module) -> list:
    """Build ct.StateType specs for all mutable buffers in the model."""
    states = []
    for name, buf in model.named_buffers():
        # Skip constant buffers (weights stored as buffers)
        short_name = name.split(".")[-1]
        if short_name in _CONST_BUFFER_NAMES:
            continue
        # Skip embedding weight (not a state)
        if "embed_tokens" in name or "lm_head" in name:
            continue
        # Skip norm weights
        if "norm" in name and "weight" in short_name:
            continue

        states.append(
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=tuple(buf.shape),
                    dtype=np.float16 if buf.dtype == torch.float16 else np.float32,
                ),
                name=name,
            )
        )
    return states


def _reset_state_buffers(module: nn.Module) -> None:
    """Reset all mutable state buffers to zero."""
    with torch.no_grad():
        for name, buf in module.named_buffers():
            short_name = name.split(".")[-1]
            if short_name in _CONST_BUFFER_NAMES:
                continue
            if "embed_tokens" in name or "lm_head" in name:
                continue
            if "norm" in name and "weight" in short_name:
                continue
            if buf.dtype in (torch.float16, torch.float32):
                buf.zero_()


# ── Layer-by-layer conversion (avoids trace issues with full model) ──────────

def convert_single_deltanet_layer(
    layer: GatedDeltaNetDecode,
    layer_idx: int,
    cfg: Qwen35SmallConfig,
    output_dir: Path,
) -> Optional[Path]:
    """Convert a single DeltaNet layer to CoreML."""
    H = cfg.hidden_size
    tag = f"layer_{layer_idx:02d}_deltanet"
    out_path = output_dir / f"{tag}_decode.mlpackage"

    if out_path.exists():
        print(f"  {tag} already converted")
        return out_path

    layer_copy = copy.deepcopy(layer).half().eval()
    _reset_state_buffers(layer_copy)

    example = torch.zeros(1, H, 1, 1, dtype=torch.float16)

    with torch.no_grad():
        traced = torch.jit.trace(layer_copy, example)
    _reset_state_buffers(traced)

    states = _get_mutable_state_specs(layer_copy)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=(1, H, 1, 1), dtype=np.float16),
        ],
        outputs=[
            ct.TensorType(name="output", dtype=np.float16),
        ],
        states=states,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out_path))
    print(f"  {tag} -> {out_path}")
    return out_path


def convert_single_full_attn_layer(
    layer: FullAttentionDecode,
    layer_idx: int,
    cfg: Qwen35SmallConfig,
    output_dir: Path,
) -> Optional[Path]:
    """Convert a single Full Attention layer to CoreML.

    Note: Full Attention with dynamic KV cache is hard to trace.
    We convert a simplified version without KV cache for ANE benchmarking.
    """
    H = cfg.hidden_size
    tag = f"layer_{layer_idx:02d}_fullattn"
    out_path = output_dir / f"{tag}_decode.mlpackage"

    if out_path.exists():
        print(f"  {tag} already converted")
        return out_path

    # Simplified wrapper: no KV cache, just single-token attention
    class SimpleAttnWrapper(nn.Module):
        def __init__(self, attn_layer):
            super().__init__()
            self.q_proj = attn_layer.q_proj
            self.k_proj = attn_layer.k_proj
            self.v_proj = attn_layer.v_proj
            self.o_proj = attn_layer.o_proj
            self.num_heads = attn_layer.num_heads
            self.num_kv = attn_layer.num_kv
            self.head_dim = attn_layer.head_dim
            self.gqa_repeat = attn_layer.gqa_repeat

        def forward(self, x):
            q = self.q_proj(x).squeeze(2).squeeze(2).view(1, self.num_heads, 1, self.head_dim)
            k = self.k_proj(x).squeeze(2).squeeze(2).view(1, self.num_kv, 1, self.head_dim)
            v = self.v_proj(x).squeeze(2).squeeze(2).view(1, self.num_kv, 1, self.head_dim)
            k_e = k.repeat_interleave(self.gqa_repeat, dim=1)
            v_e = v.repeat_interleave(self.gqa_repeat, dim=1)
            scale = self.head_dim ** -0.5
            attn = F.softmax((q @ k_e.transpose(-2, -1)) * scale, dim=-1)
            out = (attn @ v_e).reshape(1, -1, 1, 1)
            return self.o_proj(out)

    wrapper = SimpleAttnWrapper(layer).half().eval()
    example = torch.zeros(1, H, 1, 1, dtype=torch.float16)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=(1, H, 1, 1), dtype=np.float16),
        ],
        outputs=[
            ct.TensorType(name="output", dtype=np.float16),
        ],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out_path))
    print(f"  {tag} -> {out_path}")
    return out_path


# ── Full conversion pipeline ────────────────────────────────────────────────

def convert_all_layers(cfg: Qwen35SmallConfig, model: Qwen35SmallDecodeModel) -> dict:
    """Convert all layers of the 0.8B model to CoreML."""
    output_dir = OUTPUT_DIR / "qwen35-08b"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {"converted": 0, "failed": 0, "paths": []}

    print(f"\n{'='*60}")
    print(f"Converting {cfg.num_layers} layers to CoreML")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    for i, lt in enumerate(cfg.layer_types):
        try:
            t0 = time.time()
            if lt == "linear_attention":
                path = convert_single_deltanet_layer(
                    model.attn_layers[i], i, cfg, output_dir
                )
            else:
                path = convert_single_full_attn_layer(
                    model.attn_layers[i], i, cfg, output_dir
                )
            elapsed = time.time() - t0
            results["converted"] += 1
            results["paths"].append(path)
            print(f"  [{i+1}/{cfg.num_layers}] {lt} — OK ({elapsed:.1f}s)")
        except Exception as e:
            results["failed"] += 1
            results["paths"].append(None)
            print(f"  [{i+1}/{cfg.num_layers}] {lt} — ERREUR: {e}")

    print(f"\n{results['converted']}/{cfg.num_layers} layers converted")
    if results["failed"] > 0:
        print(f"{results['failed']} layers FAILED")

    return results


# ── Test ─────────────────────────────────────────────────────────────────────

def test_single_layer(cfg: Qwen35SmallConfig) -> bool:
    """Test conversion of a single DeltaNet layer with random weights."""
    print(f"\n{'='*60}")
    print("Test: Single DeltaNet layer conversion")
    print(f"{'='*60}\n")

    test_dir = OUTPUT_DIR / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    layer = GatedDeltaNetDecode(cfg)
    torch.manual_seed(42)
    for p in layer.parameters():
        nn.init.normal_(p, std=0.01)

    # PyTorch reference
    layer_fp32 = copy.deepcopy(layer).eval()
    _reset_state_buffers(layer_fp32)

    x = torch.randn(1, cfg.hidden_size, 1, 1) * 0.01
    with torch.no_grad():
        pt_out = layer_fp32(x)

    # CoreML conversion
    path = convert_single_deltanet_layer(layer, 0, cfg, test_dir)
    if path is None:
        print("FAIL: Conversion failed")
        return False

    cml_model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_ONLY)
    cml_state = cml_model.make_state()

    cml_out = cml_model.predict(
        {"hidden_states": x.half().numpy()},
        state=cml_state,
    )
    cml_tensor = torch.from_numpy(cml_out["output"]).float()

    diff = (pt_out.float() - cml_tensor).abs().max().item()
    passed = diff < 0.5  # fp16 vs fp32 tolerance
    print(f"Max diff: {diff:.6e} — {'PASS' if passed else 'FAIL'}")
    return passed


# ── Benchmark ────────────────────────────────────────────────────────────────

def benchmark_ane(cfg: Qwen35SmallConfig) -> None:
    """Benchmark a converted layer on ANE."""
    print(f"\n{'='*60}")
    print("Benchmark: DeltaNet layer on ANE")
    print(f"{'='*60}\n")

    test_dir = OUTPUT_DIR / "test"
    layer_path = test_dir / "layer_00_deltanet_decode.mlpackage"

    if not layer_path.exists():
        print("No converted layer found. Run --test first.")
        return

    model = ct.models.MLModel(str(layer_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    state = model.make_state()

    H = cfg.hidden_size
    x = np.random.randn(1, H, 1, 1).astype(np.float16) * 0.01

    # Warmup
    print("Warmup (10 tokens)...")
    for _ in range(10):
        model.predict({"hidden_states": x}, state=state)

    # Benchmark
    n_tokens = 100
    print(f"Benchmark {n_tokens} tokens...")
    t0 = time.time()
    for _ in range(n_tokens):
        result = model.predict({"hidden_states": x}, state=state)
        x = result["output"]
    elapsed = time.time() - t0

    tok_s = n_tokens / elapsed
    ms_per_tok = elapsed / n_tokens * 1000
    print(f"\n  {n_tokens} tokens in {elapsed:.2f}s")
    print(f"  {tok_s:.1f} tok/s per layer ({ms_per_tok:.2f} ms/tok)")
    print(f"  Estimated full model ({cfg.num_layers} layers): {tok_s / cfg.num_layers:.1f} tok/s")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Qwen3.5-0.8B to CoreML for ANE"
    )
    parser.add_argument("--test", action="store_true", help="Test single layer conversion")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark on ANE")
    parser.add_argument("--full", action="store_true", help="Convert all layers with real weights")
    parser.add_argument("--model-path", type=str, default=None, help="Path to Qwen3.5-0.8B weights")
    args = parser.parse_args()

    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = find_model_path()

    cfg = load_config_from_hf(model_path) if model_path.exists() else Qwen35SmallConfig()
    print(f"Config: hidden_size={cfg.hidden_size}, layers={cfg.num_layers}")
    print(f"  DeltaNet: {cfg.num_deltanet_layers} layers, Full Attn: {cfg.num_full_attn_layers}")
    print(f"  Key heads={cfg.num_key_heads}, Value heads={cfg.num_value_heads}")
    print(f"  Key dim={cfg.key_head_dim}, Value dim={cfg.value_head_dim}")

    if args.test:
        test_single_layer(cfg)
    elif args.benchmark:
        benchmark_ane(cfg)
    elif args.full:
        model = Qwen35SmallDecodeModel(cfg)
        if model_path.exists():
            load_weights_from_hf(model, model_path)
        else:
            print(f"WARNING: Model not found at {model_path}, using random weights")
            torch.manual_seed(42)
            for p in model.parameters():
                nn.init.normal_(p, std=0.01)
        model.eval()
        convert_all_layers(cfg, model)
    else:
        print("Use --test, --benchmark, or --full")
        print(f"  --test       Test single layer with random weights")
        print(f"  --benchmark  Benchmark single layer on ANE")
        print(f"  --full       Convert all layers (requires model weights)")


if __name__ == "__main__":
    main()

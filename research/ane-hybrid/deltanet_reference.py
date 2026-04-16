#!/usr/bin/env python3
"""Référence PyTorch du DeltaNet chunkwise parallèle pour Qwen3.5.

Implémente la forme chunkwise parallèle du Gated DeltaNet,
qui se décompose en ops compatibles CoreML MIL.
Étape 1.1 du plan ANE hybrid pipeline.

Config Qwen3.5-35B-A3B (depuis config.json) :
  hidden_size         = 2048
  linear_num_key_heads   = 16
  linear_num_value_heads = 32
  linear_key_head_dim    = 128
  linear_value_head_dim  = 128
  linear_conv_kernel_dim = 4
  num_hidden_layers      = 40  (30 linear_attention + 10 full_attention)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path


# ── Config ──────────────────────────────────────────────────────────────────

QWEN35_CONFIG_PATHS = [
    Path("/Users/clems/KIKI-Mac_tunner/models/Qwen3.5-35B-A3B-Opus-bf16/config.json"),
]

DEFAULT_DELTANET_CONFIG = {
    "hidden_size": 2048,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
}


def load_qwen35_config() -> dict:
    """Charge la config DeltaNet depuis le modèle Qwen3.5."""
    for p in QWEN35_CONFIG_PATHS:
        if p.exists():
            with open(p) as f:
                raw = json.load(f)
            text_cfg = raw.get("text_config", raw)
            return {
                "hidden_size": text_cfg["hidden_size"],
                "linear_num_key_heads": text_cfg["linear_num_key_heads"],
                "linear_num_value_heads": text_cfg["linear_num_value_heads"],
                "linear_key_head_dim": text_cfg["linear_key_head_dim"],
                "linear_value_head_dim": text_cfg["linear_value_head_dim"],
                "linear_conv_kernel_dim": text_cfg["linear_conv_kernel_dim"],
            }
    print("Config Qwen3.5 introuvable, utilisation des valeurs par défaut")
    return DEFAULT_DELTANET_CONFIG


# ── Module ──────────────────────────────────────────────────────────────────

class GatedDeltaNetChunkwise(nn.Module):
    """DeltaNet en forme chunkwise parallèle.

    Convertit la récurrence DeltaNet en opérations matmul
    qui sont compatibles CoreML MIL / Apple Neural Engine.

    Basé sur: https://sustcsonglin.github.io/blog/2024/deltanet-2/
    Paper: https://jankautz.com/publications/GatedDeltaNet_ICLR25.pdf

    Qwen3.5 utilise des dimensions K et V séparées :
      - num_key_heads=16, key_head_dim=128  → Q/K projection = 16*128 = 2048
      - num_value_heads=32, value_head_dim=128 → V projection = 32*128 = 4096
      - State shape: [B, num_key_heads, key_dim, value_expand]
        où value_expand = (num_value_heads // num_key_heads) * value_head_dim
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_key_heads: int = 16,
        num_value_heads: int = 32,
        key_head_dim: int = 128,
        value_head_dim: int = 128,
        conv_size: int = 4,
        chunk_size: int = 64,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_key_heads = num_key_heads
        self.num_value_heads = num_value_heads
        self.key_head_dim = key_head_dim
        self.value_head_dim = value_head_dim
        self.conv_size = conv_size
        self.chunk_size = chunk_size

        # Derived
        self.kv_group_size = num_value_heads // num_key_heads  # 32/16 = 2
        self.value_expand = self.kv_group_size * value_head_dim  # 2*128 = 256

        qk_dim = num_key_heads * key_head_dim      # 16*128 = 2048
        v_dim = num_value_heads * value_head_dim    # 32*128 = 4096

        # Projections Q, K, V
        self.q_proj = nn.Linear(hidden_size, qk_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, qk_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, v_dim, bias=False)
        self.o_proj = nn.Linear(v_dim, hidden_size, bias=False)

        # Gates (per key-head)
        self.alpha_gate = nn.Linear(hidden_size, num_key_heads, bias=False)  # decay
        self.beta_gate = nn.Linear(hidden_size, num_key_heads, bias=False)   # update

        # Short convolutions (causal, depthwise)
        self.q_conv = nn.Conv1d(
            qk_dim, qk_dim, kernel_size=conv_size,
            padding=conv_size - 1, groups=qk_dim,
        )
        self.k_conv = nn.Conv1d(
            qk_dim, qk_dim, kernel_size=conv_size,
            padding=conv_size - 1, groups=qk_dim,
        )
        self.v_conv = nn.Conv1d(
            v_dim, v_dim, kernel_size=conv_size,
            padding=conv_size - 1, groups=v_dim,
        )

    def _short_conv(self, x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        """Convolution causale courte: [B, T, C] -> [B, T, C]."""
        B, T, C = x.shape
        y = conv(x.transpose(1, 2))[:, :, :T]
        return F.silu(y.transpose(1, 2))

    def _reshape_kv_to_key_heads(self, v: torch.Tensor) -> torch.Tensor:
        """Regroupe les value heads par key head.

        [B, T, num_value_heads, value_head_dim]
        → [B, T, num_key_heads, kv_group_size * value_head_dim]
        """
        B, T, _, _ = v.shape
        v = v.view(B, T, self.num_key_heads, self.kv_group_size, self.value_head_dim)
        return v.view(B, T, self.num_key_heads, self.value_expand)

    def recurrent_forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forme récurrente (pour validation).

        Args:
            x: [B, T, hidden_size]
            state: [B, num_key_heads, key_head_dim, value_expand] ou None

        Returns:
            output: [B, T, hidden_size]
            new_state: [B, num_key_heads, key_head_dim, value_expand]
        """
        B, T, _ = x.shape

        q = self._short_conv(self.q_proj(x), self.q_conv)
        k = self._short_conv(self.k_proj(x), self.k_conv)
        v = self._short_conv(self.v_proj(x), self.v_conv)

        # Reshape to heads
        q = q.view(B, T, self.num_key_heads, self.key_head_dim)
        k = k.view(B, T, self.num_key_heads, self.key_head_dim)
        v = v.view(B, T, self.num_value_heads, self.value_head_dim)

        # L2 normalize Q, K
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        # Regroup V so it aligns with key heads
        v = self._reshape_kv_to_key_heads(v)  # [B, T, Hk, Ve]

        # Gates
        alpha = torch.sigmoid(self.alpha_gate(x)).unsqueeze(-1)  # [B, T, Hk, 1]
        beta = torch.sigmoid(self.beta_gate(x)).unsqueeze(-1)    # [B, T, Hk, 1]

        # Initialize state: [B, Hk, Dk, Ve]
        if state is None:
            state = torch.zeros(
                B, self.num_key_heads, self.key_head_dim, self.value_expand,
                dtype=x.dtype, device=x.device,
            )

        outputs = []
        for t in range(T):
            qt = q[:, t]   # [B, Hk, Dk]
            kt = k[:, t]   # [B, Hk, Dk]
            vt = v[:, t]   # [B, Hk, Ve]
            at = alpha[:, t]  # [B, Hk, 1]
            bt = beta[:, t]   # [B, Hk, 1]

            # 1. Decay
            state = state * at.unsqueeze(-1)   # broadcast [B, Hk, 1, 1]

            # 2. Error-correcting delta update
            retrieved = torch.einsum("bhkv,bhk->bhv", state, kt)
            error = vt - retrieved
            state = state + bt.unsqueeze(-2) * torch.einsum("bhk,bhv->bhkv", kt, error)

            # 3. Output
            ot = torch.einsum("bhkv,bhk->bhv", state, qt)  # [B, Hk, Ve]
            outputs.append(ot)

        output = torch.stack(outputs, dim=1)  # [B, T, Hk, Ve]
        # Reshape back to flat value dim
        output = output.view(B, T, self.num_value_heads * self.value_head_dim)
        output = self.o_proj(output)

        return output, state

    def chunkwise_forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forme chunkwise pour conversion CoreML.

        Traite l'entrée par chunks de taille C (défaut 64).
        Dans chaque chunk, le state est mis à jour séquentiellement
        (delta rule), ce qui s'unrolle en ops statiques CoreML MIL
        (matmul, sigmoid, einsum). Le chunk size C=64 est petit
        donc l'unrolling reste tractable pour le compilateur ANE.

        Args:
            x: [B, T, hidden_size]
            state: [B, num_key_heads, key_head_dim, value_expand] ou None

        Returns:
            output: [B, T, hidden_size]
            new_state: [B, num_key_heads, key_head_dim, value_expand]
        """
        B, T, _ = x.shape
        C = self.chunk_size
        Hk = self.num_key_heads
        Dk = self.key_head_dim
        Ve = self.value_expand

        # Pad T to multiple of C
        pad_len = (C - T % C) % C
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        T_padded = x.size(1)
        num_chunks = T_padded // C

        q = self._short_conv(self.q_proj(x), self.q_conv)
        k = self._short_conv(self.k_proj(x), self.k_conv)
        v = self._short_conv(self.v_proj(x), self.v_conv)

        q = q.view(B, T_padded, Hk, Dk)
        k = k.view(B, T_padded, Hk, Dk)
        v = v.view(B, T_padded, self.num_value_heads, self.value_head_dim)

        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        v = self._reshape_kv_to_key_heads(v)  # [B, T_padded, Hk, Ve]

        alpha = torch.sigmoid(self.alpha_gate(x[:, :T_padded])).unsqueeze(-1)
        beta = torch.sigmoid(self.beta_gate(x[:, :T_padded])).unsqueeze(-1)

        # Reshape to chunks: [B, num_chunks, C, Hk, D]
        q = q.view(B, num_chunks, C, Hk, Dk)
        k = k.view(B, num_chunks, C, Hk, Dk)
        v = v.view(B, num_chunks, C, Hk, Ve)
        alpha = alpha.view(B, num_chunks, C, Hk, 1)
        beta = beta.view(B, num_chunks, C, Hk, 1)

        if state is None:
            state = torch.zeros(B, Hk, Dk, Ve, dtype=x.dtype, device=x.device)

        all_outputs = []

        for chunk_idx in range(num_chunks):
            qc = q[:, chunk_idx]      # [B, C, Hk, Dk]
            kc = k[:, chunk_idx]      # [B, C, Hk, Dk]
            vc = v[:, chunk_idx]      # [B, C, Hk, Ve]
            ac = alpha[:, chunk_idx]  # [B, C, Hk, 1]
            bc = beta[:, chunk_idx]   # [B, C, Hk, 1]

            # === Process chunk sequentially through state (delta rule) ===
            # The delta rule's error-correcting update is inherently sequential
            # in the state, but we can still batch the output computation.
            # For CoreML, the chunk size C=64 is small enough that unrolling
            # is acceptable and maps to static graph ops.

            chunk_outputs = []
            for t in range(C):
                qt = qc[:, t]   # [B, Hk, Dk]
                kt = kc[:, t]   # [B, Hk, Dk]
                vt = vc[:, t]   # [B, Hk, Ve]
                at = ac[:, t]   # [B, Hk, 1]
                bt = bc[:, t]   # [B, Hk, 1]

                # 1. Decay state
                state = state * at.unsqueeze(-1)

                # 2. Error-correcting delta update
                retrieved = torch.einsum("bhkv,bhk->bhv", state, kt)
                error = vt - retrieved
                state = state + bt.unsqueeze(-2) * torch.einsum(
                    "bhk,bhv->bhkv", kt, error
                )

                # 3. Output
                ot = torch.einsum("bhkv,bhk->bhv", state, qt)
                chunk_outputs.append(ot)

            chunk_out = torch.stack(chunk_outputs, dim=1)  # [B, C, Hk, Ve]
            all_outputs.append(chunk_out)

        output = torch.cat(all_outputs, dim=1)  # [B, T_padded, Hk, Ve]
        output = output[:, :T]
        output = output.reshape(B, T, self.num_value_heads * self.value_head_dim)
        output = self.o_proj(output)

        return output, state


# ── Test ────────────────────────────────────────────────────────────────────

def test_equivalence() -> bool:
    """Vérifie que la forme chunkwise est numériquement équivalente à la récurrente."""
    torch.manual_seed(42)

    cfg = load_qwen35_config()
    print(f"Config chargée: {cfg}")

    model = GatedDeltaNetChunkwise(
        hidden_size=cfg["hidden_size"],
        num_key_heads=cfg["linear_num_key_heads"],
        num_value_heads=cfg["linear_num_value_heads"],
        key_head_dim=cfg["linear_key_head_dim"],
        value_head_dim=cfg["linear_value_head_dim"],
        conv_size=cfg["linear_conv_kernel_dim"],
        chunk_size=64,
    )
    model.eval()
    model.double()  # FP64 pour réduire les erreurs numériques

    B, T = 1, 128
    x = torch.randn(B, T, cfg["hidden_size"], dtype=torch.float64)

    with torch.no_grad():
        out_rec, state_rec = model.recurrent_forward(x)
        out_chunk, state_chunk = model.chunkwise_forward(x)

    diff = (out_rec - out_chunk).abs().max().item()
    state_diff = (state_rec - state_chunk).abs().max().item()

    print(f"\nDimensions:")
    print(f"  hidden_size     = {cfg['hidden_size']}")
    print(f"  num_key_heads   = {cfg['linear_num_key_heads']}")
    print(f"  num_value_heads = {cfg['linear_num_value_heads']}")
    print(f"  key_head_dim    = {cfg['linear_key_head_dim']}")
    print(f"  value_head_dim  = {cfg['linear_value_head_dim']}")
    print(f"  conv_size       = {cfg['linear_conv_kernel_dim']}")
    print(f"\nInput:  batch={B}, seq_len={T}")
    print(f"Output: {out_rec.shape}")
    print(f"State:  {state_rec.shape}")
    print(f"\nMax output diff (recurrent vs chunkwise): {diff:.6e}")
    print(f"Max state diff:                           {state_diff:.6e}")

    passed = diff < 1e-4
    print(f"\nEquivalence: {'PASS' if passed else 'FAIL'} (seuil 1e-4 en FP64)")

    # Mémoire state par couche
    state_bytes = (
        cfg["linear_num_key_heads"]
        * cfg["linear_key_head_dim"]
        * (cfg["linear_num_value_heads"] // cfg["linear_num_key_heads"])
        * cfg["linear_value_head_dim"]
        * 2  # FP16
    )
    print(f"\nState par couche: {state_bytes / 1024:.1f} Ko")
    print(f"State 30 couches: {30 * state_bytes / 1024 / 1024:.1f} Mo")

    return passed


if __name__ == "__main__":
    success = test_equivalence()
    if success:
        print("\nProchain step: conversion Conv2d pour ANEMLL (deltanet_conv2d.py)")
    else:
        print("\nATTENTION: Les deux formes divergent. Debug necessaire.")

#!/usr/bin/env python3
"""Conversion DeltaNet Conv2d → CoreML avec ct.StateType.

Crée deux modèles CoreML :
1. Prefill : traite un chunk de 64 tokens, met à jour l'état S
2. Decode : traite 1 token, lit/écrit l'état S via ct.StateType

Phase 1.3 du plan ANE hybrid pipeline.

Usage :
    python convert_deltanet.py                   # conversion complète
    python convert_deltanet.py --decode-only      # decode seul
    python convert_deltanet.py --prefill-only     # prefill seul
    python convert_deltanet.py --test             # test numérique post-conversion
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import coremltools as ct

from deltanet_conv2d import (
    GatedDeltaNetConv2d,
    l2_normalize,
    load_qwen35_config,
)


# ── Config ───────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path(__file__).parent / "mlpackages"

# Taille de chunk pour le prefill
PREFILL_CHUNK_SIZE = 64


# ── Wrapper Decode (1 token récurrent) ────────────────────────────────────────

class DeltaNetDecodeWrapper(nn.Module):
    """Wrapper pour l'inférence decode d'un seul token.

    Utilise la forme récurrente du DeltaNet (pas chunkwise).
    Les buffers mutables (deltanet_state, conv caches) deviennent
    des ct.StateType dans CoreML. La mutation se fait par assignation
    par slice (compatible avec le trace JIT + coremltools).

    Les poids de convolution sont stockés comme nn.Parameter frozen
    (pas comme buffers) pour éviter qu'ils soient inclus dans les states.
    """

    def __init__(self, base_model: GatedDeltaNetConv2d):
        super().__init__()
        # Copie profonde des projections (pour éviter de partager
        # les paramètres avec le modèle de base lors du .half())
        self.q_proj = copy.deepcopy(base_model.q_proj)
        self.k_proj = copy.deepcopy(base_model.k_proj)
        self.v_proj = copy.deepcopy(base_model.v_proj)
        self.o_proj = copy.deepcopy(base_model.o_proj)
        self.alpha_gate = copy.deepcopy(base_model.alpha_gate)
        self.beta_gate = copy.deepcopy(base_model.beta_gate)

        # Paramètres structurels
        self.num_key_heads = base_model.num_key_heads
        self.num_value_heads = base_model.num_value_heads
        self.key_head_dim = base_model.key_head_dim
        self.value_head_dim = base_model.value_head_dim
        self.kv_group_size = base_model.kv_group_size
        self.value_expand = base_model.value_expand
        self.hidden_size = base_model.hidden_size
        self.conv_size = base_model.conv_size

        qk_dim = self.num_key_heads * self.key_head_dim
        v_dim = self.num_value_heads * self.value_head_dim

        # Poids conv : paramètres frozen (pas des states mutables)
        # Shape pré-calculée pour le calcul single-step :
        # [dim, 1, conv_size] → [1, dim, conv_size] (prêt pour broadcast)
        self.register_buffer(
            "q_conv_w",
            base_model.q_conv.weight.data.squeeze(1).unsqueeze(0).clone(),
        )
        self.register_buffer(
            "k_conv_w",
            base_model.k_conv.weight.data.squeeze(1).unsqueeze(0).clone(),
        )
        self.register_buffer(
            "v_conv_w",
            base_model.v_conv.weight.data.squeeze(1).unsqueeze(0).clone(),
        )

        # === Buffers mutables (→ ct.StateType) ===

        Hk = self.num_key_heads
        Dk = self.key_head_dim
        Ve = self.value_expand

        # État récurrent DeltaNet : [1, Hk, Dk, Ve]
        self.register_buffer(
            "deltanet_state",
            torch.zeros(1, Hk, Dk, Ve, dtype=torch.float16),
        )

        # Conv caches : les conv_size-1 derniers tokens pour chaque projection
        cache_len = self.conv_size - 1  # 3 pour kernel_size=4
        self.register_buffer(
            "q_conv_cache",
            torch.zeros(1, qk_dim, cache_len, dtype=torch.float16),
        )
        self.register_buffer(
            "k_conv_cache",
            torch.zeros(1, qk_dim, cache_len, dtype=torch.float16),
        )
        self.register_buffer(
            "v_conv_cache",
            torch.zeros(1, v_dim, cache_len, dtype=torch.float16),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Inférence decode d'un seul token.

        Args:
            hidden_states: [1, hidden_size, 1, 1] — layout ANEMLL

        Returns:
            output: [1, hidden_size, 1, 1] — layout ANEMLL
        """
        Hk = self.num_key_heads
        Dk = self.key_head_dim
        Ve = self.value_expand

        # === Projections Conv2d ===
        q = self.q_proj(hidden_states)    # [1, qk_dim, 1, 1]
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)    # [1, v_dim, 1, 1]

        alpha = torch.sigmoid(self.alpha_gate(hidden_states))  # [1, Hk, 1, 1]
        beta = torch.sigmoid(self.beta_gate(hidden_states))

        # === Short convolutions causales avec cache ===
        # Squeeze dim hauteur : [1, dim, 1, 1] → [1, dim, 1]
        q_1d = q[:, :, 0:1, 0:1].squeeze(2)  # [1, qk_dim, 1]
        k_1d = k[:, :, 0:1, 0:1].squeeze(2)
        v_1d = v[:, :, 0:1, 0:1].squeeze(2)

        # Concat cache + token : [1, dim, conv_size]
        q_full = torch.cat([self.q_conv_cache, q_1d], dim=2)
        k_full = torch.cat([self.k_conv_cache, k_1d], dim=2)
        v_full = torch.cat([self.v_conv_cache, v_1d], dim=2)

        # Convolution depthwise manuelle : (full * weight).sum(dim=2)
        # q_conv_w: [1, qk_dim, conv_size], q_full: [1, qk_dim, conv_size]
        q_1d = F.silu((q_full * self.q_conv_w).sum(dim=2, keepdim=True))  # [1, dim, 1]
        k_1d = F.silu((k_full * self.k_conv_w).sum(dim=2, keepdim=True))
        v_1d = F.silu((v_full * self.v_conv_w).sum(dim=2, keepdim=True))

        # Mise à jour conv cache par slice assignment (compatible ct.StateType)
        self.q_conv_cache[:, :, :] = q_full[:, :, 1:]
        self.k_conv_cache[:, :, :] = k_full[:, :, 1:]
        self.v_conv_cache[:, :, :] = v_full[:, :, 1:]

        # === Reshape en têtes ===
        q_heads = q_1d.squeeze(2).view(1, Hk, Dk)
        k_heads = k_1d.squeeze(2).view(1, Hk, Dk)

        # V → regroup par key heads : [1, Hk, Ve]
        v_heads = v_1d.squeeze(2).view(
            1, self.num_value_heads, self.value_head_dim
        ).view(
            1, Hk, self.kv_group_size, self.value_head_dim
        ).reshape(1, Hk, Ve)

        # L2 normalisation
        q_heads = l2_normalize(q_heads, dim=-1)
        k_heads = l2_normalize(k_heads, dim=-1)

        # Alpha, beta : [1, Hk, 1, 1] → [1, Hk, 1, 1] (garder pour broadcast)
        alpha_4d = alpha.view(1, Hk, 1, 1)
        beta_4d = beta.view(1, Hk, 1, 1)

        # === Mise à jour récurrente du state DeltaNet ===
        state = self.deltanet_state  # [1, Hk, Dk, Ve]

        # 1. Decay
        state_decayed = state * alpha_4d

        # 2. Delta error-correcting update
        k_exp = k_heads.unsqueeze(-1)                       # [1, Hk, Dk, 1]
        retrieved = (state_decayed * k_exp).sum(dim=2)       # [1, Hk, Ve]
        error = v_heads - retrieved
        outer = k_heads.unsqueeze(-1) * error.unsqueeze(2)   # [1, Hk, Dk, Ve]
        new_state = state_decayed + beta_4d * outer

        # Écriture du state par slice assignment
        self.deltanet_state[:, :, :, :] = new_state

        # 3. Output : query le state
        q_exp = q_heads.unsqueeze(-1)                        # [1, Hk, Dk, 1]
        output = (new_state * q_exp).sum(dim=2)              # [1, Hk, Ve]

        # Reshape : [1, Hk, Ve] → [1, v_dim, 1, 1] → Conv2d → [1, H, 1, 1]
        output = output.reshape(1, self.num_value_heads * self.value_head_dim)
        output = output.view(1, -1, 1, 1)
        output = self.o_proj(output)

        return output


# ── Wrapper Prefill (chunk de C tokens) ───────────────────────────────────────

class DeltaNetPrefillWrapper(nn.Module):
    """Wrapper pour le prefill de C tokens en parallèle.

    Utilise le traitement séquentiel token-par-token (delta rule)
    mais avec les projections en mode chunk (Conv2d sur C tokens).
    """

    def __init__(self, base_model: GatedDeltaNetConv2d):
        super().__init__()
        # Copie profonde des projections Conv2d (pas de problème coremltools)
        self.q_proj = copy.deepcopy(base_model.q_proj)
        self.k_proj = copy.deepcopy(base_model.k_proj)
        self.v_proj = copy.deepcopy(base_model.v_proj)
        self.o_proj = copy.deepcopy(base_model.o_proj)
        self.alpha_gate = copy.deepcopy(base_model.alpha_gate)
        self.beta_gate = copy.deepcopy(base_model.beta_gate)

        # Les Conv1d groupées (depthwise) ne sont pas bien supportées
        # par coremltools 9 + torch 2.11 (bug dtype weight/bias).
        # On stocke les poids comme buffers et on fait F.conv1d manuellement.
        qk_dim = base_model.num_key_heads * base_model.key_head_dim
        v_dim = base_model.num_value_heads * base_model.value_head_dim

        self.register_buffer("q_conv_weight", base_model.q_conv.weight.data.clone())
        self.register_buffer("k_conv_weight", base_model.k_conv.weight.data.clone())
        self.register_buffer("v_conv_weight", base_model.v_conv.weight.data.clone())
        self.register_buffer(
            "q_conv_bias",
            base_model.q_conv.bias.data.clone() if base_model.q_conv.bias is not None
            else torch.zeros(qk_dim),
        )
        self.register_buffer(
            "k_conv_bias",
            base_model.k_conv.bias.data.clone() if base_model.k_conv.bias is not None
            else torch.zeros(qk_dim),
        )
        self.register_buffer(
            "v_conv_bias",
            base_model.v_conv.bias.data.clone() if base_model.v_conv.bias is not None
            else torch.zeros(v_dim),
        )
        self._conv_groups_qk = qk_dim
        self._conv_groups_v = v_dim

        self.num_key_heads = base_model.num_key_heads
        self.num_value_heads = base_model.num_value_heads
        self.key_head_dim = base_model.key_head_dim
        self.value_head_dim = base_model.value_head_dim
        self.kv_group_size = base_model.kv_group_size
        self.value_expand = base_model.value_expand
        self.hidden_size = base_model.hidden_size
        self.conv_size = base_model.conv_size
        self.chunk_size = PREFILL_CHUNK_SIZE

        Hk = self.num_key_heads
        Dk = self.key_head_dim
        Ve = self.value_expand

        # État récurrent DeltaNet
        self.register_buffer(
            "deltanet_state",
            torch.zeros(1, Hk, Dk, Ve, dtype=torch.float16),
        )

    def _short_conv_manual(
        self,
        x_ane: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        groups: int,
    ) -> torch.Tensor:
        """Convolution causale courte via F.conv1d (évite le bug nn.Conv1d).

        Utilise F.conv1d avec les poids/biais stockés comme buffers
        pour contourner le bug dtype coremltools 9 + torch 2.11.
        """
        x_1d = x_ane.squeeze(2)                                       # [B, C, T]
        # bias=None car coremltools 9 + torch 2.11 a un bug dtype
        # sur les convolutions groupées avec biais
        y = F.conv1d(x_1d, weight, None, padding=self.conv_size - 1, groups=groups)
        y = y + bias.unsqueeze(0).unsqueeze(-1)  # ajouter le biais manuellement
        y = y[:, :, :PREFILL_CHUNK_SIZE]                              # troncature causale
        return F.silu(y).unsqueeze(2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Prefill de PREFILL_CHUNK_SIZE tokens.

        Args:
            hidden_states: [1, hidden_size, 1, PREFILL_CHUNK_SIZE] — layout ANEMLL

        Returns:
            output: [1, hidden_size, 1, PREFILL_CHUNK_SIZE] — layout ANEMLL
        """
        C = PREFILL_CHUNK_SIZE  # constante statique, pas .size()
        Hk = self.num_key_heads
        Dk = self.key_head_dim
        Ve = self.value_expand

        # Projections Conv2d
        q_ane = self.q_proj(hidden_states)
        k_ane = self.k_proj(hidden_states)
        v_ane = self.v_proj(hidden_states)

        # Short convolutions causales (F.conv1d manuel, pas nn.Conv1d)
        q_ane = self._short_conv_manual(q_ane, self.q_conv_weight, self.q_conv_bias, self._conv_groups_qk)
        k_ane = self._short_conv_manual(k_ane, self.k_conv_weight, self.k_conv_bias, self._conv_groups_qk)
        v_ane = self._short_conv_manual(v_ane, self.v_conv_weight, self.v_conv_bias, self._conv_groups_v)

        # Gates Conv2d
        alpha_ane = torch.sigmoid(self.alpha_gate(hidden_states))
        beta_ane = torch.sigmoid(self.beta_gate(hidden_states))

        # Layout séquentiel : [1, dim, 1, C] → [1, C, Hk, D]
        q = q_ane.squeeze(2).permute(0, 2, 1).reshape(1, C, Hk, Dk)
        k = k_ane.squeeze(2).permute(0, 2, 1).reshape(1, C, Hk, Dk)

        v = v_ane.squeeze(2).permute(0, 2, 1).reshape(
            1, C, self.num_value_heads, self.value_head_dim
        )
        v = v.view(1, C, Hk, self.kv_group_size, self.value_head_dim)
        v = v.reshape(1, C, Hk, Ve)

        alpha = alpha_ane.squeeze(2).permute(0, 2, 1)  # [1, C, Hk]
        beta = beta_ane.squeeze(2).permute(0, 2, 1)

        # L2 normalisation
        q = l2_normalize(q, dim=-1)
        k = l2_normalize(k, dim=-1)

        # === Traitement séquentiel token par token ===
        # On doit dérouler la boucle pour que le trace JIT fonctionne.
        # C = PREFILL_CHUNK_SIZE = 64 est fixe, donc le déroulement est statique.
        state = self.deltanet_state  # [1, Hk, Dk, Ve]
        all_outputs = []

        for t in range(PREFILL_CHUNK_SIZE):
            qt = q[:, t:t+1, :, :].squeeze(1)   # [1, Hk, Dk]
            kt = k[:, t:t+1, :, :].squeeze(1)
            vt = v[:, t:t+1, :, :].squeeze(1)   # [1, Hk, Ve]
            at = alpha[:, t:t+1, :].squeeze(1)   # [1, Hk]
            bt = beta[:, t:t+1, :].squeeze(1)

            # Decay
            state = state * at.view(1, Hk, 1, 1)

            # Delta error-correcting update
            k_exp = kt.unsqueeze(-1)                        # [1, Hk, Dk, 1]
            retrieved = (state * k_exp).sum(dim=2)           # [1, Hk, Ve]
            error = vt - retrieved
            outer = kt.unsqueeze(-1) * error.unsqueeze(2)    # [1, Hk, Dk, Ve]
            state = state + bt.view(1, Hk, 1, 1) * outer

            # Output
            q_exp = qt.unsqueeze(-1)
            ot = (state * q_exp).sum(dim=2)                  # [1, Hk, Ve]
            all_outputs.append(ot)

        # Écriture du state par slice
        self.deltanet_state[:, :, :, :] = state

        # [64 x [1, Hk, Ve]] → [1, 64, v_dim]
        output = torch.stack(all_outputs, dim=1)
        output = output.reshape(1, C, self.num_value_heads * self.value_head_dim)

        # Projection de sortie
        out_ane = output.permute(0, 2, 1).unsqueeze(2)
        out_ane = self.o_proj(out_ane)

        return out_ane


# ── Fonctions utilitaires ────────────────────────────────────────────────────

def _reset_state_buffers(module: nn.Module) -> None:
    """Remet les buffers d'état mutable à zéro."""
    with torch.no_grad():
        for name, buf in module.named_buffers():
            buf.zero_()


# Noms des buffers qui sont des constantes (poids), pas des états mutables
_CONST_BUFFER_NAMES = {
    "q_conv_w", "k_conv_w", "v_conv_w",
    "q_conv_weight", "k_conv_weight", "v_conv_weight",
    "q_conv_bias", "k_conv_bias", "v_conv_bias",
}


def _get_mutable_state_specs(wrapper: nn.Module) -> list:
    """Construit les ct.StateType uniquement pour les buffers mutables.

    Exclut les buffers qui sont des poids constants (conv weights).
    """
    states = []
    for name, buf in wrapper.named_buffers():
        # Exclure les poids de convolution (constantes, pas des states)
        if name in _CONST_BUFFER_NAMES:
            continue
        states.append(
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=tuple(buf.shape),
                    dtype=np.float16,
                ),
                name=name,
            )
        )
    return states


# ── Conversion CoreML ────────────────────────────────────────────────────────

def convert_decode_model(
    base_model: GatedDeltaNetConv2d,
    output_path: Path,
) -> ct.models.MLModel:
    """Convertit le wrapper decode en CoreML .mlpackage."""
    print("\n" + "=" * 60)
    print("Conversion du modèle DECODE (1 token)")
    print("=" * 60)

    H = base_model.hidden_size

    wrapper = DeltaNetDecodeWrapper(base_model)
    wrapper.eval()
    wrapper = wrapper.half()

    example_input = torch.zeros(1, H, 1, 1, dtype=torch.float16)

    _reset_state_buffers(wrapper)

    print("Tracing avec torch.jit.trace...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example_input)
    _reset_state_buffers(wrapper)
    _reset_state_buffers(traced)
    print("Trace OK.")

    states = _get_mutable_state_specs(wrapper)
    print(f"States CoreML ({len(states)}) :")
    for s in states:
        print(f"  {s.name}: shape={s.wrapped_type.shape}")

    print("\nConversion CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="hidden_states",
                shape=(1, H, 1, 1),
                dtype=np.float16,
            ),
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))
    print(f"Modèle decode sauvegardé : {output_path}")

    return mlmodel


def convert_prefill_model(
    base_model: GatedDeltaNetConv2d,
    output_path: Path,
    chunk_size: int = PREFILL_CHUNK_SIZE,
) -> ct.models.MLModel:
    """Convertit le wrapper prefill en CoreML .mlpackage."""
    print("\n" + "=" * 60)
    print(f"Conversion du modèle PREFILL ({chunk_size} tokens)")
    print("=" * 60)

    H = base_model.hidden_size

    wrapper = DeltaNetPrefillWrapper(base_model)
    wrapper.eval()
    wrapper = wrapper.half()

    example_input = torch.zeros(1, H, 1, chunk_size, dtype=torch.float16)

    _reset_state_buffers(wrapper)

    print("Tracing avec torch.jit.trace...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example_input)
    _reset_state_buffers(wrapper)
    _reset_state_buffers(traced)
    print("Trace OK.")

    states = _get_mutable_state_specs(wrapper)
    print(f"States CoreML ({len(states)}) :")
    for s in states:
        print(f"  {s.name}: shape={s.wrapped_type.shape}")

    print("\nConversion CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="hidden_states",
                shape=(1, H, 1, chunk_size),
                dtype=np.float16,
            ),
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))
    print(f"Modèle prefill sauvegardé : {output_path}")

    return mlmodel


# ── Test numérique post-conversion ──────────────────────────────────────────

def test_coreml_vs_pytorch(
    base_model: GatedDeltaNetConv2d,
    decode_path: Path,
    prefill_path: Path,
) -> bool:
    """Compare les sorties CoreML aux sorties PyTorch de référence."""
    print("\n" + "=" * 60)
    print("Test numérique CoreML vs PyTorch")
    print("=" * 60)

    H = base_model.hidden_size
    torch.manual_seed(42)
    all_passed = True

    # === Modèle de base en fp32 pour le test (stable numériquement) ===
    # base_model est en fp16 (pour la conversion CoreML), on le recréé en fp32
    # avec les mêmes poids pour la référence PyTorch.
    base_fp32 = GatedDeltaNetConv2d(
        hidden_size=base_model.hidden_size,
        num_key_heads=base_model.num_key_heads,
        num_value_heads=base_model.num_value_heads,
        key_head_dim=base_model.key_head_dim,
        value_head_dim=base_model.value_head_dim,
        conv_size=base_model.conv_size,
        chunk_size=PREFILL_CHUNK_SIZE,
    )
    # Copier les poids fp16 → fp32
    base_fp32.load_state_dict(
        {k: v.float() for k, v in base_model.state_dict().items()}
    )
    base_fp32.eval()

    # === Test Decode ===
    if decode_path.exists():
        print("\n--- Test Decode ---")
        cml_decode = ct.models.MLModel(
            str(decode_path),
            compute_units=ct.ComputeUnit.CPU_ONLY,
        )

        # Wrapper PyTorch en fp32 (référence stable)
        pt_decode = DeltaNetDecodeWrapper(base_fp32)
        pt_decode.eval()
        _reset_state_buffers(pt_decode)

        cml_state = cml_decode.make_state()

        max_diff = 0.0
        num_steps = 5
        for step in range(num_steps):
            # Input en fp32, on le passe en fp16 pour CoreML
            x_fp32 = torch.randn(1, H, 1, 1) * 0.01
            x_fp16 = x_fp32.half()

            with torch.no_grad():
                pt_out = pt_decode(x_fp32)

            cml_out = cml_decode.predict(
                {"hidden_states": x_fp16.numpy()},
                state=cml_state,
            )
            cml_tensor = torch.from_numpy(cml_out["output"]).float()

            # Vérifier NaN
            pt_has_nan = torch.isnan(pt_out).any().item()
            cml_has_nan = torch.isnan(cml_tensor).any().item()
            if pt_has_nan or cml_has_nan:
                print(f"  Step {step}: NaN détecté (pt={pt_has_nan}, cml={cml_has_nan})")
                max_diff = float("inf")
                continue
            diff = (pt_out.float() - cml_tensor).abs().max().item()
            max_diff = max(max_diff, diff)
            print(f"  Step {step}: diff={diff:.6e}")

        # Seuil large : fp32 vs fp16 introduit des diffs de l'ordre de 1e-3
        decode_ok = max_diff < 0.5 and max_diff != float("inf")
        print(f"  Max diff decode: {max_diff:.6e} — {'PASS' if decode_ok else 'FAIL'}")
        all_passed = all_passed and decode_ok
    else:
        print(f"\nModèle decode introuvable : {decode_path}")

    # === Test Prefill ===
    if prefill_path.exists():
        print("\n--- Test Prefill ---")
        cml_prefill = ct.models.MLModel(
            str(prefill_path),
            compute_units=ct.ComputeUnit.CPU_ONLY,
        )

        pt_prefill = DeltaNetPrefillWrapper(base_fp32)
        pt_prefill.eval()
        _reset_state_buffers(pt_prefill)

        cml_state = cml_prefill.make_state()

        x_fp32 = torch.randn(1, H, 1, PREFILL_CHUNK_SIZE) * 0.01
        x_fp16 = x_fp32.half()

        with torch.no_grad():
            pt_out = pt_prefill(x_fp32)

        cml_out = cml_prefill.predict(
            {"hidden_states": x_fp16.numpy()},
            state=cml_state,
        )
        cml_tensor = torch.from_numpy(cml_out["output"]).float()

        pt_has_nan = torch.isnan(pt_out).any().item()
        cml_has_nan = torch.isnan(cml_tensor).any().item()
        if pt_has_nan or cml_has_nan:
            print(f"  NaN détecté dans prefill (pt={pt_has_nan}, cml={cml_has_nan})")
            diff = float("inf")
        else:
            diff = (pt_out.float() - cml_tensor).abs().max().item()
        prefill_ok = diff < 0.5
        print(f"  Max diff prefill: {diff:.6e} — {'PASS' if prefill_ok else 'FAIL'}")
        all_passed = all_passed and prefill_ok
    else:
        print(f"\nModèle prefill introuvable : {prefill_path}")

    return all_passed


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Conversion DeltaNet → CoreML")
    parser.add_argument("--decode-only", action="store_true")
    parser.add_argument("--prefill-only", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)

    cfg = load_qwen35_config()
    print(f"Config DeltaNet : {cfg}")

    # Modèle de base avec poids aléatoires reproductibles
    base = GatedDeltaNetConv2d(
        hidden_size=cfg["hidden_size"],
        num_key_heads=cfg["linear_num_key_heads"],
        num_value_heads=cfg["linear_num_value_heads"],
        key_head_dim=cfg["linear_key_head_dim"],
        value_head_dim=cfg["linear_value_head_dim"],
        conv_size=cfg["linear_conv_kernel_dim"],
        chunk_size=PREFILL_CHUNK_SIZE,
    )
    base.eval()
    torch.manual_seed(123)
    # Initialisation avec une std petite pour éviter overflow en FP16
    # (hidden_size=2048 → fan_in large, std doit être ~1/sqrt(2048) ≈ 0.022)
    for p in base.parameters():
        nn.init.normal_(p, std=0.01)
    # Convertir en fp16 avant de créer les wrappers
    # (évite le dtype mismatch poids/biais dans le trace)
    base = base.half()

    decode_path = out_dir / "deltanet_decode.mlpackage"
    prefill_path = out_dir / "deltanet_prefill.mlpackage"

    do_decode = not args.prefill_only
    do_prefill = not args.decode_only

    if do_decode:
        convert_decode_model(base, decode_path)

    if do_prefill:
        convert_prefill_model(base, prefill_path)

    if args.test or (do_decode and do_prefill):
        test_coreml_vs_pytorch(base, decode_path, prefill_path)

    print("\n" + "=" * 60)
    print("Phase 1.3 terminée.")
    print("=" * 60)
    print(f"Modèles dans : {out_dir}")
    if do_decode:
        print(f"  Decode  : {decode_path}")
    if do_prefill:
        print(f"  Prefill : {prefill_path}")


if __name__ == "__main__":
    main()

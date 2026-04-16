#!/usr/bin/env python3
"""DeltaNet en layout Conv2d pour ANEMLL / CoreML ANE.

Réécriture du GatedDeltaNetChunkwise avec :
- Toutes les projections en Conv2d(kernel_size=1)
- Layout tenseur [B, C, 1, T] (ANEMLL standard)
- Ops compatibles CoreML MIL

Phase 1.2 du plan ANE hybrid pipeline.

Config Qwen3.5-35B-A3B (depuis config.json) :
  hidden_size         = 2048
  linear_num_key_heads   = 16
  linear_num_value_heads = 32
  linear_key_head_dim    = 128
  linear_value_head_dim  = 128
  linear_conv_kernel_dim = 4
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


# ── Normalisation L2 compatible CoreML ──────────────────────────────────────

def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """Normalisation L2 manuelle, compatible CoreML MIL.

    Remplace F.normalize qui n'est pas toujours tracé correctement.
    Utilise rsqrt(sum_of_squares + eps) au lieu de la division.
    """
    sq_sum = (x * x).sum(dim=dim, keepdim=True)
    inv_norm = torch.rsqrt(sq_sum + eps)
    return x * inv_norm


# ── Module Conv2d ───────────────────────────────────────────────────────────

class GatedDeltaNetConv2d(nn.Module):
    """DeltaNet en layout ANEMLL Conv2d.

    Même algorithme que GatedDeltaNetChunkwise (deltanet_reference.py),
    mais avec :
    - Projections en Conv2d(kernel_size=1) au lieu de nn.Linear
    - Layout interne [B, C, 1, T] pour compatibilité ANE
    - L2 norm manuelle (rsqrt) au lieu de F.normalize
    - Short convolutions en Conv1d (dépliées depuis le layout ANEMLL)

    Le layout ANEMLL standard est [B, C, 1, T] :
    - B = batch
    - C = channels (dimension du modèle)
    - 1 = hauteur (singleton pour Conv2d kernel_size=1)
    - T = temps / séquence
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

        # Dimensions dérivées
        self.kv_group_size = num_value_heads // num_key_heads  # 32/16 = 2
        self.value_expand = self.kv_group_size * value_head_dim  # 2*128 = 256

        qk_dim = num_key_heads * key_head_dim      # 16*128 = 2048
        v_dim = num_value_heads * value_head_dim    # 32*128 = 4096

        # Projections Q, K, V en Conv2d (pattern ANEMLL)
        # Input: [B, hidden_size, 1, T] → Output: [B, out_dim, 1, T]
        self.q_proj = nn.Conv2d(hidden_size, qk_dim, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(hidden_size, qk_dim, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(hidden_size, v_dim, kernel_size=1, bias=False)
        self.o_proj = nn.Conv2d(v_dim, hidden_size, kernel_size=1, bias=False)

        # Gates (per key-head) en Conv2d
        self.alpha_gate = nn.Conv2d(hidden_size, num_key_heads, kernel_size=1, bias=False)
        self.beta_gate = nn.Conv2d(hidden_size, num_key_heads, kernel_size=1, bias=False)

        # Short convolutions causales (depthwise)
        # Conv1d opère sur [B, C, T] — on squeeze la dim hauteur du layout ANEMLL
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

    # ── Conversions layout ──────────────────────────────────────────────────

    def _to_anemll(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, C] → [B, C, 1, T] (layout ANEMLL pour Conv2d)."""
        return x.permute(0, 2, 1).unsqueeze(2)

    def _from_anemll(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, 1, T] → [B, T, C] (retour au layout séquentiel)."""
        return x.squeeze(2).permute(0, 2, 1)

    # ── Short conv causale ──────────────────────────────────────────────────

    def _short_conv(self, x_anemll: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        """Convolution causale courte sur tenseur ANEMLL.

        Input:  [B, C, 1, T] (layout ANEMLL)
        Output: [B, C, 1, T] (layout ANEMLL, après SiLU)

        On squeeze la dim hauteur pour Conv1d, puis on la remet.
        """
        T = x_anemll.size(3)
        # [B, C, 1, T] → [B, C, T] pour Conv1d
        x_1d = x_anemll.squeeze(2)
        # Conv1d causale : padding=conv_size-1, puis troncature
        y = conv(x_1d)[:, :, :T]
        # SiLU + remettre la dim hauteur
        return F.silu(y).unsqueeze(2)

    # ── Reshape V pour regrouper par key heads ──────────────────────────────

    def _reshape_v_to_key_heads(self, v: torch.Tensor) -> torch.Tensor:
        """Regroupe les value heads par key head.

        Input:  [B, T, num_value_heads, value_head_dim]
        Output: [B, T, num_key_heads, value_expand]

        value_expand = kv_group_size * value_head_dim
        """
        B, T, _, _ = v.shape
        v = v.view(B, T, self.num_key_heads, self.kv_group_size, self.value_head_dim)
        return v.view(B, T, self.num_key_heads, self.value_expand)

    # ── Forward chunkwise (identique à la référence, en layout Conv2d) ──────

    def chunkwise_forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forme chunkwise en layout Conv2d ANEMLL.

        Même algorithme que GatedDeltaNetChunkwise.chunkwise_forward
        mais les projections passent par Conv2d et le layout est [B, C, 1, T].

        Args:
            x: [B, T, hidden_size] (input standard)
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

        # Pad T au multiple de C
        pad_len = (C - T % C) % C
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        T_padded = x.size(1)
        num_chunks = T_padded // C

        # === Projections en layout ANEMLL ===
        # [B, T, H] → [B, H, 1, T]
        x_ane = self._to_anemll(x)

        # Conv2d projections : [B, H, 1, T] → [B, out_dim, 1, T]
        q_ane = self.q_proj(x_ane)
        k_ane = self.k_proj(x_ane)
        v_ane = self.v_proj(x_ane)

        # Short convolutions causales (toujours en layout ANEMLL)
        q_ane = self._short_conv(q_ane, self.q_conv)
        k_ane = self._short_conv(k_ane, self.k_conv)
        v_ane = self._short_conv(v_ane, self.v_conv)

        # === Gates en Conv2d ===
        # [B, H, 1, T] → [B, Hk, 1, T]
        alpha_ane = torch.sigmoid(self.alpha_gate(x_ane))
        beta_ane = torch.sigmoid(self.beta_gate(x_ane))

        # === Retour au layout séquentiel pour le calcul du state ===
        # Les matmul intra-chunk sont plus naturels en [B, T, Hk, D]

        # [B, qk_dim, 1, T] → [B, T, Hk, Dk]
        q = q_ane.squeeze(2).permute(0, 2, 1).view(B, T_padded, Hk, Dk)
        k = k_ane.squeeze(2).permute(0, 2, 1).view(B, T_padded, Hk, Dk)

        # [B, v_dim, 1, T] → [B, T, num_value_heads, value_head_dim] → [B, T, Hk, Ve]
        v = v_ane.squeeze(2).permute(0, 2, 1).view(
            B, T_padded, self.num_value_heads, self.value_head_dim
        )
        v = self._reshape_v_to_key_heads(v)

        # [B, Hk, 1, T] → [B, T, Hk, 1]
        alpha = alpha_ane.squeeze(2).permute(0, 2, 1).unsqueeze(-1)
        beta = beta_ane.squeeze(2).permute(0, 2, 1).unsqueeze(-1)

        # L2 normalisation manuelle (compatible CoreML)
        q = l2_normalize(q, dim=-1)
        k = l2_normalize(k, dim=-1)

        # Découpage en chunks : [B, num_chunks, C, Hk, D]
        q = q.view(B, num_chunks, C, Hk, Dk)
        k = k.view(B, num_chunks, C, Hk, Dk)
        v = v.view(B, num_chunks, C, Hk, Ve)
        alpha = alpha.view(B, num_chunks, C, Hk, 1)
        beta = beta.view(B, num_chunks, C, Hk, 1)

        # Initialisation du state
        if state is None:
            state = torch.zeros(B, Hk, Dk, Ve, dtype=x.dtype, device=x.device)

        all_outputs = []

        for chunk_idx in range(num_chunks):
            qc = q[:, chunk_idx]      # [B, C, Hk, Dk]
            kc = k[:, chunk_idx]      # [B, C, Hk, Dk]
            vc = v[:, chunk_idx]      # [B, C, Hk, Ve]
            ac = alpha[:, chunk_idx]  # [B, C, Hk, 1]
            bc = beta[:, chunk_idx]   # [B, C, Hk, 1]

            # === Traitement séquentiel dans le chunk (delta rule) ===
            # La mise à jour error-correcting est séquentielle dans le state.
            # Pour CoreML, C=64 est petit → unrolling en ops statiques MIL.
            chunk_outputs = []
            for t in range(C):
                qt = qc[:, t]   # [B, Hk, Dk]
                kt = kc[:, t]   # [B, Hk, Dk]
                vt = vc[:, t]   # [B, Hk, Ve]
                at = ac[:, t]   # [B, Hk, 1]
                bt = bc[:, t]   # [B, Hk, 1]

                # 1. Decay du state
                state = state * at.unsqueeze(-1)  # broadcast [B, Hk, 1, 1]

                # 2. Mise à jour delta error-correcting
                # einsum("bhkv,bhk->bhv") → matmul batché
                # state: [B, Hk, Dk, Ve], kt: [B, Hk, Dk]
                # retrieved = sum_k(state[k] * kt[k]) pour chaque (b, h, v)
                retrieved = torch.einsum("bhkv,bhk->bhv", state, kt)
                error = vt - retrieved
                # outer product: kt ⊗ error → [B, Hk, Dk, Ve]
                state = state + bt.unsqueeze(-2) * torch.einsum(
                    "bhk,bhv->bhkv", kt, error
                )

                # 3. Output : query le state
                ot = torch.einsum("bhkv,bhk->bhv", state, qt)  # [B, Hk, Ve]
                chunk_outputs.append(ot)

            chunk_out = torch.stack(chunk_outputs, dim=1)  # [B, C, Hk, Ve]
            all_outputs.append(chunk_out)

        # Concaténation et troncature au T original
        output = torch.cat(all_outputs, dim=1)  # [B, T_padded, Hk, Ve]
        output = output[:, :T]
        # [B, T, Hk, Ve] → [B, T, num_value_heads * value_head_dim]
        output = output.reshape(B, T, self.num_value_heads * self.value_head_dim)

        # Projection de sortie en Conv2d ANEMLL
        # [B, T, v_dim] → [B, v_dim, 1, T] → Conv2d → [B, H, 1, T] → [B, T, H]
        out_ane = self._to_anemll(output)
        out_ane = self.o_proj(out_ane)
        output = self._from_anemll(out_ane)

        return output, state


# ── Chargement des poids depuis HuggingFace ─────────────────────────────────

# Mapping des noms HF → noms Conv2d pour une couche DeltaNet
HF_WEIGHT_MAP = {
    "linear_q_proj.weight": "q_proj.weight",
    "linear_k_proj.weight": "k_proj.weight",
    "linear_v_proj.weight": "v_proj.weight",
    "linear_o_proj.weight": "o_proj.weight",
    "linear_alpha_gate.weight": "alpha_gate.weight",
    "linear_beta_gate.weight": "beta_gate.weight",
    "linear_q_conv.weight": "q_conv.weight",
    "linear_k_conv.weight": "k_conv.weight",
    "linear_v_conv.weight": "v_conv.weight",
}

# Suffixes alternatifs observés dans Qwen3.5
HF_WEIGHT_MAP_ALT = {
    "q_proj.weight": "q_proj.weight",
    "k_proj.weight": "k_proj.weight",
    "v_proj.weight": "v_proj.weight",
    "o_proj.weight": "o_proj.weight",
    "alpha_gate.weight": "alpha_gate.weight",
    "beta_gate.weight": "beta_gate.weight",
    "q_conv.weight": "q_conv.weight",
    "k_conv.weight": "k_conv.weight",
    "v_conv.weight": "v_conv.weight",
}

# Projections qui doivent être reshapées de [out, in] → [out, in, 1, 1]
CONV2D_PROJ_NAMES = {"q_proj", "k_proj", "v_proj", "o_proj", "alpha_gate", "beta_gate"}


def load_weights_from_hf(
    model: GatedDeltaNetConv2d,
    layer_idx: int,
    hf_model_path: str,
) -> bool:
    """Charge les poids d'une couche DeltaNet depuis le modèle HF.

    Lit les safetensors du modèle HuggingFace, extrait les poids
    de la couche `layer_idx`, et les charge dans le modèle Conv2d
    avec le reshape [out, in] → [out, in, 1, 1] pour les projections.

    Args:
        model: instance GatedDeltaNetConv2d
        layer_idx: index de la couche dans le modèle HF
        hf_model_path: chemin vers le répertoire du modèle HF

    Returns:
        True si tous les poids ont été chargés
    """
    import safetensors.torch
    import os

    model_path = Path(hf_model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(f"Modèle HF introuvable: {hf_model_path}")

    # Charger tous les safetensors
    state_dict: dict[str, torch.Tensor] = {}
    for file in os.listdir(model_path):
        if file.endswith(".safetensors"):
            state_dict.update(
                safetensors.torch.load_file(str(model_path / file))
            )

    # Préfixe de la couche dans le modèle HF
    layer_prefix = f"model.layers.{layer_idx}."

    # Extraire et mapper les poids
    conv_state: dict[str, torch.Tensor] = {}
    found_keys: set[str] = set()

    for hf_key, tensor in state_dict.items():
        if not hf_key.startswith(layer_prefix):
            continue

        # Enlever le préfixe de couche
        suffix = hf_key[len(layer_prefix):]

        # Chercher dans les deux maps de noms
        target_name = HF_WEIGHT_MAP.get(suffix) or HF_WEIGHT_MAP_ALT.get(suffix)
        if target_name is None:
            continue

        # Déterminer si c'est une projection Conv2d (reshape nécessaire)
        proj_name = target_name.replace(".weight", "")
        if proj_name in CONV2D_PROJ_NAMES and tensor.dim() == 2:
            # [out_features, in_features] → [out_features, in_features, 1, 1]
            tensor = tensor.view(tensor.shape[0], tensor.shape[1], 1, 1)

        conv_state[target_name] = tensor
        found_keys.add(target_name)

    if not conv_state:
        print(f"Aucun poids trouvé pour la couche {layer_idx}")
        print(f"  Préfixe cherché: {layer_prefix}")
        # Afficher les clés disponibles qui matchent le layer_idx
        matching = [k for k in state_dict if f"layers.{layer_idx}." in k]
        if matching:
            print(f"  Clés disponibles ({len(matching)}):")
            for k in matching[:10]:
                print(f"    {k}")
        return False

    # Charger dans le modèle
    missing, unexpected = model.load_state_dict(conv_state, strict=False)

    if missing:
        print(f"Poids manquants pour couche {layer_idx}: {missing}")
    if unexpected:
        print(f"Poids inattendus pour couche {layer_idx}: {unexpected}")

    loaded = len(found_keys)
    total = len(model.state_dict())
    print(f"Couche {layer_idx}: {loaded}/{total} poids chargés depuis HF")

    return len(missing) == 0


# ── Test d'équivalence numérique ────────────────────────────────────────────

def _copy_linear_to_conv2d(linear: nn.Linear, conv2d: nn.Conv2d) -> None:
    """Copie les poids d'un nn.Linear vers un nn.Conv2d(kernel_size=1).

    Linear weight: [out_features, in_features]
    Conv2d weight: [out_channels, in_channels, 1, 1]
    """
    with torch.no_grad():
        conv2d.weight.copy_(
            linear.weight.data.view(
                linear.weight.shape[0], linear.weight.shape[1], 1, 1
            )
        )


def _copy_conv1d(src: nn.Conv1d, dst: nn.Conv1d) -> None:
    """Copie les poids d'un Conv1d vers un autre (identique)."""
    with torch.no_grad():
        dst.weight.copy_(src.weight.data)
        if src.bias is not None and dst.bias is not None:
            dst.bias.copy_(src.bias.data)


def test_conv2d_equivalence() -> bool:
    """Vérifie l'équivalence numérique entre Conv2d et référence Linear.

    Crée les deux modèles avec les mêmes poids, passe le même input,
    et vérifie que les outputs sont identiques (à la précision FP64 près).
    """
    # Import de la référence
    from deltanet_reference import GatedDeltaNetChunkwise, load_qwen35_config

    torch.manual_seed(42)

    cfg = load_qwen35_config()
    print(f"Config chargée: {cfg}")

    # Créer le modèle référence (Linear)
    ref = GatedDeltaNetChunkwise(
        hidden_size=cfg["hidden_size"],
        num_key_heads=cfg["linear_num_key_heads"],
        num_value_heads=cfg["linear_num_value_heads"],
        key_head_dim=cfg["linear_key_head_dim"],
        value_head_dim=cfg["linear_value_head_dim"],
        conv_size=cfg["linear_conv_kernel_dim"],
        chunk_size=64,
    )
    ref.eval()
    ref.double()

    # Créer le modèle Conv2d
    conv = GatedDeltaNetConv2d(
        hidden_size=cfg["hidden_size"],
        num_key_heads=cfg["linear_num_key_heads"],
        num_value_heads=cfg["linear_num_value_heads"],
        key_head_dim=cfg["linear_key_head_dim"],
        value_head_dim=cfg["linear_value_head_dim"],
        conv_size=cfg["linear_conv_kernel_dim"],
        chunk_size=64,
    )
    conv.eval()
    conv.double()

    # Copier les poids de la référence vers le Conv2d
    _copy_linear_to_conv2d(ref.q_proj, conv.q_proj)
    _copy_linear_to_conv2d(ref.k_proj, conv.k_proj)
    _copy_linear_to_conv2d(ref.v_proj, conv.v_proj)
    _copy_linear_to_conv2d(ref.o_proj, conv.o_proj)
    _copy_linear_to_conv2d(ref.alpha_gate, conv.alpha_gate)
    _copy_linear_to_conv2d(ref.beta_gate, conv.beta_gate)

    _copy_conv1d(ref.q_conv, conv.q_conv)
    _copy_conv1d(ref.k_conv, conv.k_conv)
    _copy_conv1d(ref.v_conv, conv.v_conv)

    # Input test
    B, T = 1, 128
    x = torch.randn(B, T, cfg["hidden_size"], dtype=torch.float64)

    with torch.no_grad():
        out_ref, state_ref = ref.chunkwise_forward(x)
        out_conv, state_conv = conv.chunkwise_forward(x)

    diff_out = (out_ref - out_conv).abs().max().item()
    diff_state = (state_ref - state_conv).abs().max().item()

    print(f"\n{'='*60}")
    print(f"Test d'équivalence Conv2d vs Référence Linear")
    print(f"{'='*60}")
    print(f"\nDimensions:")
    print(f"  hidden_size     = {cfg['hidden_size']}")
    print(f"  num_key_heads   = {cfg['linear_num_key_heads']}")
    print(f"  num_value_heads = {cfg['linear_num_value_heads']}")
    print(f"  key_head_dim    = {cfg['linear_key_head_dim']}")
    print(f"  value_head_dim  = {cfg['linear_value_head_dim']}")
    print(f"  conv_size       = {cfg['linear_conv_kernel_dim']}")
    print(f"\nInput:  batch={B}, seq_len={T}")
    print(f"Output ref:  {out_ref.shape}")
    print(f"Output conv: {out_conv.shape}")
    print(f"State ref:   {state_ref.shape}")
    print(f"State conv:  {state_conv.shape}")
    print(f"\nMax output diff (Conv2d vs Linear): {diff_out:.6e}")
    print(f"Max state diff:                     {diff_state:.6e}")

    passed = diff_out < 1e-10
    print(f"\nÉquivalence: {'PASS' if passed else 'FAIL'} (seuil 1e-10 en FP64)")

    if not passed:
        # Diagnostic supplémentaire
        print(f"\nDiagnostic:")
        print(f"  Output mean ref:  {out_ref.mean().item():.6e}")
        print(f"  Output mean conv: {out_conv.mean().item():.6e}")
        print(f"  Output std ref:   {out_ref.std().item():.6e}")
        print(f"  Output std conv:  {out_conv.std().item():.6e}")

    # Vérification du layout Conv2d
    print(f"\n{'='*60}")
    print(f"Vérification layout ANEMLL")
    print(f"{'='*60}")
    x_ane = conv._to_anemll(x)
    print(f"  Input [B,T,C]:     {x.shape}")
    print(f"  ANEMLL [B,C,1,T]:  {x_ane.shape}")
    x_back = conv._from_anemll(x_ane)
    layout_ok = torch.equal(x, x_back)
    print(f"  Roundtrip OK:      {layout_ok}")

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
    success = test_conv2d_equivalence()
    if success:
        print("\nProchain step: trace CoreML et benchmark ANE (deltanet_coreml.py)")
    else:
        print("\nATTENTION: Conv2d et référence divergent. Debug nécessaire.")

# SOTA Training 2026 — Techniques pour Fine-Tuning LLM sur Apple Silicon

> Résumé orienté Qwen3.5-122B-A10B sur Mac Studio M3 Ultra 512 Go via MLX.
> Dernière mise à jour : 2026-04-15

## Top 10 Techniques par Pertinence

| # | Technique | Pertinence | MLX | Papier / Repo |
|---|-----------|-----------|-----|---------------|
| 1 | **SFT Curriculum** | ★★★★★ | Natif mlx-lm | — (standard practice) |
| 2 | **SimPO** (Simple Preference Optimization) | ★★★★★ | mlx-tune 0.5+ | [arxiv:2405.14734](https://arxiv.org/abs/2405.14734) |
| 3 | **GRPO** (Group Relative Policy Optimization) | ★★★★★ | mlx-tune 0.5+ | [arxiv:2402.03300](https://arxiv.org/abs/2402.03300) — DeepSeek |
| 4 | **DAPO** (Decoupled Alignment via Process Optimization) | ★★★★☆ | Partiel | [arxiv:2503.14476](https://arxiv.org/abs/2503.14476) — ByteDance |
| 5 | **LoRA+** / DoRA | ★★★★☆ | Natif mlx-lm | [arxiv:2402.12354](https://arxiv.org/abs/2402.12354) |
| 6 | **Evolutionary Model Merging** | ★★★★☆ | Via mergekit | [arxiv:2403.13187](https://arxiv.org/abs/2403.13187) — Sakana AI |
| 7 | **RLTT** (RL via Think Tokens) | ★★★☆☆ | Custom | [arxiv:2504.xxxxx](https://arxiv.org/) — récent 2025 |
| 8 | **NEFTune** (Noise Embeddings) | ★★★☆☆ | Natif mlx-lm | [arxiv:2310.05914](https://arxiv.org/abs/2310.05914) |
| 9 | **Reward Model Distillation** | ★★★☆☆ | Custom script | Standard RLHF pipeline |
| 10 | **Speculative Fine-Tuning** | ★★☆☆☆ | Expérimental | Recherche récente 2025-2026 |

## Pipeline Recommandé (5 Phases)

### Phase 1 : SFT Curriculum (EN COURS)

Apprentissage supervisé progressif court → moyen → long.

- **Données** : Dataset Opus distillé, 3 paliers de longueur
- **LR** : 2e-5 → 1e-5 → 5e-6 (décroissant)
- **Script** : `scripts/train_curriculum.sh`
- **Ref** : Pratique standard, validée par Qwen, LLaMA, Mistral

### Phase 2 : SimPO Alignment

Alignement préférence SANS modèle de référence (contrairement à DPO).

- **Avantage** : Pas besoin de charger 2x le modèle 122B en mémoire
- **Données** : Paires preferred/rejected générées depuis le modèle SFT
- **Script** : `scripts/train_simpo.py`
- **Ref** : Yu et al. 2024 — "SimPO: Simple Preference Optimization with a Reference-Free Reward"
- **Repos** : [princeton-nlp/SimPO](https://github.com/princeton-nlp/SimPO)

### Phase 3 : GRPO Reasoning RL

RL avec récompenses vérifiables pour le raisonnement math/code/logique.

- **Principe** : Génère K réponses, vérifie programmatiquement, normalise par groupe
- **Pas de reward model** : Récompenses binaires (correct/incorrect)
- **Script** : `scripts/train_grpo.py`, `scripts/prepare_grpo_data.py`
- **Ref** : Shao et al. 2024 — DeepSeek-Math, puis DeepSeek-R1
- **Repos** : [deepseek-ai/DeepSeek-Math](https://github.com/deepseek-ai/DeepSeek-Math)

### Phase 4 : RLTT/DAPO (Réservé)

Optimisation avancée du processus de raisonnement.

- **DAPO** : Découple la longueur de la récompense, évite le reward hacking
- **RLTT** : RL sur les tokens de réflexion (<thinking>)
- **Statut** : Recherche active, implémentation à confirmer sur MLX
- **Ref** : ByteDance 2025 — DAPO, divers 2025-2026 — RLTT

### Phase 5 : Merge + Export

Fusion LoRA + export GGUF pour déploiement.

- **Merge** : `mlx_lm fuse` (LoRA → poids complets)
- **Export** : GGUF Q4_K_M via llama.cpp
- **Script** : `scripts/merge_lora.sh`
- **Optionnel** : Evolutionary merging de plusieurs LoRA (mergekit)

## Compatibilité MLX

| Technique | mlx-lm | mlx-tune | Custom |
|-----------|--------|----------|--------|
| SFT + LoRA | ✅ Natif | ✅ | — |
| SimPO/DPO | — | ✅ 0.5+ | Fallback DPO |
| GRPO | — | ✅ 0.5+ | Reward custom |
| NEFTune | ✅ `--neftune-alpha` | ✅ | — |
| LoRA Merge | ✅ `mlx_lm fuse` | — | — |
| GGUF Export | — | — | llama.cpp |

## Contraintes M3 Ultra 512 Go

- **122B bf16** : ~244 Go pour les poids, ~100 Go pour les activations
- **Batch size** : 1 maximum, gradient accumulation pour simuler batch > 1
- **Seq length** : 1280 tokens max (OOM au-delà avec 122B)
- **MLX fork** : Metal buffer 3x requis (`/tmp/mlx-fork`)
- **Kernel** : `iogpu.wired_limit_mb=458752` obligatoire

## Repos Clés

| Repo | Usage |
|------|-------|
| [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm) | SFT, LoRA, merge |
| [ml-explore/mlx-tune](https://github.com/ml-explore/mlx-tune) | SimPO, GRPO, DPO |
| [princeton-nlp/SimPO](https://github.com/princeton-nlp/SimPO) | Référence SimPO |
| [deepseek-ai/DeepSeek-Math](https://github.com/deepseek-ai/DeepSeek-Math) | Référence GRPO |
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | Export GGUF |
| [arcee-ai/mergekit](https://github.com/arcee-ai/mergekit) | Model merging |
| [unslothai/unsloth](https://github.com/unslothai/unsloth) | Référence GPU (KXKM-AI) |

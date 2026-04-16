# Plan 4 : ANE Triple Pipeline (Mac uniquement)

**Date** : 2026-04-15
**Prerequis** : Plans 1-3 termines (32 stacks + routeur), conversion CoreML prouvee (research/ane-hybrid/)
**Machine** : Mac M3 Ultra 512 Go (ANE 32 cores, GPU Metal 76 cores, CPU 24 cores)
**Performance cible** : 2-3x tok/s effectifs (speculative), scoring GRPO gratuit, routeur < 2 ms

## Contexte

Le Mac M3 Ultra a un Neural Engine (ANE) de 32 cores a ~2W qui reste inutilise pendant l'inference GPU. Trois integrations exploitent cette ressource pour accelerer Micro_KIKI sans cout VRAM/GPU supplementaire.

La Phase 1 ANE research (`research/ane-hybrid/`) a prouve :
- DeltaNet GatedDeltaNet se convertit en CoreML (diff < 1e-6)
- ct.StateType fonctionne pour l'etat recurrent [1, H, K, V]
- ANE atteint 14.4 tok/s sur 40 couches DeltaNet (474 tok/s/couche)
- Le dispatch overhead CoreML est ~2ms/couche

Le Qwen3.5-0.8B utilise la meme architecture GatedDeltaNet que le 4B/9B/35B — meme famille, meme tokenizer. Dimensions estimees du 0.8B :
- hidden_size: 1536
- num_layers: 24 (18 DeltaNet + 6 Full Attention)
- num_key_heads: 8, num_value_heads: 16
- key_head_dim: 64, value_head_dim: 64
- vocab_size: 248064 (meme tokenizer Qwen3.5)

## Architecture triple pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    MAC M3 ULTRA 512 Go                       │
│                                                              │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────┐ │
│  │   GPU METAL      │  │   ANE (32 cores)  │  │    CPU     │ │
│  │   76 cores       │  │   ~2W, 14 tok/s   │  │  24 cores  │ │
│  │                  │  │                    │  │            │ │
│  │  Base Qwen3.5-4B │  │  A. Scorer GRPO   │  │  Routeur   │ │
│  │  + 2-4 stacks    │  │  B. Draft 0.8B    │  │  sigmoid   │ │
│  │  actifs          │  │     (speculative)  │  │  (5ms)     │ │
│  │                  │  │  C. Meta-routeur   │  │            │ │
│  │  Generation      │  │     + embedding    │  │  Offload   │ │
│  │  principale      │  │                    │  │  stacks    │ │
│  └─────────────────┘  └──────────────────┘  └────────────┘ │
│         ↕ memoire unifiee (zero-copy)  ↕                     │
└─────────────────────────────────────────────────────────────┘
```

## Fichiers

| Fichier | Responsabilite |
|---------|---------------|
| `scripts/micro_kiki/convert_08b_coreml.py` | Conversion Qwen3.5-0.8B vers CoreML (decode + prefill) |
| `scripts/micro_kiki/ane_scorer.py` | Scorer CoreML avec reward head pour GRPO |
| `scripts/micro_kiki/ane_speculative.py` | Speculative decoding ANE draft + GPU verify |
| `scripts/micro_kiki/ane_router.py` | Meta-routeur + embedding en CoreML pour ANE |
| `scripts/micro_kiki/triple_pipeline.py` | Integration unifiee des trois pipelines |

## Task 1 : Conversion Qwen3.5-0.8B vers CoreML

**Duree** : 4h
**Fichier** : `scripts/micro_kiki/convert_08b_coreml.py`
**Dependances** : coremltools 9.0, transformers, safetensors

### Sous-taches

1.1 Charger les poids Qwen3.5-0.8B depuis HuggingFace (ou cache local)
1.2 Extraire la config DeltaNet (hidden_size, num_heads, etc.)
1.3 Construire le wrapper decode (1 token recurrent) avec ct.StateType
1.4 Construire le wrapper prefill (chunk=64 tokens)
1.5 Convertir en .mlpackage avec compute_units=CPU_AND_NE
1.6 Test numerique CoreML vs PyTorch (diff < 0.01)
1.7 Benchmark ANE : mesurer tok/s sur le modele complet 0.8B

### Criteres de succes

- [ ] .mlpackage genere sans erreur
- [ ] Diff numerique < 0.01 (fp16 vs fp32)
- [ ] Execution confirmee sur ANE (pas fallback CPU)
- [ ] Benchmark > 100 tok/s pour le 0.8B seul sur ANE

## Task 2 : ANE Scorer avec reward head pour GRPO

**Duree** : 3h
**Fichier** : `scripts/micro_kiki/ane_scorer.py`
**Dependances** : Task 1 (0.8B CoreML)

### Architecture

```
Qwen3.5-0.8B CoreML (ANE)
  → hidden_states [1, hidden_size]
  → Linear(hidden_size, 1)  # reward head
  → scalar score ∈ [0, 1]
```

### Sous-taches

2.1 Charger le 0.8B CoreML (Task 1)
2.2 Ajouter une tete de reward (Linear + sigmoid) au modele CoreML
2.3 Implementer le scoring pipeline : tokenize → forward ANE → score
2.4 Implementer le scoring batch : K=4 reponses en parallele
2.5 Benchmark : mesurer le temps de scoring vs generation GPU

### Criteres de succes

- [ ] Score une reponse complete en < 100ms sur ANE
- [ ] Scoring parallele a la generation GPU (0 overhead)
- [ ] Scores discriminants (spread > 0.1 entre bonnes et mauvaises reponses)

## Task 3 : Speculative decoding ANE draft + GPU verify

**Duree** : 6h
**Fichier** : `scripts/micro_kiki/ane_speculative.py`
**Dependances** : Task 1 (0.8B CoreML)

### Architecture

```
Boucle speculative :
  1. ANE (0.8B draft) : genere N=5 tokens candidats
  2. GPU (4B + stacks) : verifie N tokens en 1 forward pass
  3. Accept : premiers K tokens ou tous sont bons
  4. Repete depuis le dernier token accepte
```

### Sous-taches

3.1 Implementer le draft model ANE (autoregressive, single-token decode)
3.2 Implementer la verification GPU (batch forward de N tokens)
3.3 Implementer la logique d'acceptation (comparaison logits)
3.4 Gerer la synchronisation ANE ↔ GPU (memoire unifiee, zero-copy)
3.5 Tuner N (nombre de tokens draft) pour maximiser le throughput
3.6 Benchmark : tok/s effectifs avec vs sans speculative

### Criteres de succes

- [ ] Acceptance rate > 60% (draft 0.8B vs target 4B)
- [ ] 2-3x tok/s effectifs (60-100 tok/s vs 30-50)
- [ ] Pas de regression de qualite (meme distribution de sortie)
- [ ] Latence totale < 2x la latence d'un token GPU seul

## Task 4 : Conversion meta-routeur vers CoreML

**Duree** : 2h
**Fichier** : `scripts/micro_kiki/ane_router.py` (partie 1)
**Dependances** : Aucune (routeur est un petit MLP)

### Architecture

```
Meta-routeur (~2M params) :
  Input: 0.45 × mid_hidden + 0.55 × last_hidden (h_dim=3072)
  → Linear(3072, 512) + GELU
  → Global attention (learned query)
  → 32 × cross-attention (domain query vectors)
  → MLP fusion (GELU, dropout 0.1)
  → 32 sigmoid outputs avec temperature scaling
```

### Sous-taches

4.1 Definir le modele PyTorch du meta-routeur (meme architecture que la spec)
4.2 Convertir en CoreML avec coremltools
4.3 Test numerique (diff < 1e-4)
4.4 Benchmark : latence routeur sur ANE vs CPU

### Criteres de succes

- [ ] .mlpackage < 10 Mo
- [ ] Latence < 2 ms sur ANE
- [ ] Diff numerique < 1e-4

## Task 5 : Conversion embedding layer vers CoreML

**Duree** : 2h
**Fichier** : `scripts/micro_kiki/ane_router.py` (partie 2)
**Dependances** : Aucune

### Architecture

```
Embedding layer (Qwen3.5-4B) :
  vocab_size=248064, hidden_size=3072
  → Lookup table: [248064, 3072] float16
  → ~1.5 Go de poids
```

### Sous-taches

5.1 Extraire les poids d'embedding depuis le modele 4B
5.2 Construire un module CoreML d'embedding lookup
5.3 Convertir en .mlpackage
5.4 Benchmark : latence embedding ANE vs CPU

### Criteres de succes

- [ ] Embedding 128 tokens en < 1 ms
- [ ] Zero-copy vers GPU pour le forward MoE

## Task 6 : Integration triple pipeline

**Duree** : 6h
**Fichier** : `scripts/micro_kiki/triple_pipeline.py`
**Dependances** : Tasks 1-5

### Architecture

```
Pipeline unifie :
  ┌─────────────────────────────────────────────────┐
  │ Prompt arrive                                    │
  │   ├→ ANE : embedding(tokens)         (~0.5 ms)  │
  │   ├→ ANE : meta_routeur(hidden)      (~2 ms)    │
  │   ├→ CPU : select top-4 stacks       (~50 ms)   │
  │   └→ GPU : forward(hidden, stacks)   (bulk)     │
  │                                                   │
  │ Generation :                                      │
  │   ├→ ANE : draft 0.8B (5 tokens)    (parallele) │
  │   └→ GPU : verify (1 forward)       (principal)  │
  │                                                   │
  │ GRPO training :                                   │
  │   ├→ GPU : genere response[i+1]     (principal)  │
  │   └→ ANE : score response[i]        (parallele)  │
  └─────────────────────────────────────────────────┘
```

### Sous-taches

6.1 Implementer le mode inference (speculative + routing ANE)
6.2 Implementer le mode GRPO (scoring ANE + generation GPU)
6.3 Gerer le multiplexage ANE (scorer OU draft, pas les deux)
6.4 Implementer le fallback CPU si ANE est saturee
6.5 Tests d'integration complets

### Criteres de succes

- [ ] Mode inference : 60-100 tok/s effectifs
- [ ] Mode GRPO : scoring overhead ~0% (parallele GPU)
- [ ] Pas de deadlock ANE/GPU
- [ ] Fallback CPU fonctionnel

## Task 7 : Benchmark complet

**Duree** : 3h
**Fichier** : `scripts/micro_kiki/triple_pipeline.py` (section benchmark)
**Dependances** : Tasks 1-6

### Metriques

| Metrique | Sans ANE | Cible avec ANE |
|----------|----------|----------------|
| Inference tok/s | 30-50 | 60-100 (speculative) |
| GRPO scoring overhead | +50% temps | ~0% (parallele) |
| Latence routeur | ~5 ms CPU | < 2 ms ANE |
| Latence embedding | ~1 ms CPU | < 0.5 ms ANE |
| Consommation totale | ~20W GPU | ~22W (GPU+ANE) |

### Sous-taches

7.1 Benchmark tok/s avec vs sans speculative decoding
7.2 Benchmark scoring GRPO overhead (parallele vs sequentiel)
7.3 Benchmark latence routeur (ANE vs CPU)
7.4 Benchmark latence embedding (ANE vs CPU)
7.5 Profiling Power consumption (powermetrics)
7.6 Rapport de benchmark complet

### Criteres de succes

- [ ] 2x tok/s minimum avec speculative ANE
- [ ] Scoring GRPO overhead < 5% (vs 50% sans ANE)
- [ ] Rapport de benchmark dans `docs/benchmarks/`

## Estimation temps total

| Task | Duree estimee |
|------|--------------|
| Task 1 : Convert 0.8B CoreML | 4h |
| Task 2 : ANE Scorer GRPO | 3h |
| Task 3 : Speculative decoding | 6h |
| Task 4 : Meta-routeur CoreML | 2h |
| Task 5 : Embedding CoreML | 2h |
| Task 6 : Integration triple | 6h |
| Task 7 : Benchmark | 3h |
| **Total** | **26h** |

## Risques

| Risque | Impact | Mitigation |
|--------|--------|-----------|
| 0.8B trop petit pour bon draft acceptance rate | Speculative ne gagne que 1.5x au lieu de 2-3x | Tuner N, essayer 1.5B si dispo |
| ANE dispatch overhead domine pour petits modeles | Latence > prevue | Fusionner couches, reduire dispatches |
| CoreML ne supporte pas le modele complet 0.8B | Blocage Task 1 | Convertir couche par couche (prouve en Phase 1) |
| Multiplexage ANE (scorer vs draft) cree des conflits | Deadlock ou starvation | Mode exclusif avec file d'attente |
| Reward head non entraine produit des scores random | GRPO inutile | Pre-entrainer sur dataset de preferences |

## Dependances

- coremltools 9.0+ (installe dans venv)
- transformers 4.52+ (pour Qwen3.5 support)
- safetensors
- MLX (pour le forward GPU du modele 4B + stacks)
- Qwen3.5-0.8B (a telecharger depuis HuggingFace)
- Meta-routeur entraine (Plan 3)
- Base Qwen3.5-4B + 32 stacks (Plans 1-3)

## References

- [CoreML Stateful Models (ct.StateType)](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html)
- [Speculative Decoding (Leviathan et al. 2023)](https://arxiv.org/abs/2211.17192)
- [research/ane-hybrid/convert_deltanet.py](../../research/ane-hybrid/convert_deltanet.py) — conversion DeltaNet prouvee
- [research/ane-hybrid/CLAUDE.md](../../research/ane-hybrid/CLAUDE.md) — resultats Phase 1
- [Gated DeltaNet (ICLR 2025)](https://jankautz.com/publications/GatedDeltaNet_ICLR25.pdf)

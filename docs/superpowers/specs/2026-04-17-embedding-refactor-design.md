# Domain-Specific Embedding Refactor — Design Spec

## Goal

Replace the hash-based embedding stub in Aeon Memory with a real
contrastive embedding model (sentence-transformers) trained on the 10
niche domains, so that memory recall is semantically meaningful.

## Architecture

```
data/final/<domain>/train.jsonl
        |
        v
+-----------------------------+
|  train_embeddings.py        |
|  +------------------------+ |
|  | Load texts per domain  | |
|  | Build pairs:           | |
|  |  - MNRL (in-batch neg) | |
|  |  + Hard negatives      | |
|  |    (confusing pairs)   | |
|  +-----------+------------+ |
|              v              |
|  sentence-transformers      |
|  all-MiniLM-L6-v2 (384d)   |
|  MultipleNegativesRanking   |
|  + TripletLoss (hard neg)   |
|              |              |
|              v              |
|  models/niche-embeddings/   |
+-----------------------------+
        |
        v
+-----------------------------+
|  AeonPalace(embed_fn=...)   |
|  AtlasIndex(dim=auto)       |
|  - No hash fallback         |
|  - Missing model -> crash   |
+-----------------------------+
```

## Decisions

### Embedding backend: sentence-transformers

`sentence-transformers` with `all-MiniLM-L6-v2` (384 dims, ~80 MB).
Runs on CPU at ~5ms/query.  Fine-tunable with contrastive loss.
Added as optional dependency: `[embeddings]` group in pyproject.toml.

### Negative sampling: MNRL + hard negatives

Two-phase training per epoch:

1. **MultipleNegativesRankingLoss (MNRL)** on all domain texts.
   In-batch negatives are free: a batch of 32 naturally contains texts
   from different domains.

2. **TripletLoss on hard negatives** for confusing domain pairs:
   - embedded <-> stm32 (firmware overlap)
   - spice <-> power (circuit simulation overlap)
   - kicad-dsl <-> electronics (PCB/component overlap)
   - embedded <-> platformio (build system vs firmware)

### Embedding dimension: configurable

Atlas already accepts `dim=` at init. The trained model outputs 384
dims. AeonPalace infers the dimension from the loaded model at
runtime.  No hardcoded 3072.

### No hash fallback

The SHA-256 hash stub (`_default_embed`) is removed. AeonPalace
requires either an `embed_fn` callable or a `model_path` pointing to
a trained sentence-transformers model. Missing both raises
`ImportError` with instructions to train.

Tests use a mock `embed_fn` (lambda returning a fixed-dim vector).

## Data

Sources: `data/final/<domain>/train.jsonl` from merge_all_sources.py.

Current state (post-merge 2026-04-17):

| Domain       | Examples | Strategy        |
|--------------|----------|-----------------|
| platformio   | 6,357    | cap at 2,000    |
| kicad-dsl    | 2,665    | cap at 2,000    |
| electronics  | 1,146    | use all         |
| freecad      | 200      | use all         |
| spice        | 32       | oversample to 100 |
| emc          | 20       | oversample to 100 |
| stm32        | 14       | oversample to 100 |
| embedded     | 14       | oversample to 100 |
| power        | 2        | oversample to 100 |
| dsp          | 0        | skip until data exists |

Oversampling uses the same strategy as train_vqc_router.py: repeat
examples with random seed until target count reached.

## Dependencies

```toml
[project.optional-dependencies]
embeddings = ["sentence-transformers>=3.0"]
```

Not in base deps.  Only needed for training and live Aeon inference.

## Files touched

| File | Action |
|------|--------|
| scripts/train_embeddings.py | Rewrite: MNRL + hard negatives, drop fictional mlx_tune API |
| src/memory/aeon.py | Edit: remove hash fallback, add load_st_model helper |
| src/memory/atlas.py | None (dim already configurable) |
| pyproject.toml | Edit: add [embeddings] optional dep |
| tests/memory/test_aeon.py | Edit: fixtures use embed_fn mock, no hash assumption |

## Training config

- Base model: `sentence-transformers/all-MiniLM-L6-v2`
- Epochs: 5
- Batch size: 32
- Warmup: 10% of steps
- MNRL loss weight: 1.0
- TripletLoss weight: 0.5 (hard negatives)
- Margin (TripletLoss): 0.3
- Output: `models/niche-embeddings/`
- Evaluation: cosine similarity intra-domain vs inter-domain (target: intra > 0.7, inter < 0.3)

## Success criteria

1. Trained model produces cosine(same-domain) > 0.7 average
2. Trained model produces cosine(cross-domain) < 0.3 average
3. Aeon recall precision >= 0.8 on 10-query test (same as story-20)
4. AeonPalace raises ImportError when no model provided
5. All existing tests pass with mock embed_fn
6. POC v2 multi-turn recall improves (Turn 4 inductor values)

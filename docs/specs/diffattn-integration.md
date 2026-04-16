# Differential Attention Integration

## Source

arxiv 2410.05258 (ICLR 2025, Microsoft). Validated by Dragon LLM per-global-layer findings.

## Design Rationale

Qwen3.5-4B has ~49 layers: ~13 full-attention and ~36 GatedDeltaNet (linear).
DiffAttn is applied ONLY to the 13 full-attention layers because:

1. Full-attention layers are the primary source of activation outliers that hurt Q4 quantization.
2. GatedDeltaNet layers already have built-in gating that serves a similar noise-cancelling role.
3. Per Dragon LLM: differential attention on global (full) layers yields the best long-context and hallucination reduction without disrupting the linear layers' efficiency.

## Mechanism

```
scores = softmax(Q1 * K1) - lambda * softmax(Q2 * K2)
```

- Q/K are split into two halves (Q1, Q2, K1, K2) at half head dimension.
- Lambda is learnable per-head, initialized at `0.8 * (layer_idx + 1) / num_layers`.
- Q2/K2 are warm-started from Q1/K1 with small perturbation (std=0.01).

## Expected Benefits

- Reduced activation outliers (better Q4_K_M quantization fidelity)
- Lower hallucination rate on long-context inputs
- Noise cancellation across attention heads (shared noise subtracted)

## Implementation

- `src/base/diff_attention.py` — DiffAttn module, wrapper, in-place patcher
- `scripts/fork_qwen_diffattn.py` — fork script with perplexity check
- `tests/test_diff_attention.py` — fully mocked unit tests

## Rollback Criteria

The fork is automatically flagged for rollback if ANY of:

1. **Perplexity delta > 3%** — measured on a short calibration text before/after patching.
2. **Activation outlier reduction < 30%** — measured during calibration pass.

On rollback: `fork_metrics.json` is saved with `status: "rollback_recommended"`, the model is NOT saved, and all configs fall back to the vanilla base model path.

## Calibration

Short pass (~5K tokens) to stabilize lambda values. Duration: ~30 min on RTX 4090.

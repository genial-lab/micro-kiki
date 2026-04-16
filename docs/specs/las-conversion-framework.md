# LAS Conversion Framework

Design doc for the LAS (Lossless ANN→SNN) converter used across Phase
N-IV of v0.3. Referenced by stories 17 (framework), 21 (MoE extension),
and 25 (Mistral dense extension).

## 1. Reference

- Paper: *Lossless ANN-SNN Conversion for Modern Transformers via
  Time-Coded Activation Alignment* (working title), arXiv:2505.09659
  (2025).
- Primary claim: near-lossless conversion of pretrained transformer
  blocks into spiking equivalents without retraining, via time-coded
  quantisation of activations plus activation-range alignment.
- Why LAS over Spikingformer (AAAI 2026, arXiv 2304.11954): LAS is
  more modern (2026-ready), supports MoE routing out of the box in its
  reference implementation, and keeps attention softmax exact via a
  staged time-coded accumulation. Spikingformer remains integrated
  (Story 30) as a cross-validation tool.

## 2. Scope in v0.3

Used by three reproductions in Phase N-IV:

| Base model                | Sub-phase | Architecture                         | Story |
|---------------------------|-----------|--------------------------------------|-------|
| Qwen3.5-27B               | N-IV-B    | dense, full-attn every layer         | 19    |
| Qwen3.5-122B-A10B         | N-IV-C    | hybrid linear/full attn + MoE top-K  | 23    |
| Mistral-Large-Opus 123B   | N-IV-D    | dense, full-attn every layer, SwiGLU | 27    |

Each reproduction produces a `SpikingKiki-*` artefact on Studio. Story
29 cross-evaluates all three + SpikingBrain-7B and picks one variant
for the v0.3 release freeze (Story 38).

## 3. Public API

Implemented in `src/spiking/las_converter.py`:

```python
class LASConverter:
    """Lossless ANN→SNN conversion via time-coded activation alignment."""

    def __init__(self, timesteps: int = 16, spike_threshold: float = 1.0):
        ...

    def convert_linear(self, layer: nn.Linear) -> SpikingLinear:
        """Per-layer conversion; preserves weight tensor, wraps in LIF neuron."""

    def convert_attention(self, block: nn.Module) -> SpikingAttention:
        """Attention block → time-coded accumulation; softmax preserved
        in a staged spike protocol."""

    def convert_moe(self, moe_block: nn.Module, top_k: int) -> SpikingMoE:
        """MoE block → spiking router + spiking experts; preserves
        top-K routing semantics (Story 21 extension)."""

    def convert_model(self, model: nn.Module) -> nn.Module:
        """Walk the model tree, dispatch per-layer conversion, wire
        residual connections in the spiking domain."""

    def activation_stats(self) -> dict:
        """Return per-layer activation bounds, spike counts, saturation
        rates — used by energy benchmark (Story 32)."""
```

## 4. Acceptance tests

Per-story tests in `tests/`:

- `test_las_smoke.py::test_linear_identity` (Story 17) — random 128→64
  linear, ANN vs SNN MSE ≤ 1e-4 on a random input batch.
- `test_las_moe.py` (Story 21) — 4-layer micro-MoE (4 experts × 128-d,
  2-token top-2 routing). Expert selection agreement ≥ 99%; output
  MSE ≤ 1e-3.
- `test_las_mistral.py` (Story 25) — 4-layer Mistral-style block
  (4096-d, 8 heads, full attn + SwiGLU MLP). Forward MSE ≤ 1e-3.

All three tests must pass before the corresponding conversion scripts
(19 / 23 / 27) are run on full-scale bases.

## 5. Compute budget

Sequential execution on Studio (M3 Ultra 512 GB unified):

| Base                    | Wall time (est.) | Peak RAM (BF16) | Disk output |
|-------------------------|------------------|------------------|-------------|
| Qwen3.5-27B             | 30-40 h          | ≤ 80 GB          | ~54 GB      |
| Qwen3.5-122B-A10B (MoE) | 100-120 h        | ≤ 480 GB         | ~244 GB     |
| Mistral-Large-Opus 123B | 80-100 h         | ≤ 470 GB         | ~233 GB     |
| **Total**               | **210-260 h**    | n/a (sequential) | ~530 GB     |

Activation checkpointing is mandatory for the 122B and 123B paths —
peak RAM without checkpointing would exceed unified memory.

## 6. Integration with other specs

- `docs/specs/spikingbrain-acquisition.md` — Story 12 decision that 7B
  is the production path; Phase N-IV bases are the custom reproduction
  alongside.
- `docs/specs/2026-04-15-micro-kiki-design.md` — hybrid-attention/MoE
  details for Qwen3.5-122B-A10B (used in Story 21 extension).
- `docs/specs/energy-methodology.md` (Story 32) — reads
  `LASConverter.activation_stats()` to compute spike-count-based
  energy ratios.
- `docs/specs/spikingkiki-cross-eval.md` (Story 29) — aggregates
  the three per-variant eval reports and picks the release variant.

## 7. Open questions

1. LAS reference implementation — is there public code tied to
   arXiv 2505.09659, or must the first cut be a clean-room implementation
   from the paper's pseudocode? Affects Story 17 effort estimate.
2. Timesteps vs accuracy curve — paper reports 16 timesteps lossless on
   ViT; confirm for decoder-only LLMs on reasoning tasks (HumanEval +
   GSM8K subset in each eval story).
3. MoE router numerics — LAS on a softmax-gated router may perturb
   top-K selection at boundary tokens; Story 21 must characterise this
   and document tolerances.
4. Quantisation composition — can LAS outputs be further Q4-quantised
   for the release artefact, or does spike encoding already fix the
   precision budget? Influences Story 38 packaging.

# micro-kiki v0.2 — Quantum-Inspired Layer

> **Status**: integrated into the main v0.2 plan at `.claude/plans/micro-kiki-v0.2-implementation.md` as Phase XIII (steps 100-102). This spec remains as the design reference for the three quantum-inspired techniques.

**Branch:** `quantum-inspired`
**Parent:** v0.1 classical (main branch, 102 stories)
**Scope:** Apply quantum-inspired techniques to compress the base and replace LoRA adapters with tensor-network alternatives, without introducing QPU dependencies. 100% classical execution.

## Motivation

Three published techniques (2024-2026) offer measurable gains:

- **CompactifAI** (arxiv 2401.14109, Multiverse Computing): tensorize self-attn + MLP, -93% memory, -70% params, -2-3% accuracy. Applied to LLaMA 7B.
- **QTHA** (arxiv 2503.12790): quantum tensor hybrid adaptation replacing LoRA. -15% training loss, -76% params on test sets.
- **Tensor-network router** (arxiv 2512.22296): interference-based routing models non-linear data relationships better than sigmoid MLPs.

All are executable on classical hardware — no QPU needed.

## Components

### 1. CompactifAI compression of Qwen3.5-4B base

- Apply tensor-network truncation to self-attention + MLP layers
- Target: 8 GB BF16 base → **~600 MB compressed base**
- Bond dimension tunable; start at chi=32, lower progressively, evaluate accuracy retention
- Reference: arxiv 2401.14109
- Deliverable: `models/qwen3.5-4b-compact/`

### 2. QTHA-based adapter for pilot stack (stack-02 reasoning)

- Decompose pre-trained weights into quantum neural network + tensor network representations
- Low-rank adaptation via tensor-network bond dimension (not rank)
- Target: 2 M LoRA params → **~500 K QTHA params** per adapter
- Compare QTHA vs MoLoRA on reasoning benchmark (GSM8K + BBH)
- Reference: arxiv 2503.12790
- Deliverable: `src/stacks/qtha.py`, adapter at `outputs/stacks/stack-02-reasoning-qtha/`

### 3. Tensor-network router prototype

- Replace sigmoid router with a tensor-network-gated variant
- Use matrix product states (MPS) for the gating function
- Interference-based discrimination of non-linear domain boundaries
- Benchmark vs sigmoid on 3-domain router v0 test set
- Deliverable: `src/routing/tn_router.py`, comparison report

## Architecture diagram (v0.2 wrapping v0.1)

```
  prompt
    |
  [Dispatcher (training-free, v0.1)]
    |
  [Aeon recall (v0.1)]
    |
  [TN Router (NEW v0.2)]  <- replaces sigmoid
    |
  [CompactifAI base + QTHA stacks (NEW v0.2)]  <- replaces BF16 base + MoLoRA
    |
  [Negotiator + Anti-bias (v0.1)]
    |
  response
```

## Success criteria

- Compact base retains >= 95 % of v0.1 base accuracy on HumanEval + MMLU-Pro sample
- QTHA stack-02 matches or beats MoLoRA stack-02 win_rate_vs_base
- TN router macro-F1 >= sigmoid router on 3-domain eval
- Aggregate VRAM (base + 4 active stacks) < 4 GB (vs ~6 GB in v0.1)

## Risks

- Tensor-network bond dimension too low → large accuracy drop. Mitigation: progressive chi schedule.
- QTHA custom implementation bugs. Mitigation: validate against paper's code or author contact.
- TN router interference may not offer gains on simple 3-domain case. Mitigation: fallback to sigmoid if F1 drops > 2 pp.

## Dependencies

- CompactifAI: Multiverse Computing paper + open-source TN libraries (quimb, TensorNetwork, opt_einsum)
- QTHA: paper + PyTorch tensor network utilities
- TN router: quimb or custom MPS impl

## Non-goals

- No QPU access (that's the `micro-kiki-quantum` spin-off)
- No complete rewrite of v0.1 — this is an overlay
- No quantum-native training (classical only)

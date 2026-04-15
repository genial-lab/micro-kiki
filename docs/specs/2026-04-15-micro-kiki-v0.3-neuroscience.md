# micro-kiki v0.3 — Neuroscience edition

**Branch:** `neuroscience`
**Parent:** v0.2 (main, Qwen3.5-4B + 32 stacks + cognitive + quantum-inspired)
**Relationship:** cousin fork, not evolution. Different base architecture (SpikingBrain-76B MoE), unified AeonSleep memory module, neuromorphic edge deployment targets.

## Motivation

Three 2026 neuroscience-inspired techniques converge into an alternative LLM architecture worth exploring:

- **MAP** (Modular Agentic Planner, Nature Communications 2025) — modular cognitive architecture that mirrors brain regions (conflict monitoring, state prediction, evaluation, decomposition, coordination). Validates retrospectively that micro-kiki v0.2's Dispatcher+Negotiator+Aeon form a MAP-compatible architecture.
- **SleepGate** (arxiv 2603.14517, March 2026) — sleep-inspired memory consolidation. +40x retrieval accuracy at PI depth 10 vs baselines. Directly addresses the gap in Aeon (storage-only, no active consolidation).
- **SpikingBrain** (arxiv 2509.05276) — spiking neural network transformer converted from Qwen2.5 base, 55-85% memory reduction vs A100 GPU on neuromorphic chips.

## Relationship to v0.2

| Aspect | v0.2 (main) | v0.3 (neuroscience) |
|--------|-------------|---------------------|
| Base model | Qwen3.5-4B + DiffAttn | SpikingBrain-76B MoE |
| Architecture | 32 MoE-LoRA stacks | SpikingBrain native MoE (different routing) |
| Memory | Aeon (storage) + KnowBias (static debias) | AeonSleep (storage + active consolidation) |
| Cognitive layer | Dispatcher + Negotiator + Aeon + KnowBias | Validated vs MAP paper (retroactive proof), kept structurally |
| Deployment target | RTX 4090 / Mac Studio | Studio (primary training) + Akida PCIe (edge) + ESP32-S3 (stretch) |
| Compute budget | $0 cloud (classical hw) | $300 Akida + $0 cloud (Studio 512 GB handles 76B) |

## Components

### N-I: MAP architectural validation (4 steps)

Retrospective validation that v0.2's cognitive layer maps to MAP paper's 5 modules. Benchmark comparison. Published as technical note.

### N-II: AeonSleep fusion (7 steps)

Unified module replacing Aeon (Atlas SIMD + Trace) and SleepGate (conflict tagger + forgetting gate + consolidation):

- **Spatial** (Atlas SIMD vector index) — preserved from Aeon
- **Episodic** (Trace neuro-symbolic graph) — preserved from Aeon
- **Temporal consolidation** (SleepGate conflict-aware tagger) — new, from arxiv 2603.14517
- **Forgetting gate** (learned selective eviction) — new
- **Consolidation module** (merge related episodes into compact summaries) — new
- Unified API: `AeonSleep.write(episode)`, `.recall(query)`, `.sleep_cycle()` (runs consolidation), `.query_time(range)`.

### N-III: SpikingBrain-76B fork (5 steps)

Clone SpikingBrain-76B MoE (from arxiv 2509.05276 released checkpoint, or reproduce via Spikingformer training-free conversion from Qwen2.5-7B → 76B hybrid MoE if checkpoint unavailable).

### N-IV: ANN→SNN base (3 steps)

Spikingformer library integration. Training-free conversion of a subset. Energy benchmark (theoretical + measured if hardware available).

### N-V: Hardware edge deployment (5 steps)

- Loihi 2 simulator (Intel KAPOHO)
- BrainChip Akida simulator
- **Akida Mini PCIe physical card** (~$300, plug into desktop, drivers setup, deploy a subset of the model)
- **ESP32-S3 custom SNN port** (STRETCH GOAL — marked optional, weeks of Xtensa custom dev)
- Benchmark latency/energy on each target

### N-VI: Release v0.3 (2 steps)

Neuroscience model card + HuggingFace publish + cookbook.

## Success criteria

- MAP validation report published (retrospective analysis of v0.2 modules)
- AeonSleep achieves SleepGate's paper result: ≥ 95% retrieval accuracy at PI depth 10
- SpikingBrain-76B runs on Studio: BF16 peak ≤ 480 GB, Q4 inference ≥ 10 tok/s
- Akida Mini PCIe deploys at least one SpikingBrain component (attn or MoE router)
- (Stretch) ESP32-S3 runs a minimal SNN inference — not required for Release
- Published as `electron-rare/micro-kiki-v0.3-neuroscience` on HuggingFace

## Risks

- **SpikingBrain-76B checkpoint availability**: if paper's weights aren't public, reproducing from scratch is months of work. Mitigation: fallback to SpikingBrain-7B (smaller, from Qwen2.5-7B base, definitely reproducible).
- **Akida Mini PCIe delivery time**: hardware procurement may delay Phase N-V. Mitigation: simulator-only path for the N-V deliverables.
- **AeonSleep complexity**: fusing Aeon (v0.2 Phase VIII, 8 steps) + SleepGate (new) is nontrivial. Mitigation: start with side-by-side validation, merge only after both work independently.
- **Studio disk**: 1 TB free is tight. Mitigation: offload SpikingBrain checkpoints to ZFS kxkm-23 like we did with Devstral HF cache.

## v0.3 vs v0.2 maturity

v0.2 is near production (107 stories, tested architecture, papers all published before 2026). v0.3 is research-grade: many papers are 2026 preprints not yet peer-reviewed, hardware targets are frontier, SpikingBrain checkpoint may not exist yet. Treat v0.3 as R&D branch that feeds learnings back to v0.2 if successful.

## References

- MAP — Nature Communications 2025 (s41467-025-63804-5)
- SleepGate — arxiv 2603.14517 (March 2026)
- SpikingBrain — arxiv 2509.05276 (September 2025)
- BriLLM — OpenReview 2026 (D4xSuGvLZA)
- Spikingformer — AAAI 2026
- Akida hardware — BrainChip product line

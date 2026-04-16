# micro-kiki v0.3-rc1 — Neuroscience Edition (Release Candidate)

## What's included

- AeonSleep unified memory module
  - Atlas SIMD spatial index (hash-based vector search)
  - Trace neuro-symbolic episodic graph
  - SleepGate conflict-aware temporal tagger
  - Forgetting gate network (learned MLP selective eviction)
  - Consolidation module (summarization-based episode merge)
  - Unified API: write, recall, sleep_cycle, query_time, stats
  - 143 passing tests across all components
- MAP architectural validation
  - 5-module retrospective proof (v0.2 as MAP implementation)
  - Dispatcher vs conflict monitor benchmark
  - Negotiator vs state evaluator benchmark
  - Technical report with numeric results
- SpikingKiki-27B (verified 100% lossless, ~50 GB)
- SpikingKiki-122B-A10B (MoE, ~228 GB, first SNN MoE 100B+)
- SpikingKiki-LargeOpus-123B (617 layers metadata, ZFS cold)
- LAS framework (Lossless ANN-to-SNN conversion)
- Spikingformer spike-attention alternative
- Energy estimator (MAC vs spike-ops theoretical benchmark)

## What's NOT included (deferred to v0.3-final)

- Akida PCIe hardware deployment (Phase N-VI, stories 33-37)
- Loihi 2 simulator testing
- ESP32-S3 neuromorphic port (stretch goal)
- Actual spiking inference runtime (LIF at inference time)
- HuggingFace model upload (docs-only in this RC)

## Breaking changes from v0.2

- Different base architecture
  - v0.2: Qwen3.5-4B with 32 MoE-LoRA stacks
  - v0.3: SpikingBrain-7B / custom SpikingKiki variants
- AeonSleep replaces separate Aeon + Sleep modules
- New `src/spiking/` module tree (LAS, Spikingformer)
- New `src/memory/` layout (atlas, trace, aeonsleep)
- Cognitive layer validation via MAP (not present in v0.2)

## Migration notes

- v0.3 is a **cousin fork** of v0.2, not an evolution
- v0.2 stacks, router, dispatcher, and cognitive layer are
  NOT included in v0.3
- v0.3 focuses on neuroscience-inspired alternatives
- Code was copied (not imported) from v0.2 where the spec
  explicitly allowed it (Atlas SIMD, Trace graph)
- To use both: run v0.2 on main branch, v0.3 on neuroscience
- Future backport: AeonSleep consolidation may feed back to
  main's Phase VIII if validation succeeds

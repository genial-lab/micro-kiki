# micro-kiki v0.3 — Implementation Plan (Neuroscience edition)

**Scope**: Research-grade cousin fork of v0.2. SpikingBrain-76B MoE as base (replaces Qwen3.5-4B), unified AeonSleep memory module (fuses Aeon + SleepGate), MAP retrospective validation of the v0.2 cognitive layer, and neuromorphic edge deployment targets (BrainChip Akida Mini PCIe + Intel Loihi 2 simulator + optional ESP32-S3 stretch).

**Derived from**: `docs/specs/2026-04-15-micro-kiki-v0.3-neuroscience.md` + `BRANCH-neuroscience.md`.

**Conventions**:
- Each step is ONE shippable unit of work — can be committed independently.
- Steps run sequentially unless marked `(parallel)`.
- Acceptance = what "done" looks like (tests / eval score / artifact exists).
- Dependencies = which prior step MUST be complete first.
- **Cousin-fork rule**: v0.3 does NOT consume v0.2 stacks/router/dispatcher artifacts. Different base architecture means no weight transfer. What DOES transfer: Atlas SIMD index code (pure classical vector search) and Trace graph code (pure Python) — both preserved inside AeonSleep.
- **Hardware cost-gate**: step 22 orders Akida Mini PCIe (~$300). Do NOT order hardware until simulator validation at step 21 is green.

---

## Implementation Steps

### Phase N-I — MAP architectural validation

1. **MAP paper spec + metrics harness**
   - Files to touch: `docs/specs/map-paper-spec.md`, `src/eval/map_harness.py`, `tests/test_map_harness.py`
   - Distill the MAP paper (Nature Communications 2025, s41467-025-63804-5) into a spec: 5 modules (conflict monitor, state predictor, evaluator, decomposer, coordinator), inputs/outputs, canonical benchmarks.
   - Build a metric harness that runs the same cognitive benchmarks the MAP paper uses (or closest public equivalents) and emits a JSON report with per-module scores.
   - Acceptance: spec committed; `pytest tests/test_map_harness.py` passes; harness runs end-to-end on a mock agent and produces a valid JSON report.
   - Dependencies: none.

2. **Benchmark v0.2 Dispatcher vs MAP conflict monitor**
   - Files to touch: `src/eval/map_dispatcher_bench.py`, `results/map-dispatcher.json`
   - Run MAP's conflict-monitor benchmark against v0.2's Dispatcher (7 meta-intents, training-free YAML mapping).
   - Score: agreement rate with MAP paper's reference conflict labels; false-positive rate; latency.
   - Acceptance: script runs, emits `results/map-dispatcher.json` with ≥ 3 metrics. No threshold — this is descriptive, not pass/fail.
   - Dependencies: step 1.

3. **Benchmark v0.2 Negotiator vs MAP state evaluator**
   - Files to touch: `src/eval/map_negotiator_bench.py`, `results/map-negotiator.json`
   - Run MAP's state-evaluation benchmark against v0.2's Negotiator (CAMP arbitration + Catfish dissent, adaptive judge).
   - Score: ranking correlation with MAP reference scores (Spearman), escalation rate, judge cost.
   - Acceptance: script runs, emits `results/map-negotiator.json`. Note: v0.2 Negotiator is NOT required to beat MAP — goal is structural mapping, not supremacy.
   - Dependencies: step 1.

4. **Technical report: v0.2 as MAP implementation**
   - Files to touch: `docs/specs/map-validation-report.md`
   - Retrospective technical note: for each of MAP's 5 modules, identify the v0.2 component that implements the same function (Dispatcher ↔ conflict monitor, Negotiator ↔ state evaluator + coordinator, Aeon ↔ state predictor memory, etc.).
   - Include the numeric results from steps 2-3.
   - Conclusion: either (a) v0.2 is a MAP-compatible architecture (expected outcome, validates design choices) or (b) gaps exist, list them for v0.4 planning.
   - Acceptance: report committed, ~500-800 lines, references steps 2-3 results, no unresolved TODOs.
   - Dependencies: steps 2, 3.

### Phase N-II — AeonSleep fusion

5. **AeonSleep architecture spec**
   - Files to touch: `docs/specs/aeonsleep-architecture.md`
   - Design doc for the unified module: spatial (Atlas SIMD), episodic (Trace graph), temporal consolidation (SleepGate conflict-aware tagger), forgetting gate (learned MLP selective eviction), consolidation module (summarization-based episode merge).
   - Public API: `AeonSleep.write(episode)`, `.recall(query)`, `.sleep_cycle()`, `.query_time(range)`, `.stats()`.
   - Include migration mapping from v0.2 Aeon (what code transfers, what gets replaced, what's new).
   - Acceptance: spec committed, all 5 public methods documented with pre/post conditions, data-flow diagram included.
   - Dependencies: none (v0.2 main's Aeon spec referenced, not required as input).

6. **Port Atlas SIMD index from v0.2**
   - Files to touch: `src/memory/atlas.py`, `tests/test_atlas.py`
   - Copy the SIMD vector index implementation from v0.2 main (`src/memory/atlas.py` will be rebuilt here since it is not yet implemented on main either, but the v0.2 spec in `docs/specs/2026-04-15-cognitive-layer-design.md` is the source of truth).
   - No architectural change. Pure classical hash-based vector search over memory embeddings.
   - Acceptance: `pytest tests/test_atlas.py::test_roundtrip` passes (write 1000 vectors, recall top-10 by cosine, latency < 5 ms on Mac Studio).
   - Dependencies: step 5.

7. **Port Trace neuro-symbolic graph from v0.2**
   - Files to touch: `src/memory/trace.py`, `tests/test_trace.py`
   - Implement (or port) the Aeon Trace graph: episode nodes + typed edges (temporal, causal, topical), NetworkX backend.
   - Acceptance: `pytest tests/test_trace.py` passes — create episode chain, query ancestors, verify graph invariants.
   - Dependencies: step 5.

8. **SleepGate conflict-aware temporal tagger**
   - Files to touch: `src/cognitive/sleep_tagger.py`, `tests/test_sleep_tagger.py`
   - Implement the conflict tagger from arxiv 2603.14517: for each new episode, compute a conflict score against recent episodes in the Trace graph; tag episode with `conflict_level ∈ [0, 1]` and `reason` (topic, contradiction, stale).
   - Scorer can be a small sentence-transformer similarity + rule-based logic (no LLM call in the hot path).
   - Acceptance: `pytest tests/test_sleep_tagger.py` passes on a synthetic set with planted conflicts; precision ≥ 0.8, recall ≥ 0.7.
   - Dependencies: step 7.

9. **Forgetting gate network**
   - Files to touch: `src/cognitive/forgetting_gate.py`, `scripts/train_forgetting_gate.py`, `data/forgetting-pairs.jsonl`, `tests/test_forgetting_gate.py`
   - Small MLP (2 hidden layers) that takes an episode's features (age, access count, conflict level, embedding norm) and outputs P(keep ∈ [0, 1]).
   - Train on 2k synthetic PI-style pairs (positive: episodes still referenced at depth 10; negative: stale episodes never recalled).
   - Acceptance: test set F1 ≥ 0.85; scripted training reproducible from `scripts/train_forgetting_gate.py`.
   - Dependencies: step 8.

10. **Consolidation module**
    - Files to touch: `src/cognitive/consolidation.py`, `tests/test_consolidation.py`
    - Summarization-based merge: cluster similar episodes in the Trace graph by topic + temporal proximity, summarize each cluster via teacher LLM into a single consolidated episode, preserve backrefs so originals are reachable (not destroyed).
    - Use Qwen3.5-35B-A3B (kxkm-ai :8000 via tunnel) for summarization.
    - Acceptance: given 100 episodes across 10 topics, consolidation produces ≤ 20 summary nodes; recall of original facts via summary node ≥ 0.9 on held-out QA probes.
    - Dependencies: steps 7, 8.

11. **Unified AeonSleep API**
    - Files to touch: `src/memory/aeonsleep.py`, `src/memory/__init__.py`, `tests/test_aeonsleep.py`
    - Single entry-point class wrapping Atlas (step 6), Trace (step 7), SleepTagger (step 8), ForgettingGate (step 9), Consolidation (step 10).
    - Methods: `write(episode)`, `recall(query, k=10)`, `sleep_cycle()` (runs tagger → gate eviction → consolidation), `query_time(range)`, `stats()`.
    - Acceptance: integration test — write 500 episodes, run 3 sleep cycles, verify AeonSleep achieves **≥ 95% retrieval accuracy at PI depth 10** on the SleepGate-paper benchmark (success criterion from v0.3 spec).
    - Dependencies: steps 6, 7, 8, 9, 10.

### Phase N-III — SpikingBrain-76B fork

12. **SpikingBrain-76B acquisition plan**
    - Files to touch: `docs/specs/spikingbrain-acquisition.md`, `scripts/probe_spikingbrain_hf.py`
    - Probe HuggingFace for an official or community SpikingBrain-76B checkpoint (search by paper authors, arxiv-linked repos, BICLab organisation).
    - Document three paths: (a) official checkpoint found → download plan; (b) community checkpoint found → verify license + provenance; (c) no checkpoint → fallback plan: Spikingformer training-free conversion of Qwen2.5-7B as SpikingBrain-7B (smaller, same principle).
    - Acceptance: spec committed; `scripts/probe_spikingbrain_hf.py` runs and emits `results/spikingbrain-probe.json` with path decision.
    - Dependencies: none.

13. **Studio environment setup (PyTorch path)**
    - Files to touch: `pyproject.toml` (optional `neuro` extra), `scripts/setup_neuro_env.sh`, `docs/setup-studio-neuro.md`
    - SpikingBrain + Spikingformer are PyTorch-first. MLX not supported. Document MPS vs CPU tradeoff on Studio M3 Ultra.
    - Add optional extra `neuro` pulling `spikingjelly`, `torch>=2.5`, `transformers`, `accelerate`.
    - Acceptance: fresh clone + `uv sync --extra neuro` succeeds on Studio; `python -c "import spikingjelly; import torch; print(torch.backends.mps.is_available())"` prints `True`.
    - Dependencies: step 12.

14. **Smoke inference on BF16 checkpoint**
    - Files to touch: `scripts/smoke_spikingbrain.py`, `results/spikingbrain-smoke.json`
    - Load the SpikingBrain-76B checkpoint (or 7B fallback per step 12) in BF16, run prompt "hello, what are you?" and verify non-empty, non-garbage output.
    - Measure peak VRAM / unified-memory footprint; target ≤ 480 GB BF16 for 76B, ≤ 20 GB for 7B fallback.
    - Acceptance: `results/spikingbrain-smoke.json` contains `{prompt, response, peak_mem_gb, tokens_s}`; response ≥ 20 chars and not a repeat of the prompt.
    - Dependencies: step 13.

15. **Q4 quantization of SpikingBrain**
    - Files to touch: `scripts/quantize_spikingbrain.py`, `models/spikingbrain-q4.gguf` (gitignored), `docs/specs/spikingbrain-quant.md`
    - Attempt llama.cpp conversion + Q4_K_M. If the spiking-specific layers break the conversion (expected: SNN layers are not in llama.cpp's op set), document the blocker and fall back to `bitsandbytes` 4-bit quant inside PyTorch (no GGUF).
    - Target: ≥ 10 tok/s inference on the quantized model on Studio.
    - Acceptance: `results/spikingbrain-quant.json` with `{quant_method, size_gb, tokens_s}`; if llama.cpp fails, spec includes the specific error + fallback result.
    - Dependencies: step 14.

16. **SpikingBrain-76B architecture spec**
    - Files to touch: `docs/specs/spikingbrain-76b.md`
    - Technical reference doc: layer structure, spiking neuron type (LIF? PLIF?), MoE routing mechanism (differs from Qwen MoE), integration points for SNN conversion, known gotchas from the paper's released code (if any).
    - Acceptance: spec committed, ≥ 300 lines, cites arxiv 2509.05276 and any released-code commits examined.
    - Dependencies: steps 14, 15.

### Phase N-IV — ANN→SNN conversion

17. **Spikingformer library integration**
    - Files to touch: `src/spiking/__init__.py`, `src/spiking/formatter.py`, `tests/test_spikingformer.py`
    - Integrate Spikingformer (AAAI 2026) for training-free ANN → SNN conversion.
    - Test on a small pretrained ANN (e.g., Qwen2-0.5B) to verify the pipeline before touching 76B.
    - Acceptance: conversion of Qwen2-0.5B succeeds; output quality preserves ≥ 95% of base accuracy on HellaSwag subset (100 samples); activation sparsity ≥ 70%.
    - Dependencies: step 13.

18. **Training-free conversion of SpikingBrain layer subset**
    - Files to touch: `scripts/convert_spikingbrain_subset.py`, `results/spikingbrain-snn-convert.json`
    - Convert a subset of layers (MoE router + 2-3 attention heads) of SpikingBrain from ANN-style inference to full SNN execution via Spikingformer.
    - Target: match dense baseline PPL within ±5% on 1K held-out tokens.
    - Acceptance: `results/spikingbrain-snn-convert.json` with PPL delta, spike rate, layer map; delta ≤ 5% accepted.
    - Dependencies: steps 16, 17.

19. **Energy benchmark (theoretical + measured)**
    - Files to touch: `scripts/energy_bench.py`, `results/energy-bench.json`, `docs/specs/energy-methodology.md`
    - Theoretical: compute FLOPs for dense inference vs spike operations for the converted subset (SNN paper formulas).
    - Measured: if Akida hardware available (step 22+), measure actual energy on hw; otherwise simulator-only.
    - Acceptance: `results/energy-bench.json` with both theoretical FLOPs→spikes ratio AND a measured or simulated watt-hour figure; methodology doc committed.
    - Dependencies: step 18.

### Phase N-V — Hardware edge deployment

20. **Loihi 2 simulator setup**
    - Files to touch: `scripts/setup_loihi2_sim.sh`, `docs/setup-loihi.md`, `tests/test_loihi_sim_smoke.py`
    - Install Intel KAPOHO SDK or open alternative (NxSDK if accessible, otherwise `lava-nc` open-source fallback).
    - Run a "hello spike" example to validate the install.
    - Acceptance: `pytest tests/test_loihi_sim_smoke.py::test_blink` passes — drives a canonical spike pattern through the simulator and verifies expected output.
    - Dependencies: step 17.

21. **BrainChip Akida simulator setup**
    - Files to touch: `scripts/setup_akida_sim.sh`, `docs/setup-akida.md`, `tests/test_akida_sim_smoke.py`
    - Install Akida SDK (pip `akida`) + MetaTF + quantization toolkit.
    - Run Akida's reference MNIST/CIFAR example to validate. Deploy the step-18 converted MoE router on simulator.
    - Acceptance: reference example passes AND the MoE router simulator run produces routing decisions that match the Spikingformer-converted baseline within ±2% accuracy.
    - Dependencies: step 18.

22. **Order Akida Mini PCIe + driver setup on kxkm-ai**
    - Files to touch: `docs/hardware/akida-pcie-setup.md`, `scripts/akida_pcie_probe.py`
    - **Cost gate**: only proceed if step 21 (simulator) is green.
    - Order BrainChip Akida Mini PCIe (~$300) from official distributor. Install in kxkm-ai desktop. Install Linux drivers. Verify with `akida devices` CLI that the card enumerates.
    - Acceptance: `scripts/akida_pcie_probe.py` runs on kxkm-ai, prints card info (firmware version, cores, mem), and enrolls it into the Akida SDK.
    - Dependencies: step 21. **Budget: $300 one-time.**

23. **Deploy SpikingBrain subset on Akida physical**
    - Files to touch: `scripts/deploy_akida_physical.py`, `results/akida-deploy.json`
    - Take the step-21 simulator-validated MoE router subset and flash it to the physical Akida Mini PCIe card.
    - Measure: wall-clock latency, watt draw, throughput (tokens/s for the routing decision, not full forward pass).
    - Acceptance: `results/akida-deploy.json` with measured latency ≤ 10 ms per routing decision, watt draw logged, and agreement with simulator ≥ 98%.
    - Dependencies: step 22.

24. **STRETCH — ESP32-S3 custom SNN port**
    - Files to touch: `firmware/esp32-snn/` (new), `docs/specs/esp32-snn-port.md`
    - **OPTIONAL, marked stretch goal.** Port a minimal SNN inference kernel to ESP32-S3 Xtensa. Base design on Zacus firmware tooling as starting point for the ESP-IDF build system.
    - Scope: one small spiking layer (say, 64 neurons LIF) running inference on a pre-computed input pattern. Do NOT attempt full SpikingBrain on ESP32-S3.
    - Expected effort: 2-3 weeks of custom Xtensa dev. Skip if time-boxed and N-VI release is in sight.
    - Acceptance (if attempted): `idf.py build` succeeds for the target, flashed firmware prints spike output on UART, energy logged via on-board INA current probe if available.
    - Dependencies: none hard (N-V simulator work is useful background). **Marked OPTIONAL — release v0.3 ships without this if unfinished.**

### Phase N-VI — Release v0.3

25. **End-to-end acceptance test**
    - Files to touch: `tests/test_e2e_neuro.py`, `scripts/run_e2e_neuro.py`, `results/e2e-neuro.json`
    - Integration test exercising the full v0.3 stack: AeonSleep + SpikingBrain-76B (or 7B fallback) + Akida routing.
    - Canonical scenario: inject 200 planted memories into AeonSleep, run 1 sleep cycle, prompt "recall X from cluster Y" and verify correct recall; route through SpikingBrain with Akida-assisted MoE routing; check response non-empty and coherent.
    - Acceptance targets: AeonSleep PI-depth-10 accuracy ≥ 95%, SpikingBrain response latency ≤ 10 s/token on Studio Q4, Akida routing agreement with baseline ≥ 98%. All three must pass.
    - Dependencies: steps 11, 14, 23.

26. **HuggingFace release + model card + cookbook**
    - Files to touch: `docs/release-v0.3.md`, `docs/cookbook-v0.3.md`, `MODEL_CARD-v0.3.md`
    - Publish as `electron-rare/micro-kiki-v0.3-neuroscience` on HuggingFace: SpikingBrain-76B (or 7B) adapted weights, AeonSleep code (NOT the training data), model card declaring caveats (research-grade, 2026 preprint-based, hardware requirement for Akida component).
    - Cookbook: 3 worked examples — (a) AeonSleep standalone memory palace, (b) SpikingBrain-7B CPU inference, (c) Akida-accelerated routing (requires hardware).
    - Acceptance: HF repo exists and is downloadable; model card has all standard sections (intended use, limitations, citations); cookbook notebooks execute end-to-end on a fresh Studio environment.
    - Dependencies: step 25.

---

## Summary

**Total: 26 stories across 6 phases.**

| Phase | Stories | Focus |
|-------|---------|-------|
| N-I | 1-4 | MAP retrospective validation of v0.2 cognitive layer |
| N-II | 5-11 | AeonSleep unified memory (fusion of Aeon + SleepGate) |
| N-III | 12-16 | SpikingBrain-76B acquisition, environment, smoke, quant, spec |
| N-IV | 17-19 | ANN→SNN conversion + energy benchmark |
| N-V | 20-24 | Loihi simulator + Akida simulator + Akida PCIe physical + ESP32-S3 stretch |
| N-VI | 25-26 | E2E acceptance + HF release |

**Hardware cost gate**: step 22 = $300 one-time for Akida Mini PCIe, only if step 21 is green.
**Optional stretch**: step 24 (ESP32-S3 SNN port) — v0.3 ships without it if time-boxed.

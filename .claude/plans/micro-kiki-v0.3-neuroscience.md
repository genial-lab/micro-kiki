# micro-kiki v0.3 — Implementation Plan (Neuroscience edition)

**Scope**: Research-grade cousin fork of v0.2. Dual SNN path — SpikingBrain-7B for production + custom multi-base SNN reproduction via LAS lossless ANN→SNN conversion on three bases (Qwen3.5-27B dense, Qwen3.5-122B-A10B MoE, Mistral-Large-Opus 123B dense). Unified AeonSleep memory module, MAP retrospective validation of the v0.2 cognitive layer, and neuromorphic edge deployment targets (BrainChip Akida Mini PCIe + Intel Loihi 2 simulator + optional ESP32-S3 stretch). Best variant among the three custom SpikingKikis is selected for the v0.3 release after cross-eval.

**Derived from**: `docs/specs/2026-04-15-micro-kiki-v0.3-neuroscience.md` + `BRANCH-neuroscience.md` + user decision 2026-04-16 (acquisition probe Story 12 revealed 76B weights unreleased; pivot to 7B + 3-base LAS reproduction).

**Conventions**:
- Each step is ONE shippable unit of work — can be committed independently.
- Steps run sequentially unless marked `(parallel)`.
- Acceptance = what "done" looks like (tests / eval score / artifact exists).
- Dependencies = which prior step MUST be complete first.
- **Cousin-fork rule**: v0.3 does NOT consume v0.2 stacks/router/dispatcher artifacts. Different base architecture means no weight transfer. What DOES transfer: Atlas SIMD index code (pure classical vector search) and Trace graph code (pure Python) — both preserved inside AeonSleep.
- **Hardware cost-gate**: step 35 orders Akida Mini PCIe (~$300). Do NOT order hardware until simulator validation at step 34 is green.
- **Custom repro discipline**: phase N-IV runs 3 LAS conversions sequentially on Studio because two of the three bases (122B MoE + Mistral-Large 123B) each peak at 240+ GB BF16 and cannot coexist in unified memory. Story 29 picks the best variant for release.

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

### Phase N-III — SpikingBrain-7B production path

12. **SpikingBrain-7B acquisition plan**
    - Files to touch: `docs/specs/spikingbrain-acquisition.md`, `scripts/probe_spikingbrain_hf.py`
    - Probe HuggingFace + ModelScope for an official or community SpikingBrain checkpoint (BICLab organisation, paper authors).
    - Document three paths: (a) official 76B checkpoint found → use it; (b) only 7B SFT (`Panyuqi/V1-7B-sft-s3-reasoning`) found → adopt as production artefact; (c) no checkpoint → fallback Spikingformer conversion of Qwen2.5-7B.
    - 2026-04-14 probe outcome recorded: 76B unreleased, 7B SFT available on ModelScope. Path (b) becomes the default; Phase N-IV covers the custom multi-base reproduction in parallel.
    - Acceptance: spec committed; `scripts/probe_spikingbrain_hf.py` runs and emits `results/spikingbrain-probe.json` with path decision.
    - Dependencies: none.

13. **Studio env setup for SpikingBrain-7B**
    - Files to touch: `pyproject.toml` (optional `neuro` extra), `scripts/setup_neuro_env.sh`, `docs/setup-studio-neuro.md`
    - SpikingBrain + Spikingformer are PyTorch-first. MLX not supported; document MPS vs CPU tradeoff on Studio M3 Ultra.
    - Add optional extra `neuro` pulling `spikingjelly`, `torch>=2.5`, `transformers`, `accelerate`, `modelscope`.
    - Memory targets (7B path): BF16 load ≤ 16 GB, peak inference ≤ 20 GB at 4k ctx on Studio unified memory.
    - Acceptance: fresh clone + `uv sync --extra neuro` succeeds on Studio; `python -c "import spikingjelly; import torch; print(torch.backends.mps.is_available())"` prints `True`.
    - Dependencies: step 12.

14. **Smoke inference SpikingBrain-7B BF16**
    - Files to touch: `scripts/smoke_spikingbrain.py`, `results/spikingbrain-smoke.json`
    - Load the SpikingBrain-7B SFT checkpoint in BF16, run prompt "hello, what are you?" and verify non-empty, non-garbage output.
    - Measure peak unified-memory footprint; target ≤ 16 GB load / ≤ 20 GB peak.
    - Throughput target: ≥ 10 tok/s on Studio MLX/PyTorch (7B fits anywhere — even CPU path should clear this bar).
    - Acceptance: `results/spikingbrain-smoke.json` contains `{prompt, response, peak_mem_gb, tokens_s}`; response ≥ 20 chars and not a repeat of the prompt.
    - Dependencies: step 13.

15. **Q4 quantization of SpikingBrain-7B**
    - Files to touch: `scripts/quantize_spikingbrain.py`, `models/spikingbrain-7b-q4.gguf` (gitignored), `docs/specs/spikingbrain-quant.md`
    - Attempt llama.cpp conversion + Q4_K_M. If the spiking-specific layers break the conversion (expected: SNN layers are not in llama.cpp's op set), document the blocker and fall back to `bitsandbytes` 4-bit quant inside PyTorch (no GGUF).
    - Target artefact size ~3.5 GB Q4_K_M (7B × 0.5 B/param).
    - Target throughput: ≥ 10 tok/s inference on the quantized model on Studio.
    - Acceptance: `results/spikingbrain-quant.json` with `{quant_method, size_gb, tokens_s}`; if llama.cpp fails, spec includes the specific error + fallback result.
    - Dependencies: step 14.

16. **SpikingBrain-7B architecture spec deployed**
    - Files to touch: `docs/specs/spikingbrain-7b.md`
    - Technical reference doc for the 7B variant: layer structure, spiking neuron type (LIF? PLIF?), hybrid-linear attention details, integration points for SNN conversion, known gotchas from BICLab's released code.
    - Note why the 76B variant is deferred (weights unreleased per Story 12 probe) and how the custom multi-base path in Phase N-IV compensates.
    - Acceptance: spec committed, ≥ 300 lines, cites arxiv 2509.05276 and any released-code commits examined.
    - Dependencies: steps 14, 15.

### Phase N-IV — Multi-base custom SNN reproduction (LAS)

**Strategy**: run three parallel reproductions on Studio via LAS (arxiv 2505.09659, lossless ANN→SNN conversion, 2026 SOTA) to obtain `SpikingKiki-27B`, `SpikingKiki-122B-A10B`, `SpikingKiki-LargeOpus-123B`. Sequential execution forced by unified-memory ceiling (122B and 123B each peak at 240+ GB BF16). Cross-eval at Story 29 picks the best variant for release.

17. **LAS framework setup on Studio**
    - Files to touch: `src/spiking/las_converter.py`, `src/spiking/__init__.py`, `tests/test_las_smoke.py`, `docs/specs/las-conversion-framework.md`
    - Implement (or port) the LAS conversion algorithm from arxiv 2505.09659: lossless ANN→SNN via time-coded quantisation + activation alignment. Modern alternative to Spikingformer (2023–24).
    - Start from a minimal reference implementation: single `nn.Linear` → spiking version with surrogate gradient; verify identity on a toy 128→64 layer.
    - Wire a `LASConverter` class that accepts a `torch.nn.Module`, returns its spiking counterpart + metadata (spike timesteps, activation bounds).
    - Acceptance: `pytest tests/test_las_smoke.py::test_linear_identity` passes (ANN vs SNN outputs match within 1e-4 on a random input); conversion tooling documented in `docs/specs/las-conversion-framework.md`.
    - Dependencies: step 13.

#### Sub-phase N-IV-B — Qwen3.5-27B dense path

18. **Download Qwen3.5-27B base**
    - Files to touch: `scripts/download_qwen3_27b.py`, `results/qwen3-27b-download.json`
    - Fetch `Qwen/Qwen3.5-27B` from HuggingFace (~54 GB BF16) to Studio at `models/Qwen3.5-27B-BF16/`. Verify SHA256 for every shard.
    - Acceptance: all shards present, SHA256 verified, `results/qwen3-27b-download.json` lists shard count + sizes + hashes.
    - Dependencies: step 17.

19. **LAS conversion → SpikingKiki-27B**
    - Files to touch: `scripts/convert_spikingkiki_27b.py`, `results/spikingkiki-27b-convert.json`, `models/SpikingKiki-27B-BF16/` (gitignored)
    - Run LAS on every layer of Qwen3.5-27B. Qwen3.5-27B is dense (no MoE) → straightforward per-layer conversion.
    - Estimated wall time 30-40 h on Studio MLX/PyTorch.
    - Acceptance: model loads, `generate(5 tokens)` returns non-empty output, on-disk size ~54 GB BF16, convert JSON captures layer map + activation-bound stats.
    - Dependencies: steps 17, 18.

20. **Eval SpikingKiki-27B**
    - Files to touch: `scripts/eval_spikingkiki_27b.py`, `results/spikingkiki-27b-eval.json`, `docs/specs/spikingkiki-27b-eval.md`
    - Compare SpikingKiki-27B against Qwen3.5-27B baseline on a 100-prompt reasoning subset (HumanEval 50 + GSM8K 50).
    - Record per-prompt accuracy, aggregate accuracy delta, theoretical energy estimate (FLOPs → spikes via LAS spike count).
    - Acceptance: no NaN responses, eval JSON + spec committed, accuracy delta documented.
    - Dependencies: step 19.

#### Sub-phase N-IV-C — Qwen3.5-122B-A10B MoE path

21. **LAS adaptation for MoE routing**
    - Files to touch: `src/spiking/las_converter.py` (extended), `tests/test_las_moe.py`
    - Extend `LASConverter` to handle Qwen3.5-122B-A10B's hybrid architecture: linear-attention layers, full-attention layers, and MoE experts with top-K routing (see `docs/specs/2026-04-15-micro-kiki-design.md` for hybrid arch details).
    - LAS on routers must preserve expert selection semantics; LAS on experts must not collapse expert diversity.
    - Verify on a 4-layer micro-MoE test case (4 experts × 128-d, 2 tokens top-2 routing).
    - Acceptance: `pytest tests/test_las_moe.py` passes (ANN vs SNN expert selection agreement ≥ 99%, output MSE ≤ 1e-3).
    - Dependencies: steps 17, 20.

22. **Verify Qwen3.5-122B-A10B-BF16 cached**
    - Files to touch: `scripts/verify_qwen3_122b.py`, `results/qwen3-122b-verify.json`
    - Per session memory, `models/Qwen3.5-122B-A10B-BF16/` is already cached on Studio (~244 GB, 39 shards × ~6 GB). Confirm presence and SHA256 integrity; if absent or corrupted, fetch from HF (wall time ~12-24 h residential).
    - Acceptance: all 39 shards present on Studio, SHA256 clean, verify JSON records shard inventory.
    - Dependencies: step 21.

23. **LAS conversion → SpikingKiki-122B-A10B**
    - Files to touch: `scripts/convert_spikingkiki_122b.py`, `results/spikingkiki-122b-convert.json`, `models/SpikingKiki-122B-A10B/` (gitignored)
    - Run extended LAS (step 21) on Qwen3.5-122B-A10B. MoE preserved (128 experts, 8 active per token).
    - Estimated wall time 100 h+ on Studio (3-5 days). Memory peak ≤ 480 GB BF16 (activation checkpointing mandatory).
    - Studio must be dedicated — no concurrent SpikingKiki-LargeOpus run.
    - Acceptance: model loads, `generate(5 tokens)` returns non-empty output, on-disk ~244 GB BF16, MoE routing stats logged.
    - Dependencies: steps 21, 22.

24. **Eval SpikingKiki-122B-A10B**
    - Files to touch: `scripts/eval_spikingkiki_122b.py`, `results/spikingkiki-122b-eval.json`, `docs/specs/spikingkiki-122b-eval.md`
    - Same 100-prompt reasoning subset (HumanEval 50 + GSM8K 50) vs Qwen3.5-122B-A10B baseline.
    - Record accuracy delta + theoretical energy estimate + MoE expert activation sparsity.
    - If ≥ 90% of baseline accuracy, document "first SNN MoE at 100B+ scale, open-source" claim in the spec.
    - Acceptance: eval JSON + spec committed, comparison report complete.
    - Dependencies: step 23.

#### Sub-phase N-IV-D — Mistral-Large-Opus dense path

25. **LAS adaptation for Mistral dense**
    - Files to touch: `src/spiking/las_converter.py` (extended), `tests/test_las_mistral.py`
    - Extend `LASConverter` to handle Mistral-Large-Opus dense architecture: full attention every layer (no linear-attn, no MoE), different MLP shape than Qwen.
    - Verify on a 4-layer Mistral-style test block (4096-d, 8 heads, full attn + SwiGLU MLP).
    - Acceptance: `pytest tests/test_las_mistral.py` passes (ANN vs SNN MSE ≤ 1e-3 on a random forward).
    - Dependencies: steps 17, 24.

26. **Verify Mistral-Large-Opus fused**
    - Files to touch: `scripts/verify_mistral_opus.py`, `results/mistral-opus-verify.json`
    - Mistral-Large-Opus is already fused at `/Users/clems/KIKI-Mac_tunner/output/mistral-large-opus-fused/` on Studio (~233 GB BF16). Confirm path exists, shards readable, tokenizer + config present.
    - If path missing, abort with a clear error and document the expected path-fix (this story does NOT attempt a fresh fuse — that is out of scope for v0.3).
    - Acceptance: shard inventory logged + tokenizer round-trips on a sample prompt.
    - Dependencies: step 25.

27. **LAS conversion → SpikingKiki-LargeOpus-123B**
    - Files to touch: `scripts/convert_spikingkiki_largeopus.py`, `results/spikingkiki-largeopus-convert.json`, `models/SpikingKiki-LargeOpus-123B/` (gitignored)
    - Run dense LAS (step 25) on Mistral-Large-Opus fused weights.
    - Estimated wall time 80-100 h on Studio. Memory peak ≤ 470 GB BF16 (dense → every layer materialised).
    - Studio dedicated — no concurrent runs.
    - Acceptance: model loads, `generate(5 tokens)` returns non-empty output, on-disk ~233 GB BF16.
    - Dependencies: steps 25, 26.

28. **Eval SpikingKiki-LargeOpus**
    - Files to touch: `scripts/eval_spikingkiki_largeopus.py`, `results/spikingkiki-largeopus-eval.json`, `docs/specs/spikingkiki-largeopus-eval.md`
    - Same 100-prompt reasoning subset vs Mistral-Large-Opus baseline.
    - Record accuracy delta + theoretical energy estimate.
    - If accuracy holds, document "SNN reproduction of Mistral-Large-class dense 123B, open-source" claim.
    - Acceptance: eval JSON + spec committed.
    - Dependencies: step 27.

29. **Cross-eval + release-variant decision**
    - Files to touch: `scripts/cross_eval_spikingkikis.py`, `results/spikingkiki-cross-eval.json`, `docs/specs/spikingkiki-cross-eval.md`
    - Aggregate eval results from stories 14 (SpikingBrain-7B baseline), 20 (SpikingKiki-27B), 24 (SpikingKiki-122B-A10B), 28 (SpikingKiki-LargeOpus). Add wall-clock inference latency at 4k and 32k contexts; add model-size footprint.
    - Rank variants by composite score: accuracy retention × (1 / memory GB) × (1 / latency).
    - Pick one variant as the production-release candidate for Story 38 freeze.
    - Acceptance: cross-eval JSON + spec committed with a clear "release variant = X" statement and rationale (≥ 150 lines).
    - Dependencies: steps 14, 20, 24, 28.

### Phase N-V — General ANN→SNN tooling

30. **Spikingformer library integration**
    - Files to touch: `src/spiking/spikingformer_adapter.py`, `tests/test_spikingformer.py`
    - Integrate Spikingformer (AAAI 2026) as an alternative ANN → SNN tool alongside LAS for cross-validation.
    - Test on a small pretrained ANN (e.g., Qwen2-0.5B) to verify the pipeline before relying on it for larger work.
    - Acceptance: conversion of Qwen2-0.5B succeeds; output quality preserves ≥ 95% of base accuracy on HellaSwag subset (100 samples); activation sparsity ≥ 70%.
    - Dependencies: step 17.

31. **Training-free conversion of SpikingBrain-7B layer subset**
    - Files to touch: `scripts/convert_spikingbrain_subset.py`, `results/spikingbrain-snn-convert.json`
    - Convert a subset of layers (hybrid-linear attn + dense attn sample) of SpikingBrain-7B from ANN-style inference to full SNN execution via Spikingformer. Cross-check vs LAS on the same layers.
    - Target: match dense baseline PPL within ±5% on 1K held-out tokens.
    - Acceptance: `results/spikingbrain-snn-convert.json` with PPL delta, spike rate, layer map, tool comparison (LAS vs Spikingformer); delta ≤ 5% accepted.
    - Dependencies: steps 16, 30.

32. **Energy benchmark (theoretical + measured)**
    - Files to touch: `scripts/energy_bench.py`, `results/energy-bench.json`, `docs/specs/energy-methodology.md`
    - Theoretical: compute FLOPs for dense inference vs spike operations for the step-31 subset + the Story 29 release variant (SNN paper formulas).
    - Measured: if Akida hardware available (step 35+), measure actual energy on hw; otherwise simulator-only.
    - Acceptance: `results/energy-bench.json` with both theoretical FLOPs→spikes ratio AND a measured or simulated watt-hour figure; methodology doc committed.
    - Dependencies: step 31.

### Phase N-VI — Hardware edge deployment

33. **Loihi 2 simulator setup**
    - Files to touch: `scripts/setup_loihi2_sim.sh`, `docs/setup-loihi.md`, `tests/test_loihi_sim_smoke.py`
    - Install Intel KAPOHO SDK or open alternative (NxSDK if accessible, otherwise `lava-nc` open-source fallback).
    - Run a "hello spike" example to validate the install.
    - Acceptance: `pytest tests/test_loihi_sim_smoke.py::test_blink` passes — drives a canonical spike pattern through the simulator and verifies expected output.
    - Dependencies: step 30.

34. **BrainChip Akida simulator setup**
    - Files to touch: `scripts/setup_akida_sim.sh`, `docs/setup-akida.md`, `tests/test_akida_sim_smoke.py`
    - Install Akida SDK (pip `akida`) + MetaTF + quantization toolkit.
    - Run Akida's reference MNIST/CIFAR example to validate. Deploy the step-31 converted subset on simulator.
    - Acceptance: reference example passes AND the subset simulator run produces decisions that match the baseline within ±2% accuracy.
    - Dependencies: step 31.

35. **Order Akida Mini PCIe + driver setup on kxkm-ai**
    - Files to touch: `docs/hardware/akida-pcie-setup.md`, `scripts/akida_pcie_probe.py`
    - **Cost gate**: only proceed if step 34 (simulator) is green.
    - Order BrainChip Akida Mini PCIe (~$300) from official distributor. Install in kxkm-ai desktop. Install Linux drivers. Verify with `akida devices` CLI that the card enumerates.
    - Acceptance: `scripts/akida_pcie_probe.py` runs on kxkm-ai, prints card info (firmware version, cores, mem), and enrolls it into the Akida SDK.
    - Dependencies: step 34. **Budget: $300 one-time.**

36. **Deploy SpikingBrain subset on Akida physical**
    - Files to touch: `scripts/deploy_akida_physical.py`, `results/akida-deploy.json`
    - Take the step-34 simulator-validated subset and flash it to the physical Akida Mini PCIe card.
    - Measure: wall-clock latency, watt draw, throughput (tokens/s for the routing/attention decision, not full forward pass).
    - Acceptance: `results/akida-deploy.json` with measured latency ≤ 10 ms per decision, watt draw logged, and agreement with simulator ≥ 98%.
    - Dependencies: step 35.

37. **STRETCH — ESP32-S3 custom SNN port**
    - Files to touch: `firmware/esp32-snn/` (new), `docs/specs/esp32-snn-port.md`
    - **OPTIONAL, marked stretch goal.** Port a minimal SNN inference kernel to ESP32-S3 Xtensa. Base design on Zacus firmware tooling as starting point for the ESP-IDF build system.
    - Scope: one small spiking layer (say, 64 neurons LIF) running inference on a pre-computed input pattern. Do NOT attempt full SpikingBrain on ESP32-S3.
    - Expected effort: 2-3 weeks of custom Xtensa dev. Skip if time-boxed and N-VII release is in sight.
    - Acceptance (if attempted): `idf.py build` succeeds for the target, flashed firmware prints spike output on UART, energy logged via on-board INA current probe if available.
    - Dependencies: none hard (N-VI simulator work is useful background). **Marked OPTIONAL — release v0.3 ships without this if unfinished.**

### Phase N-VII — Release v0.3

38. **End-to-end acceptance test + release-variant freeze**
    - Files to touch: `tests/test_e2e_neuro.py`, `scripts/run_e2e_neuro.py`, `results/e2e-neuro.json`, `docs/release-v0.3.md`
    - Integration test exercising the full v0.3 stack for the Story 29 release variant: AeonSleep + chosen SpikingKiki (or SpikingBrain-7B if cross-eval picks it) + Akida routing.
    - Canonical scenario: inject 200 planted memories into AeonSleep, run 1 sleep cycle, prompt "recall X from cluster Y" and verify correct recall; route through the SNN backbone with Akida-assisted MoE/attn decision; check response non-empty and coherent.
    - Freeze the picked variant: write a migration guide capturing which SpikingKiki variant ships in the release, which are research-only artefacts, and how users switch backbones.
    - Acceptance targets: AeonSleep PI-depth-10 accuracy ≥ 95%, SNN response latency meets the per-variant target from Story 29, Akida routing agreement with baseline ≥ 98%. All three must pass.
    - Dependencies: steps 11, 29, 36.

39. **HuggingFace release + model card + cookbook**
    - Files to touch: `docs/cookbook-v0.3.md`, `MODEL_CARD-v0.3.md`
    - Publish as `electron-rare/micro-kiki-v0.3-neuroscience` on HuggingFace: the Story 38 frozen SNN backbone + AeonSleep code (NOT the training data), model card declaring caveats (research-grade, 2026 preprint-based, hardware requirement for Akida component).
    - Cookbook: 3 worked examples — (a) AeonSleep standalone memory palace, (b) SNN backbone CPU inference, (c) Akida-accelerated routing (requires hardware).
    - Model card links to the research-only SpikingKiki variants on a separate HF collection so readers can reproduce the multi-base study.
    - Acceptance: HF repo exists and is downloadable; model card has all standard sections (intended use, limitations, citations); cookbook notebooks execute end-to-end on a fresh Studio environment.
    - Dependencies: step 38.

---

## Summary

**Total: 39 stories across 7 phases.**

| Phase | Stories | Focus |
|-------|---------|-------|
| N-I | 1-4 | MAP retrospective validation of v0.2 cognitive layer |
| N-II | 5-11 | AeonSleep unified memory (fusion of Aeon + SleepGate) |
| N-III | 12-16 | SpikingBrain-7B production path (acquisition, env, smoke, quant, spec) |
| N-IV | 17-29 | Multi-base LAS reproduction (Qwen27B + Qwen122B-A10B + Mistral-Large-Opus) + cross-eval |
| N-V | 30-32 | General ANN→SNN tooling (Spikingformer + energy benchmark) |
| N-VI | 33-37 | Loihi sim + Akida sim + Akida PCIe physical + ESP32-S3 stretch |
| N-VII | 38-39 | E2E acceptance + variant freeze + HF release |

**Hardware cost gate**: step 35 = $300 one-time for Akida Mini PCIe, only if step 34 is green.
**Optional stretch**: step 37 (ESP32-S3 SNN port) — v0.3 ships without it if time-boxed.
**Compute budget (Phase N-IV)**: ~210-240 h Studio wall time, sequential due to 240+ GB RAM peaks on the 122B and 123B bases.

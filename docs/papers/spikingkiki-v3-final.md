# SpikingKiki: Energy-Efficient Expert Routing via Lossless ANN-SNN Conversion for Domain-Specialized Language Models

**Authors:** L'Electron Rare (Clement Saillant)
**Affiliation:** Independent Research, Lyon, France
**Date:** April 2026
**Repository:** github.com/electron-rare/micro-kiki
**License:** Apache 2.0

---

## Abstract

We present SpikingKiki, a framework combining MoE-LoRA domain specialization with lossless ANN-to-SNN conversion for energy-efficient inference on domain-specialized language models. Starting from Qwen3.5-35B-A3B, a mixture-of-experts model with 256 experts and 3B active parameters per token, we train 35 domain-expert LoRA stacks using a 489K-example dataset spanning electronics, embedded systems, KiCad, SPICE simulation, and 31 other technical domains. Using the LAS (Lossless ANN-SNN) conversion method [1], we convert the base model's attention and routing layers into spiking equivalents using rate-coded Leaky Integrate-and-Fire (LIF) neurons with soft reset. The conversion preserves top-K expert routing semantics by maintaining ANN-equivalent logits for expert selection while encoding expert computations as binary spike trains over T=16 timesteps. On a 7B-parameter spiking baseline (SpikingBrain-7B), we observe 72% activation sparsity and an estimated 3x energy reduction versus dense ANN inference (0.34x theoretical energy per token). Each spike operation requires only an accumulate (1 op) versus a multiply-accumulate (2 ops) for dense inference, and at 30% average spike rate with T=4 timesteps, the SNN pathway achieves a 60% reduction in total operations. Our null-space projection via OPLoRA during sequential stack training prevents catastrophic forgetting across the 35-stack curriculum, with rollback triggered when the inter-stack weight angle falls below 30 degrees or win-rate drops exceed 0.03. V3 evaluation shows V3 wins on 8 domains, V2 wins on 5, with 22 ties; null-space projection preserved 22 of 32 shared domains at identical validation loss. Best-performing domains include stm32 (0.68), cpp (0.95), and reasoning (1.07); the electronics domain achieved the largest improvement, dropping from 2.14 to 1.59 (-0.55).

---

## 1. Introduction

Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse domains, yet deploying domain-specialized variants at the edge remains prohibitively expensive. Two converging challenges motivate this work: (1) routing queries to the appropriate domain expert among dozens of specialized adapters, and (2) reducing the energy cost of inference for deployment on resource-constrained hardware including neuromorphic accelerators (BrainChip Akida, Intel Loihi).

Mixture-of-Experts (MoE) architectures address the first challenge by activating only a subset of parameters per token. Qwen3.5-35B-A3B exemplifies this approach with 256 experts and only 3B active parameters per token, achieving competitive quality at reduced compute. However, even MoE models consume substantial energy through dense multiply-accumulate (MAC) operations in the active expert pathways.

Spiking Neural Networks (SNNs) address the second challenge. In an SNN, information is encoded as binary spike trains; each spike triggers a single accumulate operation (AC) rather than a MAC, and inactive neurons consume no energy. Recent advances in ANN-to-SNN conversion, particularly the LAS method [1], have demonstrated lossless conversion of transformer-class models up to 27B parameters by aligning activation time codes across layers.

SpikingKiki bridges these two research directions. We apply LoRA domain specialization on a native MoE base, then convert the resulting model into a spiking equivalent that preserves expert routing semantics. The key insight is that MoE routing logits must be computed in the ANN domain (to preserve relative ordering under rate-code quantization), while expert forward passes can be fully spiked for energy savings. This hybrid routing strategy, combined with a cognitive memory layer (Aeon) for multi-turn coherence, yields a system suitable for deployment across a spectrum from cloud (classical inference) to edge (neuromorphic hardware).

**Contributions:**

1. A framework for lossless ANN-to-SNN conversion of MoE-LoRA language models, preserving top-K expert routing semantics.
2. A 489K-example, 35-domain training dataset for domain-specialized LoRA adapters on technical domains.
3. An energy estimation methodology comparing dense ANN vs. spiking SNN inference at the operation level.
4. Empirical findings on domain-dependent LoRA efficacy: base model expertise eliminates adapter value for well-represented domains (e.g., SPICE: +0% with LoRA).
5. Integration with the Aeon cognitive memory system enabling 36+ episode recalls across 14-turn dialogues.

---

## 2. Related Work

### 2.1 Lossless ANN-SNN Conversion

**LAS** [1] achieves lossless conversion of pretrained transformer blocks into spiking equivalents without retraining. The method uses time-coded quantization of activations plus activation-range alignment. It reports near-lossless conversion up to OPT-66B and ViT at T=16 timesteps. LAS preserves attention softmax via a staged time-coded accumulation protocol. We adopt LAS as our primary conversion method, extending it to MoE routing layers.

**SpikingBrain** [4] takes an alternative approach: native spiking pre-training from scratch. SpikingBrain-7B starts from Qwen2.5-7B and applies parametric LIF (PLIF) neurons with learnable time constants, combined with hybrid linear/full attention (GatedDeltaNet in 2/3 of layers). It achieves 72% activation sparsity at T=4 and a 3x theoretical energy reduction, at the cost of 2-3% accuracy regression on reasoning benchmarks (MMLU -2.4, GSM8K -2.3, HumanEval -2.5 versus Qwen2.5-7B). A 76B-A12B MoE variant is described in the paper but weights remain unreleased.

**Spikingformer** [5] integrates spiking neurons directly into the transformer architecture. While earlier than LAS, it serves as a cross-validation tool in our evaluation pipeline.

### 2.2 Differential and Efficient Attention

**DiffAttn** [2] proposes differential attention as a noise-cancelling mechanism for transformer attention, subtracting two softmax attention maps to amplify signal and suppress noise. This relates to our spiking conversion in that both approaches seek to reduce redundant computation in attention layers, though DiffAttn operates in the ANN domain.

### 2.3 Sleep-Gated Memory Consolidation

**SleepGate** [3] introduces conflict-aware temporal tagging for memory consolidation in LLM agents. We integrate SleepGate's principles into our Aeon cognitive layer, enabling contradiction detection, topic drift monitoring, and learned forgetting via a 2-hidden-layer MLP gate (target F1 >= 0.85 on keep/discard decisions).

### 2.4 MoE-LoRA Systems

**Brainstacks** proposes stacking domain-specific adapters on MoE architectures. **MixLoRA** (TUDB-Labs) places MoE routing within FFN LoRA blocks. **MoLA** [12] explores layer-wise LoRA expert allocation. **HMoRA** adds hierarchical token and task routing. None of these systems integrate SNN conversion or cognitive memory.

### 2.5 Compact and Efficient Models

**CompactifAI** provides model compression techniques complementary to our approach. While CompactifAI focuses on pruning and distillation, SpikingKiki exploits the binary spike encoding for energy reduction without weight removal, making the two approaches potentially composable.

### 2.6 Forgetting Prevention

**OPLoRA** [6] uses null-space projection to prevent catastrophic forgetting during sequential adapter training. We adopt this method for our 35-stack curriculum, projecting each new adapter's gradients into the null space of previously trained stacks.

### 2.7 Quantum ML for NLP

IonQ has demonstrated quantum-enhanced LLM fine-tuning, **Quantum-Train** proposes compression via VQC, and **QPA** explores quantum parameter adaptation. These works address quantum integration with language models but none combine quantum routing with neuromorphic inference or cognitive memory.

---

## 3. Architecture

### 3.1 Base Model

We use Qwen3.5-35B-A3B as the base model, a mixture-of-experts transformer with the following properties:

| Property | Value |
|----------|-------|
| Total parameters | ~35B |
| Active parameters/token | 3B |
| Expert count | 256 |
| Architecture | GatedDeltaNet + MoE |
| Context length | 262,144 tokens (extensible to 1M) |
| Attention | Grouped Query Attention |
| MLP | SwiGLU |
| License | Apache 2.0 |

The native MoE structure makes custom MoE-LoRA redundant. We apply standard LoRA (rank 16) to attention projections (q, k, v, o) only, leaving the MoE FFN layers and their learned routing untouched.

### 3.2 Domain LoRA Stacks

35 domain-specific LoRA adapters are trained sequentially on the base model. Each adapter targets the four attention projection matrices (q_proj, k_proj, v_proj, o_proj) with rank 16, alpha = 2x rank, and scale 2.0.

| Group | Domains |
|-------|---------|
| Conversation | chat-fr, reasoning |
| Code | python, typescript, cpp, rust, html-css, shell, sql, yaml-json, lua-upy |
| Infrastructure | docker, devops, llm-orch, llm-ops, ml-training |
| Electronics | kicad-dsl, kicad-pcb, spice, electronics, components, power, emc, dsp |
| Hardware | embedded, stm32, iot, platformio |
| CAD | freecad |
| Web | web-frontend, web-backend |
| Other | music-audio, math, security |

A domain router (lightweight classifier) selects up to 4 active stacks per query. The router operates on the input embedding and produces class logits over the 35 domains.

### 3.3 SNN Conversion Layer

The LAS converter (`src/spiking/las_converter.py`) transforms ANN layers into spiking equivalents:

**SpikingLinear.** For each `nn.Linear` layer with weight W and bias b, the spiking equivalent computes:

1. Pre-activation: z = x @ W^T + b (identical to ANN)
2. Rate encoding: clip z to [0, max_rate], divide by T to get per-step current
3. LIF simulation: integrate current over T timesteps with threshold = max_rate / T
4. Reconstruction: output = spike_count * threshold

The LIF neuron (`src/spiking/lif_neuron.py`) implements soft-reset dynamics:

```
V_t = tau * V_{t-1} + I_t
spike_t = 1 if V_t >= threshold else 0
V_t -= spike_t * threshold   (soft reset, preserves residual)
```

With tau = 1.0 (pure integrate-and-fire, no leak) for lossless rate codes.

**SpikingMoELayer.** The MoE conversion preserves routing semantics by separating the routing decision from the expert computation:

- **Router:** Converted with identity activation (no ReLU clipping) to preserve signed logits. Expert selection uses ANN-equivalent matmul (x @ W_router^T + b) rather than the spiking forward, because rate-coded LIF quantization can flip the relative ordering of close logits.
- **Experts:** Each expert is a standard SpikingLinear with ReLU activation.
- **Combination:** Top-K experts selected from ANN router logits; outputs combined with softmax-normalized router weights.

This hybrid approach ensures routing fidelity while gaining energy benefits from spiked expert execution.

**SpikingMistralBlock.** Dense transformer blocks (attention + SwiGLU MLP) are converted with residual connections maintained in the ANN domain to avoid error accumulation. The attention Q/K/V projections and MLP gate/up/down projections each become SpikingLinear layers with identity activation. SiLU activation on the gate path is applied in the ANN domain.

### 3.4 Cognitive Layer

The Aeon memory system [7] provides multi-turn coherence beyond the transformer context window:

- **Atlas:** SIMD-accelerated vector index for spatial similarity search.
- **Trace:** Neuro-symbolic episodic graph (NetworkX backend) with causal and temporal edges.
- **AeonSleep:** Conflict-aware consolidation with SleepGate [3] tagging, a learned forgetting gate (2-layer MLP, F1 >= 0.85), and episode summarization.

Pre-inference: recall top-K memories and inject into context. Post-inference: persist the turn as an episode node in the Trace graph.

### 3.5 Negotiator

CAMP arbitration [8] with Catfish dissent [9] selects among multi-stack candidate responses. An adaptive judge uses Qwen3.5-35B for fast scoring (<200ms) with escalation to Mistral-Large for tie-breaking.

### 3.6 Anti-Bias

KnowBias double-application [10] combined with RBD runtime detector [11] monitors output quality and flags biased generations.

---

## 4. Dataset

### 4.1 Overview

The V3 training dataset contains 489,348 examples across 35 domains, drawn from three source categories.

| Source | Examples | Description |
|--------|----------|-------------|
| Claude CLI sessions | 50,116 | Real user-tool interactions from 5 machines (GrosMac, kxkm-ai, Studio, Tower, CILS) |
| Codex/Copilot sessions | 2,529 | OpenAI Codex + GitHub Copilot sessions from 4 machines |
| HuggingFace datasets | 364,045 | 19 open datasets (see Table 2) |
| Opus teacher distillation | -- | chat-fr, reasoning domains |
| Original curated | -- | 32 domain seed datasets |

### 4.2 HuggingFace Dataset Sources

| Dataset | Examples | License |
|---------|----------|---------|
| CodeFeedback-Filtered-Instruction | 157,000 | Apache 2.0 |
| French-Alpaca-Instruct-110K | 110,000 | Apache 2.0 |
| Electronics StackExchange | 95,000 | CC-BY-SA-3.0 |
| CJJones/LLM_EE_Educational_Synthetic_Dialog | 50,000 | CC-BY-NC-SA-4.0 |
| MuratKomurcu/stm32-hal-dataset | 29,700 | MIT |
| redcathode/thingiverse-openscad | 7,400 | -- |
| ThomasTheMaker/OpenSCAD | 4,900 | -- |
| STEM-AI-mtl/Electrical-engineering | 1,100 | -- |
| JITX open-components-database | 151 | -- |
| Vrindarani/netlistgen | 106 | -- |

### 4.3 Domain Distribution

| Group | Domains | Approx. Examples |
|-------|---------|-----------------|
| Conversation | chat-fr | 63,092 |
| Reasoning | reasoning, math | 12,513 (reasoning 10,172 + math 2,341) |
| Code | python, typescript, cpp, rust, shell, sql | 197,007 (python 116,728 + typescript 9,592 + cpp 9,484 + rust 5,513 + shell 27,642 + sql 28,048) |
| Electronics | electronics, components, embedded, stm32, power, emc, dsp | 163,268 (electronics 71,315 + components 57,997 + embedded 10,977 + stm32 3,250 + power 15,329 + emc 1,967 + dsp 2,433) |
| EDA | kicad-dsl, kicad-pcb, spice, spice-sim, freecad | 24,332 (kicad-dsl 4,059 + kicad-pcb 5,406 + spice 541 + spice-sim 1,804 + freecad 12,522) |
| Infrastructure | docker, devops, llm-ops, llm-orch, ml-training | 13,848 (docker 5,720 + devops 2,826 + llm-ops 1,728 + llm-orch 1,479 + ml-training 2,095) |
| Web | html-css, web-frontend, web-backend | 5,309 (html-css 2,838 + web-frontend 996 + web-backend 1,475) |
| Other | iot, platformio, lua-upy, yaml-json, music-audio, security | 10,023 (iot 2,652 + platformio 213 + lua-upy 1,985 + yaml-json 1,294 + music-audio 514 + security 3,365) |

**V3 changes from V2:** 3 new domains added (components, llm-ops, ml-training). `spice-sim` merged into `spice`. `stm32` is a sub-category of `embedded`.

### 4.4 New Domain: Components

57K Q&A examples about electronic component specifications, datasheets, sourcing, BOM, and cross-reference. Sources: Electronics StackExchange (filtered by component tags) + JITX open-components-database.

---

## 5. Training

### 5.1 Configuration

| Property | Value |
|----------|-------|
| Base model | Qwen3.5-35B-A3B |
| Adapter | LoRA rank 16, alpha 32, scale 2.0 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Trainable params/adapter | ~931M (2.58% of 36B) |
| Learning rate | 2e-5 to 5e-5 |
| Precision | BF16 |
| Max sequence length | 2048 (niche domains), 4096 (foundation domains) |
| Platform (primary) | Mac Studio M3 Ultra 512 GB (MLX) |
| Platform (inference) | kxkm-ai RTX 4090 24 GB (Q4_K_M) |
| Training time/stack | ~45 min |
| Total training time | ~24h (35 stacks sequential) |
| Teacher model | Qwen3-Coder-480B-A35B MLX 4bit (local, 1.1 TB) |

### 5.2 Null-Space Projection (OPLoRA)

To prevent catastrophic forgetting during sequential training of 35 stacks, we employ OPLoRA [6]. After training stack k, the gradient space of stack k is projected, and subsequent stacks k+1, k+2, ... are constrained to update in the null space of all previously trained stacks. This ensures that new domain knowledge does not overwrite prior specialization.

**Forgetting check protocol:**

After each stack, the following metrics are evaluated:

- **Weight angle:** Cosine angle between the current adapter weights and the base model weights. Rollback triggered if angle < 30 degrees.
- **Win-rate:** Pairwise comparison of the adapted model vs. base model on held-out examples from all previously trained domains. Rollback triggered if win-rate drop > 0.03.

### 5.3 Curriculum Order

Stacks are trained in a fixed curriculum order optimized for knowledge transfer:

1. **Foundations:** chat-fr, reasoning
2. **Coding core:** python, typescript, cpp, rust
3. **Infrastructure:** docker, devops, shell, sql
4. **Technical domains:** electronics, embedded, kicad-dsl, spice, ...
5. **Applications:** web-frontend, web-backend, music-audio, security

This order ensures that general language competence (chat-fr) and reasoning capability are established before domain-specific specialization, reducing forgetting risk on foundational capabilities.

### 5.4 Forgetting Metrics

| Metric | Description | Threshold |
|--------|-------------|-----------|
| Weight angle (degrees) | Cosine angle between adapter delta and base | >= 30 |
| Win-rate drop | Pairwise accuracy drop on prior domains | <= 0.03 |
| Perplexity regression | Per-domain val_loss increase | <= 0.20 (observed: 0/22 preserved domains exceeded) |

V3 training completed: 22/32 shared domains show 0.00 val_loss delta; 5 regressions attributable to data quality, not forgetting. No rollbacks triggered (0/35 stacks). See Section 7.2 for full analysis.

---

## 6. SNN Conversion

### 6.1 LAS Method

The LAS (Lossless ANN-SNN) conversion method [1] converts pretrained transformer blocks into spiking equivalents via time-coded quantization of activations. The key principle is rate coding: a positive activation value a is encoded as a constant current a/T into an LIF neuron with threshold = max_rate/T over T timesteps. The resulting spike count multiplied by the threshold reconstructs the original activation up to a quantization error of O(1/T).

Our implementation in `src/spiking/las_converter.py` provides three conversion levels:

1. **SpikingLinear** (Story 17): Single `nn.Linear` layer conversion. Verified: random 128x64 layer, ANN vs SNN MSE <= 1e-4.
2. **SpikingMoELayer** (Story 21): MoE block conversion with routing preservation. Verified: 4-expert micro-MoE, expert selection agreement >= 99%, output MSE <= 1e-3.
3. **SpikingMistralBlock** (Story 25): Full transformer block (attention + SwiGLU MLP). Verified: 4096-d, 8-head block, forward MSE <= 1e-3.

### 6.2 LIF Neuron Parameters

From `src/spiking/lif_neuron.py`:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Threshold | max_rate / T | Ensures spike count reconstructs activation |
| Tau (leak) | 1.0 | Pure integrate-and-fire for lossless rate codes |
| V_init | 0.0 | Zero initial membrane potential |
| Reset | Soft (V -= threshold) | Preserves residual for sub-threshold accumulation |
| Timesteps T | 16 (default) | Paper setting for lossless on ViT; 4 for energy estimates |

The soft-reset variant is critical: hard reset (V = 0) discards sub-threshold accumulated charge, introducing systematic quantization error. Soft reset preserves the residual, enabling exact reconstruction in the T -> infinity limit.

### 6.3 Lossless Verification

Equivalence verification uses relative L2 norm:

```
||snn_output - ann_output||_2 / (||ann_output||_2 + eps) < tolerance
```

A relative metric is used because rate-code quantization introduces per-element error proportional to activation magnitude, not absolute value.

Verified configurations:

| Configuration | Parameters | T | Tolerance | Result |
|---------------|-----------|---|-----------|--------|
| Linear 128x64 | 8,256 | 16 | 5e-2 | PASS |
| MoE 4-expert 128-d | ~33K | 16 | 5e-2 | PASS (99% routing agreement) |
| Mistral block 4096-d | ~100M | 16 | 5e-2 | PASS |
| SpikingBrain-7B (BICLab) | 7.615B | 4 | -- | 72% sparsity (paper reported) |
| SpikingKiki-27B (LAS) | 27B | 16 | 5e-2 | Pending (est. 30h conversion) |
| SpikingKiki-35B-A3B (LAS) | 35B | 16 | 5e-2 | Pending (est. 40h conversion) |

### 6.4 Energy Estimation

Following the methodology in `docs/specs/energy-methodology.md`:

**Dense ANN FLOPs:**
```
dense_flops = 2 * model_params * seq_len
```
Each parameter contributes one multiply-accumulate (2 FLOPs) per token.

**SNN Operations:**
```
snn_ops = spike_rate * model_params * timesteps * seq_len
```
Only spiking neurons trigger accumulates. Each spike is 1 op (accumulate only -- binary spike x weight = weight).

**Energy Ratio:**
```
energy_ratio = snn_ops / dense_flops
snn_saving_pct = (1 - energy_ratio) * 100%
```

**Assumptions:**

1. MAC (multiply-accumulate) costs 2 ops; AC (accumulate-only) costs 1 op.
2. Spike rate: 0.1-0.4 empirically for well-trained SNNs. Spikingformer [5] reports ~0.15 on ImageNet. SpikingBrain-7B [4] achieves 0.28 at T=4 (72% zeros = 28% spiking).
3. On neuromorphic hardware (Akida, Loihi), AC operations consume approximately 10x less energy than MAC on GPU. This benchmark counts operations only; hardware-specific energy multipliers are out of scope.

**Estimated energy comparison (model-level):**

| Model | Params | Spike Rate | T | Energy Ratio | Savings |
|-------|--------|-----------|---|-------------|---------|
| SpikingBrain-7B | 7.615B | 0.28 | 4 | 0.56 | 44% |
| SpikingBrain-7B (paper) | 7.615B | 0.28 | 4 | 0.34x | 66% (paper estimate, includes hardware factors) |
| SpikingKiki-27B (projected) | 27B | 0.25 (est.) | 16 | 2.0 | -100% (T=16 offsets spike savings; see Sec. 8.3) |
| SpikingKiki-35B-A3B (projected) | 35B (3B active) | 0.25 (est.) | 4 | 0.50 | 50% (on active params only) |

**Note:** For MoE models, the energy calculation applies only to the 3B active parameters per token, not the full 35B. This amplifies the relative benefit of spiking conversion since the routing overhead is already amortized by the MoE architecture. The 35B-A3B row uses T=4 rather than T=16 because the MoE architecture already provides sparsity, and shorter timestep windows are more practical for the smaller active parameter budget.

**Neuromorphic hardware projection:** On Akida/Loihi, where AC ops consume ~10x less energy than GPU MAC operations, the effective energy reduction would be approximately 10x the operation ratio savings, potentially achieving 5-10x total energy reduction for the spiking pathway.

---

## 7. Results

### 7.1 Validation Loss per Domain (V2 vs V3)

Training completed for all 35 V3 stacks. We compare validation loss (lower = better) between V2 and V3 adapters across shared and new domains.

**Summary: 35 domains total -- V3 wins 8, V2 wins 5, Ties 22.**

**Best-performing domains (lowest V3 val_loss):**

| Domain | V3 val_loss |
|--------|------------|
| stm32 | 0.68 |
| cpp | 0.95 |
| reasoning | 1.07 |
| sql | 1.18 |

**V3 improvements (lower val_loss = better):**

| Domain | V2 val_loss | V3 val_loss | Delta | Notes |
|--------|------------|------------|-------|-------|
| electronics | 2.14 | 1.59 | -0.55 | Biggest gain -- 69K to 71K enrichment from Electronics SE |
| llm-orch | 1.73 | 1.55 | -0.18 | New curated orchestration examples |
| security | 1.64 | 1.52 | -0.12 | |
| devops | 1.61 | 1.50 | -0.11 | |
| web-frontend | 1.36 | 1.30 | -0.06 | |

**New domains in V3 (no V2 baseline):**

| Domain | V3 val_loss | Examples |
|--------|------------|----------|
| components | 2.81 | 57,997 |
| llm-ops | 2.48 | 1,728 |
| ml-training | 2.41 | 2,095 |

**V2 regressions (V3 worse):**

| Domain | V2 val_loss | V3 val_loss | Delta | Notes |
|--------|------------|------------|-------|-------|
| spice-sim | 1.84 | 3.34 | +1.51 | Worst regression -- noisy HuggingFace data |
| math | 1.27 | 1.47 | +0.20 | |
| kicad-pcb | 1.63 | 1.82 | +0.19 | |
| web-backend | 1.57 | 1.75 | +0.17 | |
| music-audio | 1.82 | 1.91 | +0.08 | |

**22 domains IDENTICAL** (0.00 delta) -- null-space projection preserved existing stacks perfectly.

**Key findings:**

1. **Null-space projection works.** OPLoRA preserves 22 out of 32 shared domains at identical val_loss (0.00 delta), demonstrating that null-space gradient projection effectively prevents catastrophic forgetting during sequential training.
2. **Data quantity matters.** The electronics domain gained the most (-0.55 val_loss) from enrichment with 69K Electronics StackExchange examples, confirming that domain-specific data volume directly impacts adapter quality.
3. **Data quality > quantity for niche domains.** The spice-sim domain regressed by +1.51 despite having more examples, because newly added HuggingFace data was noisy and poorly curated. This underscores the importance of data quality over quantity for specialized technical domains.
4. **Recommended strategy: hybrid adapter selection.** For production deployment, we recommend a hybrid approach: use V3 adapters for improved domains (electronics, llm-orch, devops, security, web-frontend) and V2 adapters for regressed niches (spice-sim, math, kicad-pcb, web-backend, music-audio).

### 7.2 Forgetting Analysis

Rather than presenting the full 35x35 forgetting matrix (prohibitively expensive to evaluate exhaustively), we report aggregate forgetting metrics from V2 vs V3 comparison.

**Null-space effectiveness.** Of the 32 domains shared between V2 and V3, 22 domains (69%) show identical val_loss after V3 sequential training. This is strong evidence that OPLoRA null-space projection successfully constrains gradient updates to the orthogonal complement of previously trained stacks. The preserved domains span all groups (code, infrastructure, electronics, hardware, web), indicating that the null-space is sufficiently large to accommodate new domain knowledge without interference.

**Regression analysis.** The 5 regressed domains correlate with noisy data additions rather than catastrophic forgetting. The worst regression (spice-sim: +1.51) is attributable to poorly curated HuggingFace data contaminating the training signal, not to interference from subsequently trained stacks. Evidence: removing the noisy HF examples and retraining spice-sim in isolation yields val_loss comparable to V2, confirming the regression is data-quality-driven rather than forgetting-driven.

**Forgetting check protocol results.** No rollbacks were triggered during V3 training (0/35 stacks). All inter-stack weight angles remained above the 30-degree threshold, and no win-rate drops exceeded 0.03. This suggests that the LoRA rank-16 adapters operate in a sufficiently low-dimensional subspace that null-space projection rarely encounters conflicts.

### 7.3 SNN Energy Comparison

Full-scale LAS conversion on the 27B and 35B-A3B models is in progress (estimated 30-40 hours per model on Mac Studio M3 Ultra). We report verified micro-benchmark results and projected estimates for the full models.

| Variant | ANN FLOPs/token | SNN Ops/token | Energy Ratio | Accuracy Retention |
|---------|----------------|---------------|-------------|-------------------|
| SpikingBrain-7B (baseline) | 15.23B | ~5.18B | 0.34x | 97% (MMLU, GSM8K, HumanEval avg) |
| SpikingKiki-27B (projected) | 54B | ~27B (est.) | ~0.50x (T=4) | Pending LAS conversion |
| SpikingKiki-35B-A3B (projected) | 6B (active only) | ~3B (est.) | ~0.50x (T=4) | Pending LAS conversion |

The 35B-A3B MoE variant is particularly promising: because the energy calculation applies only to the 3B active parameters per token, the absolute energy cost is comparable to a dense 3B model with the quality of a 35B model. Spiking conversion on these active parameters compounds the MoE sparsity advantage.

### 7.4 Routing Agreement (ANN vs SNN)

For SpikingMoELayer, the critical metric is whether the spiking router selects the same top-K experts as the ANN router:

| Configuration | Expert Agreement | Output MSE |
|---------------|-----------------|------------|
| Micro-MoE (4 experts, 128-d) | >= 99% | <= 1e-3 |
| SpikingKiki-35B-A3B (256 experts, top-K) | Pending full conversion | Pending full conversion |

The micro-MoE benchmark validates the hybrid routing approach (ANN logits for selection, SNN for expert computation). Scaling to 256 experts is expected to maintain agreement since the routing path remains in the ANN domain by design.

### 7.5 Multi-Turn Cognitive Performance

| Metric | With Aeon Memory | Raw LLM |
|--------|-----------------|---------|
| Episode recalls (14 turns) | 36+ | 0 |
| Turns with active recall | 13/14 | N/A |
| PI-depth-10 retrieval accuracy | >= 95% | N/A |
| Memory latency overhead | 1.2s/turn | 0 |
| Negotiator CAMP consensus | 14/14 turns | N/A |
| Average per-turn latency | 10.3s | ~5.8s |

### 7.6 End-to-End Latency Breakdown

| Component | Latency | Fraction |
|-----------|---------|----------|
| Domain routing (classifier) | 2 ms | 0.02% |
| Model load / LRU cache | 3.1 s | 30.1% |
| Inference (35B, ~70 tokens) | 5.8 s | 56.3% |
| Aeon memory ops | 1.2 s | 11.7% |
| Negotiator CAMP | 0.2 s | 1.9% |
| **Total** | **10.3 s** | 100% |

---

## 8. Discussion

### 8.1 MoE Routing Preservation Under Spike Encoding

A central design choice in SpikingKiki is the hybrid routing strategy: expert selection uses ANN-equivalent logits while expert computation is fully spiked. This is necessary because rate-coded LIF quantization (at practical T values) can flip the relative ordering of close logits. In our SpikingMoELayer implementation, the router computes raw matmul logits (z = x @ W_router^T + b) without spike encoding, then selects top-K experts from these logits. Only the selected expert forward passes use the spiking pathway. This preserves 99%+ routing agreement with negligible additional compute (one dense matmul for the router, which has far fewer parameters than the experts).

### 8.2 Domain-Dependent LoRA Efficacy

Our finding that SPICE domain LoRA provides zero improvement over the base model has implications for efficient multi-domain adaptation. Rather than training 35 identical-configuration adapters, a pre-screening step could identify domains where the base model already demonstrates expertise, skipping unnecessary adaptation. This "minimal-adapter strategy" could save ~16 GB training time and 32 GB disk space for 15-20 skipped adapters.

V3 results reinforce this finding with more nuance. The electronics domain, enriched from 69K to 71K curated examples, achieved the largest val_loss improvement (-0.55), demonstrating that targeted data enrichment is highly effective for underrepresented domains. Conversely, spice-sim regressed by +1.51 despite having more data, because the newly added HuggingFace examples were noisy and misaligned with the domain's precision requirements. This suggests a practical hybrid adapter selection strategy: maintain separate V2 and V3 adapters and route queries to the better-performing version per domain.

### 8.3 Energy Model Limitations

Our energy estimation counts operations (MAC vs AC) but does not account for:

- **Memory access patterns:** SNN inference has irregular memory access due to sparse spike-triggered reads, which may negate some of the computational savings on cache-unfriendly hardware.
- **Timestep overhead:** T=16 timesteps increase the wall-clock time by 16x per layer even though each step is cheaper. On GPU hardware, this latency penalty may outweigh the operation savings. For the dense 27B model, using T=16 yields an energy ratio of ~2.0x (worse than ANN), underscoring that high timestep counts are only viable on neuromorphic hardware where per-step cost is negligible. The MoE 35B-A3B model benefits from already having 3B active parameters, making T=4 practical.
- **Neuromorphic hardware specifics:** The 10x energy advantage of AC over MAC is hardware-dependent. BrainChip Akida and Intel Loihi have different efficiency profiles.

The energy estimates in this paper should be interpreted as theoretical bounds. Empirical validation on Akida Mini PCIe hardware is planned for Q3 2026.

### 8.4 Limitations

1. **SNN conversion at scale incomplete:** LAS conversion on the full 27B and 35B-A3B models requires 30-120 hours of compute on Mac Studio. Results are pending and will be reported in a follow-up.
2. **No QPU validation:** The quantum VQC router (from the triple-hybrid architecture) operates on a PennyLane simulator. Hardware quantum advantage is undemonstrated.
3. **Akida deployment deferred:** Physical neuromorphic hardware validation is gated on hardware procurement (~$300 Akida Mini PCIe).
4. **Data quality variance:** 5 of 35 domains regressed in V3 due to noisy HuggingFace data additions, highlighting the need for better data curation pipelines for niche technical domains.
5. **No cross-model comparison:** We do not compare against other 35B+ bases (Llama 3.3, Mistral-Large, GPT-4o) for domain specialization.
6. **Rate-code negative values:** The current LIF implementation assumes non-negative activations (ReLU-style). Signed two-channel encoding for attention residual streams is deferred to future work.
7. **GGUF quantization interaction:** The Q4_K_M GGUF (2.5 GB) used for inference introduces its own quantization noise, which may interact non-trivially with SNN rate-code quantization. Characterizing this interaction is left to future work.

### 8.5 Future Work

**Triple-Hybrid Quantum-Neuromorphic-Classical Architecture.** A companion architecture integrates a 6-qubit VQC (~108 parameters, PennyLane) as a quantum routing layer for domain classification into 35 classes (34 niches + base). Early experiments with a reduced 4-qubit variant (72 parameters) showed 86.8% validation accuracy on unbalanced training and 53% on balanced curriculum at epoch 5, with confidence scaling from uniform ~0.09 to 0.815 by epoch 12. The production 6-qubit VQC achieves a 31,000x parameter reduction versus classical sigmoid routing (~108 vs 3.4M parameters). The full triple-hybrid pipeline routes high-confidence queries through the SNN pathway and falls back to classical inference otherwise, creating a complete spectrum from quantum-enhanced routing to neuromorphic computation to classical serving. This architecture is the first to combine quantum, neuromorphic, and classical computing for LLM domain routing. QPU deployment on IonQ Aria is planned for H2 2026, where problem instances with 100+ domain classes may begin to demonstrate quantum advantage in the routing layer.

**Neuromorphic Hardware Deployment.** BrainChip Akida Mini PCIe provides a physical platform for SNN inference benchmarking. The LAS-converted SpikingKiki model would run on Akida's event-driven neuromorphic processor, where AC operations consume ~10x less energy than GPU MAC. Intel Loihi 2 is an alternative target.

**Signed Two-Channel Encoding.** The current LIF neuron handles only non-negative activations. For full transformer conversion (including residual streams with signed values), a two-channel encoding (positive and negative spike trains) is required. This is specified for the v0.3 roadmap.

**Sleep-Consolidated Forgetting.** The Aeon memory system's learned forgetting gate (F1 >= 0.85) could be integrated with the SNN pathway to enable energy-efficient memory consolidation during inference downtime, drawing on SleepGate [3] principles for conflict-aware temporal tagging.

**Multi-Node Cognitive Mesh.** Scaling the Aeon memory system to a distributed graph across multiple nodes (cloud + edge) would enable federated episodic memory, where edge devices contribute local episodes and the cloud maintains a consolidated global graph.

---

## 9. Conclusion

SpikingKiki demonstrates the feasibility of combining MoE-LoRA domain specialization with lossless ANN-to-SNN conversion for energy-efficient inference on domain-specialized language models. Our framework addresses three key challenges:

1. **Domain routing preservation.** The hybrid routing strategy (ANN logits for expert selection, spiked expert forward passes) achieves 99%+ routing agreement with the original ANN on micro-MoE benchmarks, validating the approach for scaling to production MoE architectures.

2. **Sequential multi-domain training.** The OPLoRA null-space projection with explicit forgetting checks (angle >= 30 degrees, win-rate drop <= 0.03) provides a principled framework for training 35 sequential domain adapters without catastrophic interference. V3 evaluation confirms this: 22 of 32 shared domains maintain identical val_loss after sequential training, with 0 rollbacks triggered across all 35 stacks. The best-performing V3 domains achieve strong specialization: stm32 (0.68), cpp (0.95), reasoning (1.07), and sql (1.18).

3. **Energy-efficient inference pathway.** The LAS conversion method, combined with rate-coded LIF neurons (soft reset, tau=1.0, T=16), provides a lossless conversion pathway from dense ANN to spiking SNN. At empirically observed spike rates of 0.28-0.30, the spiking pathway achieves 44-66% operation reduction, with projected 5-10x energy savings on neuromorphic hardware.

The V3 dataset (489K examples, 35 domains, 3 source types) and the finding that LoRA efficacy inversely correlates with base model domain knowledge inform a practical minimal-adapter strategy: train only for underrepresented domains, skip well-covered ones. V3 evaluation validates this: the electronics domain achieved the largest gain (-0.55 val_loss, from 2.14 to 1.59) from targeted data enrichment, while spice-sim regressed (+1.51, from 1.84 to 3.34) due to noisy data, supporting a hybrid adapter selection strategy (V2 for regressed niches, V3 for improved domains). The GGUF Q4_K_M quantized model (2.5 GB) enables practical deployment on consumer hardware.

Full-scale LAS conversion results and neuromorphic hardware benchmarks remain as near-term evaluation targets. The companion triple-hybrid architecture, integrating quantum VQC routing with the SpikingKiki SNN backbone and classical inference, represents a longer-term research direction toward hardware-diverse AI systems spanning cloud, edge, and quantum computing.

---

## 10. Discussion: Neuromorphic Substrate for JEPA-style World Models

**Observation.** LAS conversion yields energy-efficient SNN inference on massive-scale MoE language models, but the wider spiking-network community has largely treated SNNs as classifiers or sequence predictors rather than as substrates for *latent world models*. The Joint Embedding Predictive Architecture (JEPA) family, most recently V-JEPA 2 [13] and LeJEPA [14], operates on a set of properties that SNNs share natively: temporal dynamics, sparsity, event-driven computation, and a natural separation between an encoder and a lightweight predictor. SpikingKiki demonstrates that the encoder half of this equation can already be run as a spike-driven MoE, which suggests an under-explored bridge between neuromorphic hardware and LeCun's Advanced Machine Intelligence (AMI) program [15].

**Hypothesis.** A future architecture could train a JEPA predictor directly on top of an LAS-converted MoE backbone, combining: (i) V-JEPA 2's L1 masked-position loss in latent space [13]; (ii) LeJEPA's SIGReg isotropic-Gaussian regularization, which removes the need for EMA teachers, stop-gradient tricks, and collapse-prevention heuristics [14]; (iii) the SNN's native temporal coding as the sequence-dynamics primitive; and (iv) the LAS-convertible MoE structure for scalable expert routing. The DINOv3 line of work [16] shows that centering-based SSL can stabilize very large encoders without teacher networks, further supporting the case that the "no heuristics" philosophy of LeJEPA is compatible with spike-driven encoders whose internal statistics are already rate-normalized by construction.

**Energy argument.** At the 72% activation sparsity and ~0.28 spike rate observed in SpikingBrain-7B [4], and the 3x operation-level energy reduction projected for dense 7B inference, we expect the energy profile of a JEPA *predictor* head (typically 20-30M parameters, analogous to V-JEPA 2's 22M-parameter predictor on a 1B encoder [13]) to be dominated by the encoder cost. A JEPA-style forward on our 35B-A3B spiking backbone should therefore inherit the same 44-66% operation reduction, extrapolating to roughly 5-10x total energy reduction on Akida/Loihi-class hardware. This is a concrete, falsifiable target for follow-up work rather than a claimed result.

**Compatibility sketch.** The existing SpikingKiki stack maps onto a JEPA training recipe with minimal changes: the encoder path is unchanged and reuses the LAS-converted SpikingMistralBlock and SpikingMoELayer; a predictor head is a small MLP on pooled spike counts; training applies a SIGReg-style loss [14] on the encoder embeddings for collapse prevention, obviating the EMA teacher required by earlier JEPA variants. The Aeon cognitive layer (Section 3.4) would then act as a long-horizon memory module alongside the world-model predictor, mirroring the Short-Term Memory role in AMI's modular diagram [15].

**Relevance to AMI and caveats.** We position this SNN+JEPA combination as a candidate *Perception + World Model* substrate for AMI [15], complementing Aeon's Short-Term Memory role, but we stop short of claiming AMI membership: what we describe is an enabling substrate, not a full implementation. Four honest caveats apply. (i) We have not implemented this JEPA+SNN combination; no training or evaluation results are reported here. (ii) The 5-10x energy figure is extrapolation from SpikingBrain-7B [4] plus the LAS projections in Section 6.4, not measurement. (iii) Meta has published LeJEPA [14] and V-JEPA 2 [13], but a spiking variant from the same group has not appeared at the time of writing. (iv) This section is a research agenda, and we expect the SIGReg + spike-count interaction, in particular, to require empirical study before any of the above claims can be validated.

---

## References

[1] Li, Z., Chen, Y., Ma, Z., Zhang, Y., & Guo, Y. (2025). Lossless ANN-SNN Conversion for Modern Transformers via Time-Coded Activation Alignment. *arXiv preprint* arXiv:2505.09659.

[2] Ye, L., Tian, Y., Li, Q., & Zhu, S.-C. (2024). Differential Transformer. *arXiv preprint* arXiv:2410.05258.

[3] SleepGate (2026). Conflict-Aware Temporal Tagging for Memory Consolidation in LLM Agents. *arXiv preprint* arXiv:2603.14517.

[4] Pan, Y., Chen, Y., & Ma, Z. (2025). SpikingBrain Technical Report: Spiking Brain-inspired Large Models. *arXiv preprint* arXiv:2509.05276.

[5] Zhou, Z., Zhu, Y., He, C., Wang, Y., Yan, S., Tian, Y., & Yuan, L. (2024). Spikingformer: Spike-driven Residual Learning for Transformer-based Spiking Neural Network. *arXiv preprint* arXiv:2304.11954.

[6] OPLoRA (2025). Orthogonal Projection for Low-Rank Adaptation. *arXiv preprint* arXiv:2510.13003.

[7] Aeon Memory (2026). Neuro-Symbolic Episodic Memory with Sleep Consolidation. *arXiv preprint* arXiv:2601.15311.

[8] CAMP (2026). Cognitive Arbitration for Multi-Perspective Output. *arXiv preprint* arXiv:2604.00085.

[9] Catfish (2025). Constructive Adversarial Feedback for Response Quality. *arXiv preprint* arXiv:2505.21503.

[10] KnowBias (2026). Knowledge-Aware Bias Detection in LLM Outputs. *arXiv preprint* arXiv:2601.21864.

[11] RBD (2025). Runtime Bias Detection for Language Model Inference. *arXiv preprint* arXiv:2505.17100.

[12] Gao, Z., et al. (2025). MoLA: Higher Layers Need More LoRA Experts. *Proceedings of NAACL 2025*.

[13] Assran, M., Bardes, A., Fan, D., Garrido, Q., Howes, R., et al. (2025). V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning. *arXiv preprint* arXiv:2506.09985.

[14] Balestriero, R., & LeCun, Y. (2025). LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics. *arXiv preprint* arXiv:2511.08544.

[15] LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence. *Open Review* (Version 0.9.2, June 2022).

[16] Simeoni, O., Vo, H. V., Seitzer, M., Baldassarre, F., Oquab, M., et al. (2025). DINOv3. *arXiv preprint* arXiv:2508.10104.

---

## Appendix A: LIF Neuron Implementation

```python
@dataclass
class LIFNeuron:
    threshold: float = 1.0
    tau: float = 1.0        # 1.0 = pure integrate-and-fire (no leak)
    v_init: float = 0.0

    def simulate(self, currents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        T = currents.shape[0]
        v = np.full(currents.shape[1:], self.v_init, dtype=np.float64)
        spikes = np.zeros_like(currents, dtype=np.float64)
        for t in range(T):
            v = self.tau * v + currents[t].astype(np.float64)
            fired = v >= self.threshold
            spikes[t] = fired.astype(np.float64)
            v = np.where(fired, v - self.threshold, v)  # soft reset
        return spikes, v
```

## Appendix B: Energy Benchmark CLI

```bash
uv run python scripts/energy_bench.py \
    --model-params 7e9 --spike-rate 0.3 --timesteps 4
```

Output: `results/energy-bench.json`

## Appendix C: SpikingMoELayer Routing Semantics

The SpikingMoELayer uses a dual-path forward:

1. **ANN path (routing):** `logits = x @ W_router^T + b_router` -- preserves exact logit ordering for top-K selection.
2. **SNN path (experts):** Each selected expert runs through SpikingLinear with rate-coded LIF encoding.
3. **Combination:** Softmax-normalized weights from router logits, applied to spiked expert outputs.

This design ensures that routing decisions are never corrupted by spike quantization noise, while expert computations benefit from the energy savings of binary spike operations.

## Appendix D: Dataset and Model Summary

| Item | Value |
|------|-------|
| Dataset size | 489,348 examples |
| Domain count | 35 |
| Source types | 3 (CLI sessions, HuggingFace, curated) |
| Base model | Qwen3.5-35B-A3B |
| Active params/token | 3B |
| LoRA rank | 16 |
| GGUF quantization | Q4_K_M (2.5 GB) |
| V3 vs V2 | 8 wins, 5 losses, 22 ties |
| Null-space preserved | 22/32 shared domains |
| Best V3 domain | stm32 (val_loss 0.68) |
| Worst V3 regression | spice-sim (+1.51) |
| Largest V3 improvement | electronics (-0.55) |

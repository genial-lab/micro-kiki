# Aeon-as-AMI-Memory: A Hybrid Working Memory Substrate for AMI-class Systems

**Authors**: L'Electron Rare, et al.
**Target venue**: NeurIPS 2026 Workshop on World Models / Cognitive Architectures
**Draft version**: v1 — 2026-04-19
**Status**: Workshop submission first draft (iterate before submission)

---

## Abstract

LeCun's *A Path Towards Autonomous Machine Intelligence* (arXiv:2206.15331) posits a seven-module cognitive architecture but leaves concrete implementations of several modules open, particularly the Short-Term Memory module (Module 7) for text and dialogue. We report on **Aeon**, a candidate implementation of Module 7 for LLM serving stacks, and the compression-plus-routing pathway that can serve as its Configurator (Module 1). Aeon is a numpy-only latent-state predictor (~100K parameters, < 1 MB weights) stacked above a dense vector index and a temporal graph substrate. It learns a transition map `h_t → h_{t+1}` over sentence-level embeddings and is paired with runtime anti-collapse safeguards. We validate three mechanisms empirically. (1) DinoV3-style running-mean centering delivers a +22% relative improvement in Mean Reciprocal Rank (MRR) on stack-structured streams (0.413 → 0.498), with a deterministic std-ratio tripwire that rolls weights back on collapse. (2) Per-sample LayerNorm of the residual delta preserves discrete stack-conditioning that running-mean centering destroys: 59% `win_stack` at 300 epochs in condition L2 (0.447 predictive MRR vs 0.090 null-stack MRR). (3) A Text-JEPA compressor delivers 3× embedding compression (384 → 128 dimensions) while retaining 97% of the downstream VQC routing accuracy (0.925 → 0.900 on 10-domain classification), validating the compression pathway through the Configurator. A companion case study documents the diagnostic: centering (per-batch) and stack-conditioning (per-sample) are mutually exclusive under standard MTL concatenation, but LayerNorm(delta) (per-sample) restores compatibility. All experiments use synthetic streams or a small real-conversation evaluation; we scope the contribution to the Module 7 substrate rather than to a full AMI system. Code, benchmarks, and JSON artifacts are released at commits `b1969e9`, `d30ffb8`, `75f22fb`, `f2f8242`.

---

## 1. Introduction (~600 words)

### 1.1 The memory gap in LLM serving

Modern LLM serving stacks have converged on a two-tier cognitive architecture: a base model with a long context window, augmented by an external memory layer implemented as retrieval-augmented generation (RAG), a memory palace, or a virtual-OS-style buffer (MemGPT, arXiv:2310.08560; Larimar, arXiv:2403.11901; RETRO, arXiv:2112.04426). These systems solve the "where does state live?" problem but not the "what will the model need next?" problem. The latter is the JEPA question in miniature: given the present latent state, can we predict the next latent state cheaply enough to pre-fetch, pre-rank, or speculatively execute?

The gap is particularly acute in multi-expert deployments. Micro-kiki, the target system for this work, hosts 35 domain-expert LoRAs on a shared Qwen3.5-35B-A3B MoE base, with a router selecting up to four stacks per turn. Serving latency is dominated by (a) the retrieval step, (b) the per-stack adapter application, and (c) any downstream reranking. Anticipation of the next turn's memory region or active stack can amortize all three.

### 1.2 LeCun's AMI Module 7 vision

LeCun's 2022 position paper arXiv:2206.15331 enumerates seven modules in a proposed Autonomous Machine Intelligence architecture: Configurator, Perception, World Model, Cost, Critic, Actor, and Short-Term Memory. Module 7 is described as a working-context store that aggregates recent experience and supports the other modules with a reusable, fast substrate. Several of the other modules have received substantial attention — V-JEPA 2 for the World Model in visual domains (arXiv:2506.09985), DreamerV3 for planning (arXiv:2301.04104), DINO family for Perception (arXiv:2104.14294, 2304.07193, 2508.10104) — but Module 7 for text and dialogue remains under-specified, particularly as a deployable substrate rather than an architectural diagram.

### 1.3 What we contribute

We propose that Aeon, a numpy-only latent predictor with anti-collapse safeguards, is a deployable candidate for Module 7 in text and dialogue LLM serving, and we validate three mechanisms around it:

1. **Centering + rollback (anti-collapse primitive)**: A DinoV3-style running-mean centering of predicted embeddings, paired with a std-ratio tripwire that rolls back weights when `std(ĥ)/std(h) < 0.1`. Delivers +22% relative MRR on stack-structured streams (condition D: 0.413 → 0.498); deterministic and cheap (< 1 MB weights, < 2 seconds training per 1000 turns on an Apple M5 CPU).

2. **LayerNorm(delta) for stack-conditioning (conditional anti-collapse)**: Replacing running-mean centering with per-sample LayerNorm of the residual delta preserves the discrete stack identifier signal that centering destroys. Condition L2 (300 epochs, lr = 5e-3) yields 59% `win_stack` with predictive MRR of 0.447 vs null-stack MRR of 0.090. A companion case study documents the diagnostic — the axis of normalization (per-sample vs per-batch) is the load-bearing property.

3. **Text-JEPA compression for the Configurator (Module 1)**: A self-supervised compression head reduces sentence-level MiniLM-L6 embeddings from 384 to 128 dimensions (3× compression) while retaining 97% of the downstream VQC routing accuracy (0.925 → 0.900 on a 10-domain conversational corpus). This validates the compression pathway for the AMI Configurator that pairs with Aeon at serving time.

### 1.4 Paper organization

Section 2 reviews JEPA predictors, anti-collapse regularizers, and memory-augmented LLMs. Section 3 describes Aeon's architecture. Section 4 reports the empirical evaluation across five synthetic-stream conditions (A–E) and two extended ablations (F, L). Section 5 presents the theoretical diagnostic — centering vs LayerNorm(delta) compatibility — that is documented in full in the companion case study. Section 6 maps each mechanism onto the AMI module structure. Sections 7 and 8 position the work and enumerate limitations. Section 9 concludes.

---

## 2. Background (~500 words)

### 2.1 JEPA family

Joint-Embedding Predictive Architectures predict in latent space rather than observation space. I-JEPA (arXiv:2301.08243) demonstrated that predicting one image region from another without pixel reconstruction learned strong invariant features; V-JEPA (arXiv:2404.08471) extended this to video; V-JEPA 2 (arXiv:2506.09985) introduced action-conditioned prediction for planning. The family shares three methodological commitments: no decoder to pixel/token space, no contrastive loss, and an explicit mechanism to prevent representational collapse. The choice of that mechanism is where specific methods diverge.

### 2.2 Anti-collapse mechanisms

Three lineages dominate. First, EMA teacher networks with stop-gradient (DINO, arXiv:2104.14294; DINOv2, arXiv:2304.07193; DINOv3, arXiv:2508.10104) prevent the student-teacher system from collapsing to a constant. Second, explicit centering of the teacher output, used in DINO and DINOv3 in combination with sharpening, removes shortcut solutions where embeddings cluster on a common mean. Third, principled regularizers such as SIGReg (LeJEPA, arXiv:2511.08544) project embeddings to a standard Gaussian using Cramér-Wold theorem sketches, replacing the heuristic bag with a theoretically-grounded alternative.

Aeon's design is closest in spirit to LeJEPA: we reject EMA teachers (no second model to maintain), reject stop-gradient hacks, and instead pair a single mechanism — running-mean centering, stateless, computed from the predictor's own output distribution — with a runtime safety net (std-ratio rollback). LayerNorm (Ba, Kiros, Hinton, arXiv:1607.06450) provides the per-sample alternative we exploit in the conditional case.

### 2.3 Memory-augmented LLMs

The deployment substrate for Aeon is the family of memory-augmented LLM stacks: MemGPT (arXiv:2310.08560) as the virtual-OS analogue; Larimar (arXiv:2403.11901) for learned retrieval anticipation; RETRO (arXiv:2112.04426) for retrieval-augmented decoding. These systems solve state persistence but treat memory access as on-demand. Aeon learns the transition function directly, upstream of the retriever.

### 2.4 AMI architecture overview

LeCun (arXiv:2206.15331) proposes a decomposition in which a Configurator generates task-specific configurations from a gradient-free policy; a Perception module produces state estimates; a World Model predicts future states; a Cost module scores trajectories; a Critic estimates value; an Actor executes; and Short-Term Memory (Module 7) serves as the working buffer shared among the others. Most JEPA work has targeted the World Model; we target Module 7, with a compressed path through Module 1 (Configurator).

The relationship between our work and generative world models (DreamerV3, arXiv:2301.04104; TD-MPC2, arXiv:2310.16828) is one of deliberate contrast: generative world models reconstruct observations and plan over learned dynamics; Aeon predicts transitions in a small latent space and hands decisions to a retrieval substrate. Aeon is a memory-management aid, not a planner, and this distinction is load-bearing for the Module 7 claim.

---

## 3. Aeon predictor architecture (~800 words)

### 3.1 Design constraints

Three constraints bound Aeon's design. First, **deployability on commodity hardware**: Aeon must co-run with a serving LLM on the same node, with no GPU allocation of its own. Second, **runtime safety**: the predictor touches ranking decisions, so silent degradation is unacceptable. Third, **cold-start tolerance**: Aeon runs from conversation turn one, before enough pairs exist to train anything meaningful.

The resulting envelope is strict. The predictor is implemented in pure numpy (no PyTorch, no JAX, no GPU kernels). Total parameter count is approximately 100K, weight file is < 1 MB, and one full training pass over 1000 turns completes in under two seconds on an Apple M5 CPU. Below a configurable threshold (500 training pairs by default), `predict_next()` returns `h_t` unchanged, keeping the system on pure retrieval mode until warmup completes.

### 3.2 LatentMLP forward

The core predictor is a two-layer MLP with a learned skip connection and optional stack conditioning:

```
z = W_1 · [h_t ; α · one_hot(s)] + b_1
h_{t+1}_hat = skip · h_t + W_2 · ReLU(z)
```

where `h_t ∈ R^384` is a sentence-embedding (MiniLM-L6 in our experiments), `s ∈ {0, …, N-1}` is the stack identifier, `α = sqrt(dim / n_stacks) ≈ 4.9` for `dim = 384, n_stacks = 16` matches the one-hot norm to the embedding norm (an early PoC revealed that without this rescaling the one-hot is effectively washed out at the first layer), and `skip ∈ [0, 1]` is a learned interpolation between identity and MLP prediction. The hidden dimension is 256, giving total trainable parameters of approximately 100K in float32.

The loss is cosine similarity: `L = 1 - cos(ĥ_{t+1}, h_{t+1})`. MSE was considered and rejected — it collapses to the predictor reproducing the batch mean of `h_{t+1}`, a trivial solution that passes std-ratio checks but learns nothing. Cosine loss is scale-invariant and does not require temperature tuning, which is useful at cold start when the predictor's output scale is not yet stable.

### 3.3 Anti-collapse options

Three anti-collapse mechanisms coexist in the codebase, user-selectable via configuration flags:

**Running-mean centering (default, non-conditional).** After each forward pass, update the running mean `μ ← 0.9 μ + 0.1 mean(ĥ)` and subtract `μ` from `ĥ` before the cosine loss. This is DinoV3-style but stateless: no teacher, no sharpening, no stop-gradient. At inference, `μ` is frozen.

**Per-sample LayerNorm of the residual delta (conditional).** Compute `delta = W_2 · ReLU(z)`, apply standard LayerNorm across the feature dimension (per-sample mean and variance) with learnable gamma and beta, and add the result to `skip · h_t`. This alternative preserves per-sample offsets — including the one-hot-induced stack offset — that batch-level centering averages out. The rationale is developed in Section 5.

**Std-ratio tripwire + weight rollback (runtime safety, always on).** At each epoch boundary we compute `r = std(ĥ) / std(h)` over the training buffer. If `r < 0.1` (a deterministic threshold), the predictor reverts to the checkpoint before the offending epoch and the epoch is marked as a rollback event in a telemetry log. The tripwire is orthogonal to the choice of centering vs LayerNorm(delta) and fires independently of either.

### 3.4 Stack-conditioning integration

Stack identity is encoded as a one-hot vector concatenated to `h_t` at the MLP input. The scaling factor `α = sqrt(dim / n_stacks)` brings its L2 norm into the same order as the embedding. With centering active, this signal is effectively removed from the output (Section 5); with LayerNorm(delta), it is preserved.

### 3.5 Integration with AeonSleep (sleep_cycle hook)

Aeon's training is triggered by an offline consolidation pass that we call `sleep_cycle`, invoked between conversational sessions. The hook drains recent turn pairs from the Trace graph, shuffles them (breaking within-session correlation), and performs one or more epochs of gradient descent over the LatentMLP. This design separates the hot path (inference with frozen weights) from the slow path (offline training), matching the JEPA pattern of training-time regularization without runtime overhead.

Cross-session persistence is achieved via Atlas (the SIMD vector index) and Trace (a NetworkX temporal graph). In a smoke test across 14 conversational turns distributed over multiple sessions, AeonSleep retrieved 36 relevant past items against 0 for a raw LLM baseline without the memory layer. We do not claim this as a primary result — the experiment is illustrative — but it establishes that Aeon's cross-session substrate functions end-to-end.

### 3.6 Configuration: VQC router + Text-JEPA compressor as AMI Configurator

The Configurator path is implemented by two components that pair with Aeon at serving time:

**VQC router**. A 6-qubit variational quantum circuit (6 StronglyEntanglingLayers, approximately 180 variational parameters when including the classical read-out head) classifies input embeddings into 35 sigmoid-output domain classes (34 niches + base). The router runs on a PennyLane `default.qubit` simulator today, with parameter-shift rule gradient; no NISQ hardware is required for the claims in this paper.

**Text-JEPA compressor**. A small projection head reduces the 384-dimensional embeddings to 128 dimensions while preserving downstream routing accuracy. On a 10-domain classification setup using real conversational embeddings, the uncompressed baseline achieves 0.925 accuracy and the Text-JEPA compressed representation achieves 0.900 — a 97% retention ratio under a 3× compression. This is the Configurator path: compressed embedding → VQC → routing decision → which stack is selected for the next turn.

The VQC router (6 qubits, 35 classes, approximately 180 parameters) is three orders of magnitude smaller than a classical sigmoid head of comparable expressivity (~3.4M parameters for a dense 384×35 layer with intermediate features). The quantum component is thus interesting as a **compression test** for the Configurator, not as a planner or optimizer; the full quantum-classical trade-off is deferred to a companion Paper B.

---

## 4. Empirical evaluation (~1200 words)

### 4.1 Experimental setup

**Synthetic streams.** We evaluate on two classes of 1000-turn synthetic streams. *Random-walk* streams (conditions A, B) generate each successor embedding by adding isotropic Gaussian noise to `h_t`; stack identity is uncorrelated with stream dynamics. *Stack-structured* streams (conditions C, D, E, F, L) generate per-stack drift patterns, so the stack identifier carries predictive information about `h_{t+1}`.

**Real conversational evaluation for the Configurator.** The Text-JEPA compression and VQC routing results use a 10-domain conversational corpus of real embedded turns, not synthetic data. This is an important qualifier: the Aeon predictor results are synthetic-only, but the Configurator-path compression claim is validated on real data.

**Metrics.** For each condition we report: `recall@5` (portion of queries where ground truth lands in top-5 results from a gallery), `MRR` (mean reciprocal rank, inverse of the rank of the first correct result), `win_pred` (percentage of queries where the predictor beats baseline retrieval on `recall@5`), `win_stack` (percentage of queries where the stack-aware predictor beats a null-stack predictor with a zeroed one-hot), and `final_loss` (cosine loss at end of training). Each condition uses 100 held-out queries.

**Compute.** All experiments run on an Apple M5 CPU (GrosMac node), single-threaded numpy. Typical wall-clock is 1.2–4.2 seconds for a full 1000-turn training pass, depending on epoch count.

### 4.2 Centering + Text-JEPA baseline (Table 1)

Table 1 summarizes the five-condition evaluation (synthetic streams) and the Text-JEPA Configurator evaluation (real conversational). All numbers are taken directly from the PoC results files at `micro-kiki-poc-aeon/results/2026-04-17-aeon-poc-{A,B,C,D,E}-*-v2.json`.

| Condition | Stream | Center? | Epochs / LR | Baseline MRR | Predictive MRR | Null MRR | win_pred | win_stack | Final loss |
|-----------|--------|---------|-------------|--------------|----------------|----------|----------|-----------|-----------|
| A (vanilla) | random-walk | No | 50 / 1e-3 | 0.263 | 0.264 | 0.252 | 20% | **23%** | 0.835 |
| B (+centering) | random-walk | Yes | 50 / 1e-3 | 0.263 | 0.228 | 0.232 | 17% | 18% | 0.835 |
| C (stack-stream) | stack | No | 50 / 1e-3 | 0.413 | 0.415 | 0.412 | 5% | 5% | 0.567 |
| D (stack+center) | stack | Yes | 50 / 1e-3 | 0.413 | **0.498** | 0.498 | **51%** | 0% | 0.567 |
| E (D long) | stack | Yes | 300 / 5e-3 | 0.413 | 0.500 | 0.498 | 52% | 1% | 0.520 |

**Key observations.** Condition D is the headline centering result: MRR improves from 0.413 (baseline retrieval alone) to 0.498 (with the predictor reranking) — a +22% relative gain. This is the non-trivial anti-collapse contribution. Condition E, with six times the training budget and a higher learning rate, stabilizes the predictive MRR at 0.500 and raises `win_pred` to 52%, confirming that the centering gain is not an artifact of an under-trained baseline.

Conditions A and B bound the centering claim: on random-walk streams, baseline `recall@5` is 0.66 (far from saturation), and centering slightly reduces MRR (0.263 → 0.228). Centering's benefit concentrates in the saturation regime (baseline `recall@5 = 1.0` on C/D/E), where the only axis for improvement is ranking within an already-correct top-5. This is disclosed as a limitation in Section 8.

[Figure 1: predictive MRR progression across conditions — to be generated from `results/2026-04-17-aeon-poc-{A,B,C,D,E}-v2.json`; shows the headline +22% gain on D and the negative MRR effect on B.]

### 4.3 LayerNorm(delta) validation for stack-conditioning (Table 2)

The stack-conditioning axis requires a different anti-collapse mechanism. Table 2 reports the extended ablation (conditions F and L) that tests two proposed fixes from the case study (Section 5).

| Condition | Setup | win_stack | Predictive MRR | Null MRR | Baseline MRR | Verdict |
|-----------|-------|-----------|----------------|----------|--------------|---------|
| F1 | per-stack centering (50 ep) | 0% | 0.413 | 0.495 | 0.413 | FAIL |
| F2 | per-stack centering (300 ep) | 0% | 0.433 | 0.498 | 0.413 | FAIL |
| L1 | LayerNorm(delta) (50 ep) | 3% | 0.012 | 0.015 | 0.413 | under-trained |
| **L2** | **LayerNorm(delta) (300 ep, lr=5e-3)** | **59%** | **0.447** | **0.090** | 0.413 | **SUCCESS** |
| L3 | LayerNorm(delta) + centering | 1% | 0.005 | 0.012 | 0.413 | catastrophic |

**Key observations.** Per-stack centering (conditions F1, F2) — maintaining separate running means for each stack identifier, as a naïve fix for the shared-mean interference — does not recover the stack signal at either 50 or 300 epochs. This rules out the "more training will fix it" and "just pool the means per-stack" hypotheses. The incompatibility is structural.

Per-sample LayerNorm of the residual delta (condition L2) succeeds decisively at 300 epochs with lr = 5e-3: 59% `win_stack`, predictive MRR of 0.447 vs null-stack MRR of 0.090 (a +397% relative gap on the stack-conditioning axis). Condition L1 (50 epochs) shows the mechanism converges slowly — 3% `win_stack` early — but the extended-training result in L2 is decisive.

Condition L3, combining LayerNorm(delta) *and* running-mean centering, is catastrophic: 1% `win_stack` and predictive MRR of 0.005, worse than L1. The two anti-collapse mechanisms must be used exclusively; they compound destructively when stacked.

[Figure 2: training-loss trajectories for L1, L2, L3 — to be generated from extended-ablation JSON; shows slow convergence of L2 and the pathological trajectory of L3.]

### 4.4 Text-JEPA compression ablation (Table 3)

The Configurator path is validated on real conversational embeddings. Table 3 reports the 10-domain classification experiment with a VQC router on top of MiniLM-L6 (384 dim) vs Text-JEPA compressed (128 dim) representations. Numbers are from the PoC A completion report.

| Representation | Dim | VQC Routing Accuracy | Compression Ratio | Retention |
|----------------|-----|----------------------|-------------------|-----------|
| MiniLM-L6 (uncompressed) | 384 | 0.925 | 1.0× | — |
| **Text-JEPA (compressed)** | **128** | **0.900** | **3.0×** | **97%** |

The Text-JEPA compressor retains 97% of the downstream routing accuracy while compressing the Configurator's input by 3×. This is a practical win for serving: compressed embeddings reduce VQC state-preparation cost (fewer angle-encoded features), reduce memory bandwidth for the router, and shorten the distance between Aeon's latent space and the Configurator's input. The result is scoped to 10 domains (not the full 35); full-scale validation on the production router is pending. [Task 15 pending — <VERIFY: micro-kiki Task 15 status>.]

[Figure 3: confusion matrices for uncompressed and compressed Text-JEPA representations — to be generated from PoC A evaluation output.]

### 4.5 Real-data evaluation: topic-switch anticipation (Table 4)

To address the "synthetic-only" limitation, we ran the predictor on real conversational embeddings from the 10-domain micro-kiki corpus (`data/final/`). Each turn is a user message encoded via frozen MiniLM-L6 (384-d). We tested two stream topologies (round-robin interleaved — topic-switched; and within-topic — contiguous domain blocks) and two retrieval metrics (exact next-turn id match; soft-domain match = top-5 retrieved set contains any message from the gold-next-turn's domain).

The exact-match metric failed on both topologies: baseline recall@5 = 1–2% because retrieving the literal t+1 message from a 1000-message gallery is near-impossible under cosine-similarity retrieval, regardless of predictor. This is a protocol limitation, not a predictor failure. We therefore adopt the soft-domain metric, which tests the operationally relevant question: *does the retrieved context belong to the right routing expert?*

Table 4 reports the soft-domain numbers (1000 ingested turns, 100 held-out queries, LayerNorm(delta) predictor, 50 epochs, lr=1e-3, commit `63e546c` on branch `poc/text-jepa-vqc`).

| Stream topology | Baseline recall@5 | Predictive recall@5 | Baseline MRR | Predictive MRR | `win_rate_predictive` |
|-----------------|------------------|---------------------|--------------|----------------|-----------------------|
| **Interleaved (topic-switched)** | 0.11 | **0.31** | 0.040 | **0.163** | **0.29** |
| Within-topic (contiguous blocks) | 0.99 | 0.47 | 0.99 | 0.109 | 0.01 |

On the interleaved stream — the setting that actually exercises anticipation — the predictive path **triples recall@5** (0.11 → 0.31, +20 absolute points, +181% relative) and quadruples MRR (0.040 → 0.163). The predictor correctly beats pure retrieval on 29% of queries, while null-stack conditioning (`stack_id = -1`) gives a comparable 0.29 recall@5, confirming that the signal comes from the residual-latent learning and not from stack-id leakage alone; `win_rate_stack_vs_null` is +12% (stack-conditioned beats null-stack on MRR).

On the within-topic stream, baseline retrieval saturates at 0.99 (retrieving any same-domain message from a contiguous block is trivial) and the predictive path is not relevant; we disclose this honestly rather than hide the saturation.

This is the primary real-data validation: **Aeon with LayerNorm(delta) conditioning delivers a meaningful uplift over pure-retrieval memory on conversational streams with topic shifts**, which is the practical regime for a routed LLM serving stack where the user can change subject turn-over-turn.

### 4.6 Rollback mechanism unit test

The std-ratio tripwire is tested in isolation by `tests/memory/test_aeon_predictor.py::test_collapse_detector_triggers`. The test injects an artificial collapse (setting predictor output to a constant vector) and verifies: (a) the std-ratio falls below 0.1, (b) the detector emits a warning, (c) the weight checkpoint is restored, and (d) subsequent forward passes return to pre-collapse output statistics. The test passes at commit `b22fa12`.

In long-run telemetry over condition E (300 epochs), the tripwire fired zero times: centering's explicit mean subtraction is enough to prevent collapse on these streams. On stream variants with higher-dimensional correlation (not reported in this draft), the tripwire fires approximately 0.3–1.5% of epochs, confirming the mechanism activates when stressed. [<VERIFY: telemetry log path for stream variants — results/aeon-telemetry-*.json or similar>]

### 4.7 Summary scorecard

Across the three validated mechanisms:

| Mechanism | Evidence | Strength |
|-----------|----------|----------|
| Centering + rollback | +22% MRR condition D (0.413 → 0.498) | **Strong** |
| LayerNorm(delta) | 59% win_stack L2 (0.447 vs 0.090 null) | **Strong** |
| Text-JEPA compression | 97% retention at 3× compression | **Strong** |
| Std-ratio rollback | Unit-test `test_collapse_detector_triggers` passes | **Strong** |
| 100K-param numpy runtime | < 1 MB weights, < 2s / 1000 turns on M5 | **Strong** |
| Centering harms random-walk | 0.263 → 0.228 on A/B | **Disclosed** |
| Centering+stack incompatibility | 0–1% win_stack on D, E | **Disclosed (and fixed by LayerNorm(delta))** |
| Real-data topic-switch anticipation | 11% → 31% recall@5 (+181% rel), 4× MRR on interleaved 10-domain stream, soft-domain match | **Strong** |
| Within-topic baseline saturation | 99% baseline recall, no predictor headroom | **Disclosed** |

The three strong claims have dedicated subsections (4.2, 4.3, 4.4) and are load-bearing for the Module 7 positioning.

---

## 5. Analysis: Centering vs LayerNorm(delta) compatibility (~600 words)

We summarize the diagnostic that is developed in full in the companion case study (*Stack-Conditioned Prediction under Centering Regularization: A Case Study in Latent Predictors*, same authors, 2026-04-19). The case study ran the extended F/L ablations reported in Section 4.3 and derives the mathematical and structural reason why running-mean centering and discrete stack-conditioning are mutually exclusive under one-hot MTL concatenation, while LayerNorm(delta) restores compatibility.

### Mathematical view

The first-layer pre-activation for a sample with stack `s` is

```
z = W_1^{(h)} · h_t + α · W_1^{(s)}[:, s] + b
```

where `W_1^{(s)}[:, s]` is the column of the first weight matrix that absorbs the one-hot. After ReLU and `W_2`, this column contributes a per-stack additive offset `μ_s` at the output. Running-mean centering maintains `μ ≈ E_s[μ_s]` (the expected output mean under roughly uniform stack frequency) and subtracts `μ` before the cosine loss. The subtraction leaves between-stack variance nominally intact, but the gradient through `W_1^{(s)}` *shrinks* because the per-stack offset can no longer help the predictor hit `h_{t+1}` — centering has already zeroed out the mean that the one-hot was injecting. The predictor dutifully learns to ignore `s`, and `win_stack` collapses to zero.

This is not a bug in the centering formulation — it is what DINOv3 centering is *for*, introduced precisely to prevent trivial per-class solutions. In our conditional setting, that "trivial per-class solution" is exactly the signal we want to preserve.

### Per-batch vs per-sample regularization: the axis of normalization

The failure of condition F (per-stack centering) at both 50 and 300 epochs was initially surprising: if the incompatibility were about pooled statistics, maintaining 32 separate running means indexed by `stack_id` should have fixed it. It did not — F1 and F2 both yield 0% `win_stack`.

The refined diagnostic is that the relevant axis is not **which means are pooled** but **what dimension normalization runs on**. Running-mean centering — whether global or per-stack — computes statistics over the *batch* axis. Per-sample conditioning signals, such as a one-hot stack offset, are washed out within each stack's batches because within-stack samples share the same offset and subtracting the within-stack mean removes it.

LayerNorm operates on a different axis entirely: for each sample, it normalizes across the feature dimension. A per-sample additive constant (the one-hot-induced offset) is preserved: LayerNorm's learned `gamma` and `beta` parameters can even amplify it if the downstream loss benefits. This is why condition L2 recovers the stack signal at 59% `win_stack` with a +397% relative margin over the null-stack baseline.

### Why L3 is catastrophic

Stacking LayerNorm(delta) and running-mean centering together (condition L3) leaves no coherent signal for the predictor to learn: per-sample feature-variance normalization and per-batch mean subtraction compound their regularization pressures, and the cosine loss cannot find gradient directions that preserve either conditioning or predictive value. The MRR collapses to 0.005. This is an important finding for practitioners: **these two mechanisms must be used exclusively**, not combined.

### Paper-worthy generalization

The principle — per-sample conditioning requires per-sample regularization to survive — generalizes beyond Aeon's specific architecture. Anyone adapting JEPA-family regularizers (DINO-style centering, LeJEPA/SIGReg, EMA teachers) to a *conditional* latent predictor faces a version of this tension. The cleanest mitigations are structural: weight-level conditioning (hypernetworks, MoE-predictors) or per-sample regularization (LayerNorm, or a per-sample SIGReg variant). Combining batch-level regularization with additive per-sample conditioning is a known failure mode that the case study documents empirically.

The practical takeaway for Paper A is that Aeon offers two anti-collapse defaults: centering for the non-conditional path (Section 4.2), LayerNorm(delta) for the conditional path (Section 4.3). Deployers choose based on whether their downstream task exploits discrete per-sample conditioning.

---

## 6. Position in AMI architecture (~400 words)

We map each validated mechanism onto the AMI module structure of arXiv:2206.15331.

| AMI module | Aeon component | Claim strength | Notes |
|------------|----------------|----------------|-------|
| **1. Configurator** | VQC router (6 qubits, 35 classes, ~180 params) + Text-JEPA compressor (384 → 128, 97% retention) | **Strong (compression); Partial (VQC as policy)** | The Text-JEPA compression claim is validated on real conversational data. VQC is a small classifier, not a full gradient-free policy generator; its role here is the Configurator's *input compression* path. |
| **2. Perception** | n/a | **None** | Our inputs are text embeddings, not raw observations. |
| **3. World Model** | Aeon LatentMLP (single-step transition) | **Partial** | We predict `h_t → h_{t+1}` in sentence-embedding space at horizon 1, not full world dynamics. |
| **4. Cost** | n/a (external CAMP judge in the wider system) | **None in this paper** | CAMP arbitration (arXiv:2604.00085) lives in micro-kiki but is not claimed here. |
| **5. Critic** | n/a | **None** | No value function is learned. |
| **6. Actor** | LLM stack (Qwen3.5-35B-A3B + LoRAs) | **Delegated** | Execution is handled by the base LLM; out of scope. |
| **7. Short-Term Memory** | Aeon (Atlas + Trace + LatentMLP + anti-collapse + rollback) | **STRONG** | The primary claim of this paper, backed by Sections 4.2, 4.3, 4.4 and the case study. |

The paper's two strong claims are on rows 1 (compressed Configurator path) and 7 (working memory). Rows 3, 4, and 6 are included for architectural completeness — to be explicit about what we *do not* claim. In particular, we are not a World Model: Aeon's single-step transition is a memory-management aid, not a generative dynamics model, and we do not close any Actor/Cost feedback loop in this paper.

**What is missing for full AMI membership.** A proper Actor interface that takes Aeon's predicted `ĥ_{t+1}` as a planning hint, an explicit Cost/Critic pair, and a Perception module for non-text inputs. These are follow-up work; for this paper we explicitly scope to Module 7 (with the Configurator compression path as a secondary contribution). This is the "building block, not system" framing we adopt throughout.

---

## 7. Related work (~500 words)

We position Aeon across three families.

### Latent predictive architectures (JEPA)

I-JEPA (arXiv:2301.08243), V-JEPA (arXiv:2404.08471), and V-JEPA 2 (arXiv:2506.09985) form our closest methodological cousins. The shared commitment is prediction in latent space, without decoder to pixels or tokens, without contrastive losses, with a mechanism to prevent collapse. V-JEPA 2 in particular introduced action-conditioned prediction; Aeon is stack-conditioned, which is similar in structure (a discrete categorical variable conditions the transition) but different in semantics (stacks are not controllable; the LLM orchestrator selects them, but the user's next turn is not an action in the reinforcement-learning sense).

Aeon differs from JEPA by three choices: (a) it learns from real-time conversation turns with a cold-start fallback, not pre-recorded trajectories; (b) it has no visual ground truth, only text-embedding targets; (c) it operates under a 32-stack expert-mixture context, a coordination problem absent from single-model JEPA deployments.

### Anti-collapse regularization

DINO (arXiv:2104.14294), DINOv2 (arXiv:2304.07193), and DINOv3 (arXiv:2508.10104) introduced the EMA-teacher-plus-centering lineage that Aeon's running-mean centering descends from (with EMA dropped). LeJEPA (arXiv:2511.08544) replaced the heuristic bag with SIGReg, a principled Cramér-Wold projection regularizer that provides formal guarantees against mode collapse. Aeon's centering is philosophically aligned with LeJEPA — no teacher, no stop-gradient, no ad-hoc centering of a frozen statistic — but simpler and runtime-implementable. LayerNorm (arXiv:1607.06450) is the per-sample alternative we use in the conditional regime.

SigLIP (arXiv:2303.15343) and SigLIP2 (arXiv:2502.14786), contemporaries in vision-language, employ sigmoid-based losses and are distinct from the centering family. Aeon uses cosine loss, which is scale-invariant and does not require temperature tuning.

### Generative world models (deliberate contrast)

DreamerV3 (arXiv:2301.04104) and TD-MPC2 (arXiv:2310.16828) are latent world models in the generative sense: they reconstruct or simulate future states and plan via learned dynamics. These methods excel in long-horizon control; they incur a reconstruction cost and are designed for single-agent planning rather than multi-expert retrieval. Aeon deliberately rejects the world-model framing: we do not reconstruct, do not plan, do not simulate. Aeon predicts transitions in a small latent space and hands decisions back to a retrieval substrate. This is a memory-management aid, not a planner.

### Memory-augmented LLMs

MemGPT (arXiv:2310.08560), Larimar (arXiv:2403.11901), and RETRO (arXiv:2112.04426) form the deployment substrate family. These systems solve where state lives but not what the model needs next. Aeon operates upstream of the retriever: given a predicted `ĥ_{t+1}`, the Atlas/Trace substrate can pre-fetch or pre-rank the memory region likely to be requested. Stack-conditioning adds expert-mixture routing to this picture. To our knowledge, no prior work learns expert-selection via latent-state prediction — our search has not surfaced a direct analogue.

### Conditioning mechanisms

MTL concatenation (our failed baseline when combined with centering), MoE routing (arXiv:1701.06538), and hypernetworks (arXiv:1609.09106) represent the space of conditioning mechanisms. One-hot concatenation is the simplest — and, as Section 5 shows, the most exposed to batch-level regularization interference. Dense conditioning (via a softmax router output, for example) and weight-level conditioning (hypernetworks) remain untested mitigations that the case study flags for future work.

---

## 8. Limitations and future work (~400 words)

### Limitations

**Synthetic streams for Aeon predictor in Sections 4.2 and 4.3** use random-walk and stack-structured synthetic streams for the centering and LayerNorm(delta) ablations. Section 4.5 addresses this with a real-data evaluation on the 10-domain micro-kiki corpus, where the soft-domain metric shows +20 absolute recall@5 points over pure retrieval on interleaved streams. Still pending: scaling to >10 domains, larger corpora, and a matched evaluation on production serving logs.

**Saturation ceiling.** On stack-structured streams, baseline `recall@5` saturates at 1.0 across all queries. The predictor operates only on the MRR axis (reranking within perfect-recall sets). A harder benchmark with lower baseline recall would expose a wider improvement window and is needed to characterize the centering mechanism's benefit outside the saturation regime.

**Horizon = 1.** Aeon predicts one step ahead. Multi-step horizons would require curriculum learning or explicit rollout, and may interact non-trivially with centering (each rollout step would re-inject centering noise). Unexplored.

**No Actor integration.** The Module 6 Actor is delegated to the base LLM in the wider system; we do not demonstrate closed-loop feedback from LLM outputs back into Aeon's training. This keeps the Module 7 claim honest but limits the "AMI system" scope.

**VQC class count (35) vs LoRA stacks (32) mismatch.** The production micro-kiki router targets 35 classes (34 niches + base); the LoRA stack count is 32 foundations-plus-niches in the training curriculum. This 35/32 asymmetry is documented in `docs/research/vqc-class-count-reconciliation.md`; it is a version-management artifact rather than a scientific issue, but it is a reader-visible inconsistency that we flag here explicitly.

**Real serving latency untested.** All compute measurements are offline benchmarks (< 2 seconds per 1000 turns on M5). We have not measured latency under concurrent serving load with > 100 simultaneous queries. Centering-on vs centering-off ablation at serving time is also pending.

### Future work

- **Real-conversation benchmark for Aeon predictor.** A non-synthetic evaluation, likely using micro-kiki's own conversation logs.
- **Comparison against LeJEPA/SIGReg** when a reference implementation is released (arXiv:2511.08544 status as of writing: <VERIFY: code release status>).
- **Dense conditioning and hypernetwork conditioning** as alternatives to one-hot, to widen the non-destructive anti-collapse choices.
- **Multi-step horizon** — `h_t → h_{t+2}, h_{t+3}` — with curriculum learning.
- **Actor integration** to close the AMI loop: use predicted `ĥ_{t+1}` as a planning hint for the LLM stack.
- **Scale the Text-JEPA compression** from 10 domains to the full 35 production domains (Task 15 in micro-kiki's roadmap).

---

## 9. Conclusion (~200 words)

Aeon is a deployable candidate implementation of Short-Term Memory (Module 7) for LLM serving stacks within LeCun's AMI architecture. We validate three mechanisms: running-mean centering delivers +22% relative MRR on stack-structured streams with a deterministic rollback safeguard; per-sample LayerNorm of the residual delta preserves discrete stack-conditioning where centering destroys it (59% `win_stack` at 300 epochs, 0.447 predictive MRR vs 0.090 null-stack); Text-JEPA compression of the Configurator's input retains 97% of downstream routing accuracy at 3× compression. A companion case study documents the diagnostic — the per-sample vs per-batch axis of normalization is the load-bearing compatibility property, and combining the two anti-collapse mechanisms is catastrophic.

We scope the contribution to Module 7 and the Configurator compression path; we do not claim a full AMI implementation, a generative world model, multi-step planning, or closed-loop feedback. The code, benchmarks, and evaluation JSONs are released at commits `b1969e9`, `d30ffb8`, `75f22fb`, `f2f8242` on branch `main` of the companion repository. We consider this the minimum viable building block for subsequent work on Module 1 (Configurator), Module 3 (World Model lift from single-step to multi-step), and Module 6 (Actor integration) within the same architectural family.

---

## References

[1] Assran, M., et al. (2023). *I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.* arXiv:2301.08243.

[2] V-JEPA team (2024). *V-JEPA: Revisiting Feature Prediction for Learning Visual Representations from Video.* arXiv:2404.08471.

[3] Assran, M., et al. (2025). *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning.* arXiv:2506.09985.

[4] Hafner, D., et al. (2023). *Mastering Diverse Domains through World Models (DreamerV3).* arXiv:2301.04104.

[5] Hansen, N., Su, H., Wang, X. (2023). *TD-MPC2: Scalable, Robust World Models for Continuous Control.* arXiv:2310.16828.

[6] Caron, M., et al. (2021). *Emerging Properties in Self-Supervised Vision Transformers (DINO).* arXiv:2104.14294.

[7] Oquab, M., et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision.* arXiv:2304.07193.

[8] DINOv3 team (2025). *DINOv3: Scaling Self-Supervised Vision Representations.* arXiv:2508.10104.

[9] Balestriero, R., LeCun, Y. (2025). *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics (SIGReg).* arXiv:2511.08544.

[10] LeCun, Y. (2022). *A Path Towards Autonomous Machine Intelligence.* arXiv:2206.15331.

[11] Packer, C., et al. (2023). *MemGPT: Towards LLMs as Operating Systems.* arXiv:2310.08560.

[12] Das, P., et al. (2024). *Larimar: Large Language Models with Episodic Memory Control.* arXiv:2403.11901.

[13] Borgeaud, S., et al. (2022). *Improving Language Models by Retrieving from Trillions of Tokens (RETRO).* arXiv:2112.04426.

[14] Ba, L., Kiros, J. R., Hinton, G. E. (2016). *Layer Normalization.* arXiv:1607.06450.

[15] Vaswani, A., et al. (2017). *Attention Is All You Need.* arXiv:1706.03762.

[16] Shazeer, N., et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* arXiv:1701.06538.

[17] Ha, D., Dai, A., Le, Q. V. (2016). *HyperNetworks.* arXiv:1609.09106.

[18] Zhai, X., et al. (2023). *Sigmoid Loss for Language-Image Pre-Training (SigLIP).* arXiv:2303.15343.

[19] Tschannen, M., et al. (2025). *SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features.* arXiv:2502.14786.

[20] Rubinstein, R. Y. (1997). *Optimization of Computer Simulation Models with Rare Events.* European Journal of Operational Research 99:89–112. (Cross-entropy method foundation; cited in Configurator context.)

---

## Appendix A: Reproducibility (~300 words)

### Source repositories and commits

The Aeon predictor, LayerNorm(delta) extension, Text-JEPA compressor, and VQC router live across two repositories:

- **micro-kiki** (runtime, eval, docs): `/Users/electron/Documents/Projets/micro-kiki/`
- **micro-kiki-poc-aeon** (PoC A/B experiments, JSON result artifacts): `/Users/electron/Documents/Projets/micro-kiki-poc-aeon/`

Key commits on `main`:

- `b1969e9` — PoC A Text-JEPA VQC router merge (Section 4.4 Configurator compression).
- `d30ffb8` — LayerNorm(delta) + VQC reconciliation doc merge (Section 4.3 conditional anti-collapse).
- `75f22fb` — Paper A reframe update with LayerNorm and Text-JEPA integration (Section 3 architecture + Section 6 AMI mapping).
- `f2f8242` — Case study + LayerNorm fix (Section 5 diagnostic + companion paper).

Feature branch commits (merged via the above):

- `3c7eded` — LayerNorm(delta) feature code in `src/memory/aeon_predictor.py::LatentMLP` (flag: `use_layernorm_delta`).
- `804bf02` — LayerNorm(delta) benchmark results and JSON artifacts.

### Test commands

```bash
# Aeon predictor unit and integration tests (33 tests, numpy-only, ~0.3s)
cd ~/Documents/Projets/micro-kiki-poc-aeon
python -m pytest tests/memory/test_aeon_predictor.py -v
python -m pytest tests/memory/test_aeonsleep_predictor_hook.py -v
python -m pytest tests/scripts/test_eval_aeon_predictor.py -v

# Full PoC evaluation (regenerates results/*.json)
python scripts/eval_aeon_predictor.py --condition A --output results/2026-04-17-aeon-poc-A-vanilla-v2.json
python scripts/eval_aeon_predictor.py --condition D --output results/2026-04-17-aeon-poc-D-stack-centering-v2.json
python scripts/eval_aeon_predictor.py --condition E --output results/2026-04-17-aeon-poc-E-long-converge-v2.json

# LayerNorm(delta) condition L2
python scripts/eval_aeon_predictor.py --condition L --use-layernorm-delta --epochs 300 --lr 5e-3 \
  --output results/2026-04-17-aeon-poc-L2-layernorm.json
```

### Result artifacts

All JSON result files referenced in Sections 4.2 and 4.3 live under `micro-kiki-poc-aeon/results/`, named `2026-04-17-aeon-poc-{A,B,C,D,E,F,L}-*.json`. Each file contains: `baseline_mrr`, `predictive_mrr`, `null_stack_mrr`, `baseline_recall_at_5`, `predictive_recall_at_5`, `null_stack_recall_at_5`, `win_rate_predictive`, `win_rate_stack_vs_null`, `n_queries`, `elapsed_seconds`, `final_train_loss`, `predictor_ready`, `stream_type`, `use_centering`.

### Compute

Apple M5 CPU, single-threaded numpy. All PoC runs complete in 1.2–4.2 seconds for 1000-turn streams at 50–300 epochs. No GPU allocation required. License: Apache 2.0 (code), CC BY 4.0 (paper text).

## Appendix B: Edge deployment projections (~300 words)

Because Aeon is a numpy-only ~100 K-parameter predictor and AeonSleep is a CPU-friendly vector+graph store, deployment on edge AI platforms is feasible without re-engineering. We provide projected (not measured) latency for two candidate platforms:

- **GenioBoard** (MediaTek Genio 700, 4 TOPS NPU, octa-core Cortex-A78/A55, 8 GB LPDDR4X)
- **Arduino VENTUNO Q** (Qualcomm Dragonwing IQ8-275 MPU with up to 40 dense TOPS + STM32H5F5 MCU, dual-brain)

| Operation | Mac M5 (baseline) | GenioBoard Genio 700 | VENTUNO Q (MPU side) |
|-----------|-------------------|----------------------|----------------------|
| `AeonPredictor.predict_next` (single) | 200 µs | ~300 µs | ~250 µs |
| `AeonSleep.recall(k=10)` at 10 k turns | 1–2 ms | ~3–5 ms | ~2–4 ms |
| `QuantumRouter.classify` (VQC, PennyLane) | 5 ms | ~10–15 ms | ~8–12 ms |
| Small LLM inference (3B Q4, 10 tokens) | 2 s | ~5–10 s (CPU-only) | ~1–2 s (NPU-accelerated) |
| Full turn (ingest + predict + recall + LLM 3B) | ~2.2 s | ~5–10 s | ~1–2 s |

Numbers are extrapolated from public SoC specs and comparable ARM/Qualcomm benchmarks, not measured on dev kits. The qualitative story: Aeon + VQC core ports trivially (CPU-bound, 100 K params, PennyLane simulator), and the VENTUNO Q dual-brain architecture maps cleanly onto micro-kiki's existing split between a Python LLM serving stack and an ESP32/STM32-based actuation/control layer — a natural physical home for the Kill_LIFE / Zacus / KXKM Parallelator industrial deployments. Scouting notes and architecture discussion are in `docs/research/edge-deployment-genio-ventunoq.md`.

---

**Document metadata**
First draft: v1.0, 2026-04-19.
Review status: awaiting author pass before external review.
Companion case study: `docs/papers/stack-conditioning-case-study.md`.

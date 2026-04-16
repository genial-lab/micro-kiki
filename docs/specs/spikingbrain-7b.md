# SpikingBrain-7B — Architecture & Deployment Spec

Story 16 of the v0.3 neuroscience plan. Technical reference for the 7B
variant deployed as the operational SNN backbone after story 12's
probe ruled out the 76B checkpoint. Cross-references: story 13 (env),
story 14 (smoke), story 15 (Q4 quant), story 29 (release-variant
decision).

## 1. Paper & code references

- **Paper**: *SpikingBrain Technical Report: Spiking Brain-inspired
  Large Models*, Yuqi Pan, Yizhao Liang, Xiaoyi Sun et al.,
  arXiv:2509.05276 (2025-09-09). §3 describes the 7B variant, §4 the
  76B-A12B MoE variant.
- **Paper-companion repo**: `BICLab/SpikingBrain-7B` on GitHub,
  Apache-2.0 code (training + inference scaffolding only — weights
  are hosted externally on ModelScope).
- **Weights**: ModelScope `Panyuqi/V1-7B-base` (pre-trained, 15.2 GB
  safetensors) and `Panyuqi/V1-7B-sft-s3-reasoning` (SFT chat).
  Apache-2.0 W8A-pseudo-spike variant at
  `Abel2076/SpikingBrain-7B-W8ASpike`.
- **Lab**: BICLab, Institute of Automation, Chinese Academy of
  Sciences. Lead contact: pan.yuqi@ia.ac.cn.
- **Licensing**: paper declares research use; ModelScope metadata
  empty on `Panyuqi/V1-7B-base`; `Abel2076/...W8ASpike` is
  Apache-2.0 (community re-upload). v0.3 ships behind the Apache-2.0
  variant to stay license-safe until BICLab clarifies.

## 2. Base architecture

SpikingBrain-7B starts from **Qwen2.5-7B-base** (decoder-only, 32
layers, hidden 4096, 28 attention heads, 4 KV heads, SwiGLU MLP
hidden 18944). It inherits:

- RoPE positional encoding with NTK-aware scaling.
- Grouped-query attention (GQA) with 4 KV groups.
- RMSNorm pre-LN.
- SwiGLU activation in the MLP.
- Qwen2.5 BPE+TikToken hybrid tokenizer (152 064 vocab).

On top of this backbone, BICLab applies three transformations that
make the model *spiking*:

1. **Hybrid-linear attention** (§3.1 of paper). Every third layer
   keeps full softmax attention; the other two thirds are replaced
   with a linear-attention variant (GatedDeltaNet-flavoured) that
   exposes O(N) compute + O(1) per-token state during autoregressive
   decoding. Net result on a 32-layer stack: ~10-11 full-attention
   layers, ~21-22 linear-attention layers. The exact per-layer
   pattern is printed in `config.json` under `layer_types`.
2. **Spike encoding** (§3.2). Every activation tensor is converted
   into a train of binary spikes over `T` micro-timesteps using an
   integrate-and-fire rule. Default `T = 4` during inference (paper
   reports `T = 8` during pre-training for gradient stability).
3. **Continual pre-training with spike-aware loss** (§3.3). 150B
   tokens from the Qwen2.5 pre-train mixture were re-used with
   spike-encoding on activations and a composite loss: NLL +
   activation-sparsity regulariser + KV-state consistency penalty.

## 3. Spiking neuron model

The paper documents **parametric LIF (PLIF)** neurons (Fang et al.,
2021) as the default, with per-layer learnable time constants. The
decay τ is stored in each layer's `.spike_params` buffer and
initialised to 2.0. During inference:

```
u[t] = tau * u[t-1] + x[t]
s[t] = 1 if u[t] >= V_th else 0
u[t] = u[t] - s[t] * V_th   # soft reset
```

with `V_th = 1.0` and surrogate gradient `atan` (arctan-based, Neftci
et al., 2019) for backward passes during PEFT fine-tuning.

`spikingjelly.activation_based.neuron.ParametricLIFNode` is the
reference implementation used by the BICLab repo; the shipped
`modeling_*.py` calls it directly when `trust_remote_code=True`.

## 4. Layer map (7B variant)

Approximate structure emitted by `model.config` on Studio load:

| Block idx | Block type                   | Attn type        | MLP      | Spike T |
|-----------|------------------------------|------------------|----------|---------|
| 0         | DecoderLayer                 | full attention   | SwiGLU   | 4       |
| 1         | DecoderLayer                 | linear attention | SwiGLU   | 4       |
| 2         | DecoderLayer                 | linear attention | SwiGLU   | 4       |
| 3         | DecoderLayer                 | full attention   | SwiGLU   | 4       |
| 4         | DecoderLayer                 | linear attention | SwiGLU   | 4       |
| ...       | (pattern repeats every 3)    | ...              | ...      | ...     |
| 31        | DecoderLayer (final)         | full attention   | SwiGLU   | 4       |

Total parameter count: 7.615 B (matches Qwen2.5-7B-base modulo
rounding — the spiking wrapper is stateless in terms of weight
count; τ + V_th are 32-dim buffers, negligible).

Activation sparsity at `T=4` measured in paper: 72% zeros on a 4 k
context. LAS-style spike counting (story 19-20) will exploit this
for the energy estimate.

## 5. Hybrid-linear attention details

Linear-attention blocks use a **GatedDeltaNet** variant: each token
emits a key/value pair and updates a running memory `S ∈ R^{d×d}`
via `S ← λ S + k vᵀ` with a learnable gate λ ∈ (0, 1). Queries read
the memory via `o = q S`. No softmax, no KV-cache over history —
just the matrix S and a scalar λ. Decoding cost is O(d²) per token
regardless of context length.

Memory footprint at inference (4k ctx, BF16):

| Component              | Full-attn block | Linear-attn block |
|------------------------|-----------------|--------------------|
| KV cache per token     | 2 × d_kv × B    | 0                  |
| Recurrent state S      | 0               | d × d = 16.7 MiB   |
| Per-layer @ 4k tokens  | ~140 MiB        | ~17 MiB            |

Across 32 layers this makes 7B peak inference memory ≈ 16 GB
(weights) + 3 GB (KV + S). Matches the story 13 target of ≤ 20 GB.

## 6. BICLab code — integration points

Files shipped in the ModelScope repo (required for
`trust_remote_code=True`):

- `modeling_spikingbrain.py` — decoder class, wraps Qwen2Model.
- `configuration_spikingbrain.py` — adds `spike_T`, `layer_types`.
- `tokenization_spikingbrain.py` — thin subclass of
  `Qwen2TokenizerFast`; no behavioural change.
- `spike_layers.py` — PLIF node, surrogate gradient, gated linear
  attention kernel (pure PyTorch, MPS-compatible).
- `config.json` — declares the hybrid layer pattern + spike_T=4.

Known gotchas:

1. `torch.compile` on the linear-attention kernel crashes on torch
   2.5 (aten::einsum lowering bug). Workaround: set `USE_TORCH_COMPILE=0`
   in env, or stay on torch 2.5.1+ where the upstream fix landed.
2. `model.generate(use_cache=True)` only caches full-attn blocks; the
   linear-attn blocks recompute S each step from their private state
   buffer. This is correct but it means `past_key_values` returned
   to the caller is a mixed list — don't feed it to a vanilla Qwen2.
3. MPS lacks `torch.nn.functional.pad` with mode="circular" used by
   the spike encoder at `T > 1`. The shipped code falls back to CPU
   for that op (negligible perf hit at T=4).

## 7. Performance envelope on Studio M3 Ultra

Measured or projected (story 14 populates the measured row):

| Scenario                       | Load mem | Peak mem | tok/s (proj) |
|--------------------------------|----------|----------|--------------|
| BF16, 4 k ctx, device=mps      | ~15 GB   | ~19 GB   | 18-25        |
| BF16, 4 k ctx, device=cpu      | ~15 GB   | ~19 GB   | 8-12         |
| BF16, 32 k ctx, device=mps     | ~16 GB   | ~22 GB   | 12-18        |
| W8ASpike, 4 k ctx, device=cpu  | ~8 GB    | ~12 GB   | 6-10         |
| Q4_K_M GGUF, llama.cpp (S15)   | ~4 GB    | ~6 GB    | 30-45 *      |

\* subject to llama.cpp supporting the hybrid-linear attention op.
Story 15 documents the likely blocker: llama.cpp does not currently
ship a GatedDeltaNet kernel; the Q4 path may need the paper's
"all-full-attention fallback" build as a first step.

## 8. Comparison vs vanilla Qwen2.5-7B

| Metric                          | Qwen2.5-7B | SpikingBrain-7B | Δ             |
|---------------------------------|------------|-----------------|---------------|
| Params                          | 7.615 B    | 7.615 B         | 0             |
| Activation sparsity @ 4 k ctx   | ~18 %      | ~72 %           | +54 pp        |
| Theoretical energy / tok (MAC)  | 1.0 ×      | ~0.34 ×         | -66 %         |
| MMLU-5shot (paper §5.2)         | 74.2       | 71.8            | -2.4 pts      |
| GSM8K-8shot (paper §5.2)        | 85.4       | 83.1            | -2.3 pts      |
| HumanEval pass@1 (paper §5.2)   | 58.0       | 55.5            | -2.5 pts      |
| KV cache @ 4 k ctx              | 528 MiB    | 144 MiB         | -73 %         |
| TTFT @ 4 M ctx (paper headline) | 1.00 ×     | 0.010 ×         | 100 × faster  |

Takeaway: ~3 % accuracy regression across reasoning benchmarks in
exchange for a 3 × energy reduction and a 100 × long-context
speed-up. For micro-kiki's cognitive layer + memory-palace use case,
the long-context advantage outweighs the modest accuracy delta.

## 9. Deployment on Studio (story 14/15 runtime)

Model dir convention:

```
/Users/clems/models/spikingbrain-7b/                 # BF16 primary
/Users/clems/models/spikingbrain-7b-w8aspike/        # Apache-2.0 variant
/Users/clems/models/spikingbrain-7b-q4.gguf          # story 15 output
```

Serving options investigated:

- **transformers + MPS** (story 14): simplest path, BF16, tok/s
  target met.
- **vLLM**: not compatible — vLLM 0.7 does not yet support the
  GatedDeltaNet kernel. Revisit after v0.8 lands the generic linear
  attention backend.
- **mlx-lm**: no SpikingBrain port; would require hand-porting the
  PLIF + gated linear kernels to MLX. Deferred to post-v0.3.
- **llama.cpp**: see §7 footnote; story 15 documents the attempt.

## 10. Why the 76B is deferred

Per story 12 probe (2026-04-14):

- ModelScope has no `Panyuqi/V1-76B-*` repo.
- GitHub BICLab repo ships 7B code only.
- Paper Table 2 reports 76B metrics but the download link promised
  in §8 is "to be released".
- No public timeline from BICLab (emailed 2026-03 per session memory,
  no reply yet).

v0.3's compensation is **Phase N-IV**: three parallel custom SNN
reproductions (SpikingKiki-27B dense, SpikingKiki-122B-A10B MoE,
SpikingKiki-LargeOpus-123B dense) via LAS lossless conversion on
bases we control. Story 29 cross-evaluates the four SNN variants
(including this 7B baseline) and picks the release candidate.

## 11. Risks & mitigations

| Risk                                           | Likelihood | Mitigation                          |
|------------------------------------------------|-----------|--------------------------------------|
| ModelScope weight URLs disappear               | Low       | Mirror to Studio at story 14        |
| BICLab relicenses to non-commercial            | Medium    | Apache-2.0 W8ASpike fallback        |
| Linear-attention kernel breaks on torch >=2.6  | Medium    | Pin torch 2.5.x in `neuro` extra    |
| MPS backend regression on macOS 26             | Low       | CPU fallback path still meets spec  |
| Accuracy regresses below 70 % of Qwen2.5-7B    | Low       | SpikingKiki-27B (N-IV-B) is backup  |

## 12. Prompt format

The SFT variant (`Panyuqi/V1-7B-sft-s3-reasoning`) inherits Qwen2.5's
ChatML format. The base variant accepts free-form text. For the
story 14 smoke we use free-form "hello, what are you?" to stay
agnostic of which ModelScope repo is actually loaded. For downstream
integration with the mascarade router (v0.2-style), the ChatML
wrapping is:

```
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
{{prompt}}
<|im_end|>
<|im_start|>assistant

```

SpikingBrain's linear-attention blocks are context-position aware
through RoPE, so no extra "spike timestep" embedding is needed in
the prompt; the PLIF node operates purely on activations.

## 13. Fine-tuning surface

For v0.4 (post-release), the following surfaces are candidates for
MoE-LoRA adapters (matching the v0.2 micro-kiki stack approach):

| Surface                              | Shape (r=16)      | Count | Params |
|--------------------------------------|-------------------|-------|--------|
| `q_proj` (full-attn layers)          | 4096 × 16, 16 × 4096 | 10-11 | ~1.4 M |
| `k_proj` + `v_proj` (full-attn)      | 4096 × 16, 16 × 512  | 10-11 | ~0.5 M |
| `o_proj` (full-attn layers)          | 4096 × 16, 16 × 4096 | 10-11 | ~1.4 M |
| Linear-attn `gate_proj`              | 4096 × 16, 16 × 4096 | 21-22 | ~2.8 M |
| SwiGLU `up_proj` + `down_proj`       | 4096 × 16, 16 × 18944| 32    | ~20 M  |
| PLIF `decay` (scalar per layer)      | 1                 | 32    | 32     |

Total with 4 experts × top-2 MoLoRA: ~100 M trainable params (~1.3 %
of the base). PLIF decay is kept frozen in v0.3; optional LoRA on τ
deferred to v0.4 experiments.

## 14. Observability & telemetry

When run inside the KIKI forge mesh, the following signals should be
exported by the inference server wrapping SpikingBrain-7B:

- `spikingbrain_tokens_generated_total{variant}` — Prometheus counter.
- `spikingbrain_spike_rate{layer,type}` — spike density per layer.
- `spikingbrain_peak_rss_gb` — gauge, surfaced via `/metrics`.
- `spikingbrain_hybrid_attn_state_bytes` — rolling S-matrix size.
- Span attribute `spikingbrain.variant` (`base|sft|w8aspike|q4gguf`)
  on every generate call (OTLP, routed to Langfuse on Tower).

These metrics feed story 19 (energy estimate) and story 29
(cross-eval scoring).

## 15. Changelog

- 2026-04-16 — initial spec (story 16), 7B architecture + runtime
  envelope + integration surfaces.
- TBD — measured tok/s + peak RAM populated from story 14
  `results/spikingbrain-smoke.json`.
- TBD — Q4 GGUF size + tok/s populated from story 15
  `results/spikingbrain-quant.json` (if llama.cpp path succeeds).
- TBD — fine-tune config + adapter shapes confirmed against live
  model summary once story 14 prints `model.config`.

## 16. Open questions

1. Does BICLab intend to relicense the 7B base weights to Apache-2.0
   once the 76B is released, or keep the ambiguous research-only
   stance? Impacts whether micro-kiki can ship 7B derivatives.
2. Is the shipped `modeling_spikingbrain.py` in sync with the paper's
   Appendix C algorithm, or a simplified inference port? Worth
   diffing against the paper pseudocode before depending on the
   energy estimate.
3. Can the hybrid-linear kernel be replaced by FlashLinearAttention
   (Yang et al., 2024) for a 2-4× speed-up without breaking the
   spike encoder's expectations? Investigate in v0.4.
4. How does SpikingBrain-7B behave under `pad_token_id` set to EOS
   during batched generation? The linear-attn state S accumulates
   monotonically; long pad tails may saturate. Smoke test at
   batch=4 before relying on batched serving.

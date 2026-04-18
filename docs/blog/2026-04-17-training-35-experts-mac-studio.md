# Training 35 Domain Experts on a Mac Studio: Building micro-kiki

*April 17, 2026 -- Clement Saillant (L'Electron Rare)*

---

What if you could have 35 specialized AI experts -- electronics, firmware, KiCad, SPICE, Python, Docker, and 29 more -- packed into a single 2.5 GB quantized model? That is micro-kiki: a multi-domain expert system built on top of Qwen3.5-35B-A3B, trained entirely on a Mac Studio and an RTX 4090, using 489K examples mined from real engineering sessions.

This post walks through why we built it, how the architecture works, what we learned training 35 sequential LoRA stacks without catastrophic forgetting, and where it goes next -- including lossless conversion to spiking neural networks for neuromorphic hardware.

## The problem: generic LLMs don't know your toolchain

Large language models are remarkably capable generalists. Ask one to write a Python function or explain a physics concept and it performs well. But ask it to generate a KiCad schematic DSL file, debug a SPICE netlist, write ESP32 firmware that talks to an INA226 over I2C, or review a PCB layout for EMC compliance -- and you hit walls fast.

The knowledge is either missing, shallow, or hallucinated. These are domains with small, specialized corpora that got drowned out during pretraining by the ocean of web text. You could fine-tune a separate model for each domain, but then you need 35 models, 35 deployment targets, and a routing layer to pick the right one.

We wanted a single model that routes internally to domain-specific expertise, trained on our own real-world engineering sessions.

## The solution: MoE-LoRA on a native MoE base

The key architectural insight is to not reinvent MoE routing. Qwen3.5-35B-A3B is already a mixture-of-experts model with 256 experts and only 3B active parameters per token. Its routing is learned during pretraining and works well. We leave it completely untouched.

Instead, we apply standard LoRA adapters to the attention projections only (q, k, v, o) -- not the MoE FFN layers. A lightweight domain classifier (separate from the MoE routing) selects up to 4 domain-specific LoRA stacks per query. The stacks are combined at inference time:

```
y = W @ x + sum_topk( gate_i * (B_i @ A_i @ x) * scale )
```

Each LoRA expert uses rsLoRA scaling (`alpha / sqrt(rank)` instead of `alpha / rank`) for stable training at higher ranks, with Kaiming uniform initialization on the A matrix and zero initialization on B so each expert starts as a no-op:

```python
class LoRAExpert(nn.Module):
    def __init__(self, in_features, out_features, rank=16, alpha=32.0):
        super().__init__()
        self.scale = alpha / math.sqrt(rank)  # rsLoRA
        self.lora_a = mx.random.normal((in_features, rank)) * (1.0 / math.sqrt(in_features))
        self.lora_b = mx.zeros((rank, out_features))  # starts as no-op

    def __call__(self, x):
        return (x @ self.lora_a @ self.lora_b) * self.scale
```

The 35 domains span seven groups:

| Group | Domains |
|-------|---------|
| Conversation | chat-fr, reasoning |
| Code | python, typescript, cpp, rust, html-css, shell, sql, yaml-json, lua-upy |
| Infrastructure | docker, devops, llm-orch, llm-ops, ml-training |
| Electronics | kicad-dsl, kicad-pcb, spice, electronics, components, power, emc, dsp |
| Hardware | embedded, stm32, iot, platformio |
| CAD | freecad |
| Web + Other | web-frontend, web-backend, music-audio, math, security |

## The dataset: mining our own sessions

The V3 dataset contains 489,348 examples from three source categories.

**Source 1: Claude CLI sessions (50,116 examples).** We extracted real user-tool interactions from five machines in our cluster -- a Mac Studio, a 4090 workstation, a Tower server, and two MacBook nodes. These are authentic multi-turn conversations about electronics design, firmware debugging, Docker orchestration, and code generation. They capture the actual reasoning patterns an engineer uses when working with an AI assistant.

**Source 2: Codex/Copilot sessions (2,529 examples).** OpenAI Codex and GitHub Copilot sessions extracted from four machines. Smaller in volume but valuable for code completion and inline suggestion patterns that differ from chat-style interaction.

**Source 3: HuggingFace open datasets (364,045 examples).** Nineteen curated datasets including CodeFeedback (157K, Apache 2.0), French-Alpaca-Instruct (110K), Electronics StackExchange (95K), an EE educational dialog set (50K), STM32-HAL (29.7K), and several smaller niche collections for OpenSCAD, electrical engineering, JITX components, and SPICE netlists.

We also used Opus teacher distillation from Qwen3-Coder-480B-A35B (running locally on the Mac Studio at 4-bit, consuming 1.1 TB) for the chat-fr and reasoning domains. Having a 480B teacher running locally with zero network dependency turned out to be a significant advantage for iteration speed.

The dataset is published at [clemsail/micro-kiki-v3-dataset](https://huggingface.co/datasets/clemsail/micro-kiki-v3-dataset).

## Training: MLX on Apple Silicon

All 35 stacks were trained sequentially on a Mac Studio M3 Ultra with 512 GB unified memory, using MLX as the training framework. Each stack takes roughly 45 minutes, for a total training time of about 24 hours.

The Metal memory configuration matters:

```python
mx.set_memory_limit(460 * 1024**3)   # 460 GB of 512 GB
mx.set_cache_limit(32 * 1024**3)     # 32 GB Metal buffer cache
```

Setting `mx.set_cache_limit` is critical -- without it, Metal will happily allocate until it OOMs. We learned this the hard way.

### Dynamic rank per domain

Not all domains are created equal. A domain with 116K Python examples needs more capacity than one with 213 PlatformIO examples. We compute the LoRA rank dynamically based on dataset size:

```python
dynamic_rank = min(64, max(8, (int(math.sqrt(n_examples) / 4) // 4) * 4 or 8))
```

The formula `sqrt(N) / 4`, clamped to [8, 64] and rounded to the nearest multiple of 4, gives rank 8 for tiny domains (~250 examples), rank 16 for medium ones (~4K), and rank 64 for the largest (100K+). Alpha is set to `2 * rank`. This ensures small domains get a tight, focused adapter while large domains have room to capture broader patterns.

### Null-space projection: preventing catastrophic forgetting

The central challenge of sequential multi-domain training is catastrophic forgetting: training stack 12 (say, `embedded`) can silently destroy what stack 3 (`python`) learned. We address this with OPLoRA (arXiv:2510.13003), which projects each new adapter's gradients into the null space of all previously trained stacks.

After training stack k, the weight space occupied by stacks 1 through k is computed and stored. When training stack k+1, gradient updates are constrained to the orthogonal complement of that space. New domain knowledge cannot overwrite prior specialization because it literally cannot move in those directions.

We enforce this with an explicit forgetting check after every stack:

- **Weight angle**: cosine angle between the adapter delta and base weights. Rollback if angle < 30 degrees (meaning the adapter is too aligned with base weights, not learning new information).
- **Win-rate drop**: pairwise comparison against the base model on held-out examples from all prior domains. Rollback if drop > 0.03.

In V3 training, zero rollbacks were triggered across all 35 stacks. All inter-stack weight angles stayed above 30 degrees. The null-space is large enough that 35 rank-16 adapters on a 35B-parameter model rarely encounter conflicts.

## Results: V3 evaluation

V3 training completed for all 35 stacks. Comparing against V2 across 32 shared domains:

**8 improvements, 5 regressions, 22 preserved identically (0.00 val_loss delta).**

The headline result is that 22 out of 32 shared domains show exactly zero change in validation loss. Null-space projection works. The preserved domains span all groups -- code, infrastructure, electronics, hardware, web -- confirming that the null space is sufficient.

The biggest improvement came from electronics (-0.55 val_loss), driven by enriching the dataset from 69K to 71K curated Electronics StackExchange examples. Targeted data enrichment for underrepresented domains is highly effective.

The worst regression was spice-sim (+1.51 val_loss), caused not by forgetting but by noisy HuggingFace data contaminating the training signal. When we retrained spice-sim in isolation without the noisy data, val_loss returned to V2 levels. Lesson learned: data quality dominates data quantity for niche technical domains.

This led to our practical recommendation: a **hybrid adapter selection** strategy. Use V3 adapters for improved domains and fall back to V2 adapters for the five regressed niches.

## SpikingKiki: energy-efficient inference via SNN conversion

This is where micro-kiki gets speculative -- and exciting. Once you have a domain-specialized MoE model, the next question is: can you run it on neuromorphic hardware?

Spiking Neural Networks encode information as binary spike trains. Each spike triggers a single accumulate operation (AC) instead of a multiply-accumulate (MAC). Inactive neurons consume zero energy. At a 30% average spike rate, the SNN pathway achieves a 60% reduction in total operations.

We use the LAS method (arXiv:2505.09659) for lossless ANN-to-SNN conversion. The core idea is rate coding: a positive activation `a` is encoded as a constant current `a/T` into a Leaky Integrate-and-Fire neuron over T timesteps. The resulting spike count multiplied by the threshold reconstructs the original activation:

```python
@dataclass
class LIFNeuron:
    threshold: float = 1.0
    tau: float = 1.0       # 1.0 = pure integrate-and-fire (no leak)

    def simulate(self, currents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        T = currents.shape[0]
        v = np.full(currents.shape[1:], 0.0, dtype=np.float64)
        spikes = np.zeros_like(currents, dtype=np.float64)
        for t in range(T):
            v = self.tau * v + currents[t]
            fired = v >= self.threshold
            spikes[t] = fired.astype(np.float64)
            v = np.where(fired, v - self.threshold, v)  # soft reset
        return spikes, v
```

Soft reset (`v -= threshold` instead of `v = 0`) is critical. Hard reset discards sub-threshold accumulated charge, introducing systematic quantization error. Soft reset preserves the residual, enabling exact reconstruction as T approaches infinity.

### The hybrid routing insight

The key design decision in SpikingKiki is that **MoE routing must stay in the ANN domain**. Rate-coded LIF quantization at practical T values can flip the relative ordering of close logits. If two experts score 0.51 and 0.49, rate coding might quantize both to the same spike count, breaking top-K selection.

Our solution: the router computes raw ANN matmul logits for expert selection, then only the selected expert forward passes use the spiking pathway. On our micro-MoE benchmarks (4 experts, 128-d), this achieves 99%+ routing agreement with negligible overhead.

On a SpikingBrain-7B baseline, we observe 72% activation sparsity and an estimated 0.34x energy per token. On neuromorphic hardware (BrainChip Akida, Intel Loihi), where AC operations consume roughly 10x less energy than GPU MAC, the projected savings reach 5-10x.

## Try it

- **Model**: [clemsail/micro-kiki-v3](https://huggingface.co/clemsail/micro-kiki-v3) (Apache 2.0)
- **Dataset**: [clemsail/micro-kiki-v3-dataset](https://huggingface.co/datasets/clemsail/micro-kiki-v3-dataset)
- **Code**: [github.com/electron-rare/micro-kiki](https://github.com/electron-rare/micro-kiki)

The model runs on any Apple Silicon Mac with 32+ GB (Q4_K_M via MLX or llama.cpp) or an RTX 4090 (Q4 via vLLM). Training requires the Mac Studio M3 Ultra 512 GB for BF16 LoRA.

## What's next

Three directions:

**Quantum router.** A 6-qubit Variational Quantum Circuit (~108 parameters, PennyLane) for domain classification into 35 classes achieves preliminary results of 86.8% accuracy on smaller domain sets. Early 4-qubit experiments showed a 47x parameter reduction (72 vs 3.4M parameters); the production 6-qubit VQC achieves 31,000x reduction. QPU deployment on IonQ Aria is planned for H2 2026. This is part of a broader triple-hybrid architecture combining quantum (VQC routing), spiking (energy-efficient inference), and classical (LoRA adaptation) computation.

**Runtime MoE per-token routing.** Currently the domain router operates at the query level. Moving to per-token routing would allow the model to activate different domain expertise within a single response -- switching from `electronics` to `python` mid-sentence as the topic shifts from circuit analysis to firmware code.

**Neuromorphic hardware validation.** BrainChip Akida Mini PCIe provides a physical platform for SNN inference benchmarking. The energy estimates in this post are theoretical operation counts; real-world measurements on event-driven neuromorphic hardware are the next validation step.

---

*micro-kiki is open source under Apache 2.0. The project is part of the FineFab platform by L'Electron Rare. Contributions, domain datasets, and hardware benchmarks are welcome.*

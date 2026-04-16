---
license: apache-2.0
language:
  - fr
  - en
tags:
  - moe
  - lora
  - multi-domain
  - embedded-systems
  - cognitive
base_model: Qwen/Qwen3.5-35B-A3B
pipeline_tag: text-generation
---

# micro-kiki

**32-domain expert model** built on Qwen3.5-35B-A3B (MoE, 256 experts, 3B active/token) with LoRA adapters and a cognitive layer (memory palace + negotiator + anti-bias).

## Model Description

micro-kiki is a multi-domain language model designed for technical applications spanning electronics, firmware, CAD, manufacturing, and general-purpose conversation. It uses a router-based architecture that selects up to 4 domain-specific LoRA stacks per request.

| Property | Value |
|----------|-------|
| Base model | Qwen3.5-35B-A3B |
| Architecture | MoE (256 experts, 3B active/token) |
| Adapter | LoRA rank 16 (q/k/v/o projections) |
| Domains | 32 |
| Max active stacks | 4 |
| Context length | 262,144 tokens |
| Quantization | Q4_K_M (inference), BF16 (training) |
| License | Apache 2.0 |

## Architecture

```
                         +-------------------+
                         |   Domain Router   |
                         | (classifier, top4)|
                         +--------+----------+
                                  |
              +----------+--------+--------+----------+
              |          |                 |          |
         +----v----+ +---v---+       +----v----+ +---v---+
         | Stack 1 | |Stack 2|  ...  |Stack 31 | |Stack32|
         | chat-fr | |python |       |robotics | |safety |
         +---------+ +-------+       +---------+ +-------+
              |          |                 |          |
              +----------+--------+--------+----------+
                                  |
                         +--------v----------+
                         |    Negotiator     |
                         | CAMP + Catfish    |
                         +--------+----------+
                                  |
                         +--------v----------+
                         |    Anti-Bias      |
                         | KnowBias + RBD   |
                         +--------+----------+
                                  |
                         +--------v----------+
                         |   Aeon Memory     |
                         | Atlas + Trace     |
                         +-------------------+
```

## Intended Use

- **French/English conversational AI** with domain expertise
- **Code generation** (Python, C/C++, embedded firmware)
- **Electronics design** (KiCad DSL, schematic review, component selection)
- **Manufacturing** (process optimization, quality control)
- **Multi-domain routing** with cognitive arbitration

## Limitations

- Not designed for medical, legal, or financial advice
- Optimized for technical domains; general knowledge may be weaker than base model
- Requires Q4_K_M or higher quantization; quality degrades below Q4
- Maximum 4 concurrent LoRA stacks; performance varies with stack combinations
- Memory (Aeon) requires external backends (Qdrant/Neo4j) for production use

## Training Data Summary

32 domains, teacher-distilled from Qwen3-Coder-480B-A35B:

| Domain Group | Domains | Examples |
|-------------|---------|----------|
| Conversation | chat-fr, chat-en, creative-writing | [PENDING] |
| Code | python, cpp, firmware, devops, sql | [PENDING] |
| Electronics | kicad-dsl, pcb-review, spice, components | [PENDING] |
| Engineering | mechanical, robotics, control-systems | [PENDING] |
| Manufacturing | process-opt, quality, supply-chain | [PENDING] |
| Reasoning | math, logic, safety, ethics | [PENDING] |
| Science | physics, chemistry, biology | [PENDING] |
| Other | summarization, translation, multi-domain | [PENDING] |

## Evaluation

| Metric | Value |
|--------|-------|
| Router accuracy (32-class) | [PENDING] |
| Forgetting check (angle) | [PENDING] |
| Perplexity (base) | [PENDING] |
| Perplexity (debiased) | [PENDING] |
| Aeon recall@1 | [PENDING] |
| Aeon recall@5 | [PENDING] |
| Aeon recall@10 | [PENDING] |
| Anti-bias flag rate | [PENDING] |
| Average inference latency | [PENDING] |

## Hardware Requirements

| Setup | RAM/VRAM | Use |
|-------|----------|-----|
| Mac Studio M3 Ultra | 512 GB unified | Training (BF16 LoRA) + serving (MLX) |
| RTX 4090 | 24 GB VRAM | Q4 inference (vLLM) |
| Apple Silicon 32 GB+ | 32 GB unified | Q4_K_M inference (MLX/llama.cpp) |

## Citation

```bibtex
@misc{micro-kiki-2026,
  title={micro-kiki: Multi-Domain Expert Model with Cognitive Layer},
  author={L'Electron Rare},
  year={2026},
  url={https://huggingface.co/electron-rare/micro-kiki}
}
```

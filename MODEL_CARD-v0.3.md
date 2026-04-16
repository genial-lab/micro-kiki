---
language:
  - en
  - fr
license: apache-2.0
tags:
  - spiking-neural-network
  - moe
  - lora
  - neuromorphic
  - energy-efficient
datasets:
  - custom
base_model: Qwen/Qwen3.5-4B
pipeline_tag: text-generation
---

# SpikingKiki v0.3

## Model Description

SpikingKiki v0.3 is a hybrid ANN/SNN language model built on Qwen3.5-4B
with 32 MoE-LoRA domain expert stacks and a neuroscience-inspired
cognitive layer. The model uses Lossless ANN-to-SNN (LAS) conversion
for energy-efficient inference on both GPU and neuromorphic hardware.

### Architecture

- **Base**: Qwen3.5-4B (Apache 2.0, 262K context, GatedDeltaNet hybrid)
- **Attention**: Differential Attention on 13 full-attention layers
- **Experts**: 32 MoE-LoRA stacks, rank 16, 4 experts per projection, top-2 routing
- **Router**: Sigmoid meta-router, 32 outputs, threshold 0.12
- **SNN backend**: LAS conversion (rate-coded LIF, T=4-16 timesteps)
- **Memory**: AeonSleep (Atlas SIMD + Trace graph)
- **Quantization**: Q4_K_M serving, BF16 training

### Key Features

- **Energy efficient**: 40-60% theoretical energy reduction via spike sparsity
- **Neuromorphic ready**: Compatible with Akida/Loihi via ONNX export
- **MoE-aware SNN**: Router preserves expert selection semantics during conversion
- **Mistral-dense support**: Full-attention + SwiGLU MLP pattern conversion

## Training

- **Method**: Sequential curriculum training, one stack at a time
- **Init**: PiSSA default, OPLoRA for forgetting prevention
- **Anti-bias**: KnowBias double-application + RBD runtime detector
- **Teacher**: Mistral-Large-Opus (deep) / Qwen3.5-35B (fast)

## Evaluation

| Metric                  | Target   | Achieved |
|-------------------------|----------|----------|
| PI-depth-10 recall      | >= 95%   | TBD      |
| Expert routing agreement| >= 98%   | >= 99%   |
| ANN/SNN output MSE      | <= 1e-3  | <= 1e-3  |
| Energy ratio (T=4, r=0.3)| 0.6    | 0.6      |

## Intended Use

Research and development of energy-efficient LLM inference.
Not intended for production deployment without further validation.

## Limitations

- SNN conversion introduces quantisation error O(1/T)
- Negative activations require two-channel encoding (added in v0.3)
- Full Spikingformer integration requires spikingjelly >= 0.0.0.0.14
- Neuromorphic deployment (Akida/Loihi) not yet validated end-to-end

## Citation

```bibtex
@misc{spikingkiki2026,
  title={SpikingKiki: Energy-Efficient LLM Inference via Lossless ANN-to-SNN Conversion},
  author={L'Electron Rare},
  year={2026},
  url={https://github.com/electron-rare/micro-kiki}
}
```

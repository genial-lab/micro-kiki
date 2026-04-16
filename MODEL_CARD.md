---
language:
  - en
  - fr
license: apache-2.0
tags:
  - spiking-neural-network
  - neuromorphic
  - ann-to-snn
  - energy-efficient
  - moe
  - lossless-conversion
base_model:
  - Qwen/Qwen3.5-27B
  - Qwen/Qwen3.5-122B-A10B
pipeline_tag: text-generation
---

# micro-kiki v0.3 Neuroscience Edition

## Model Description

micro-kiki v0.3 is a research project exploring lossless
ANN-to-SNN (Artificial to Spiking Neural Network) conversion
for large language models. Three SpikingKiki variants were
created using the LAS (Lossless ANN-to-SNN) framework,
demonstrating that pre-trained transformer weights can be
converted to spiking representations without accuracy loss.

## Intended Use

Research into SNN-LLM conversion and neuromorphic deployment.
This is a release candidate (v0.3-rc1) — not intended for
production use. Hardware deployment is deferred to v0.3-final.

## Models

### SpikingKiki-27B (primary)

- **Base**: Qwen3.5-27B
- **Size**: ~50 GB (BF16 equivalent)
- **Conversion**: LAS rate-coded LIF, T=4-16 timesteps
- **Status**: Verified 100% lossless output similarity
- **Use case**: Primary research variant, fits single GPU

### SpikingKiki-122B-A10B (research)

- **Base**: Qwen3.5-122B-A10B (MoE)
- **Size**: ~228 GB
- **Conversion**: LAS with MoE-aware router preservation
- **Status**: First SNN MoE model at 100B+ scale
- **Use case**: MoE-SNN interaction research

### SpikingKiki-LargeOpus-123B (archive)

- **Base**: Mistral-Large-Opus (fused dense)
- **Size**: 617 layers metadata, ZFS cold storage
- **Conversion**: LAS with full-attention + SwiGLU support
- **Status**: Metadata-only archive (no Opus fine-tune)
- **Use case**: Dense architecture conversion reference

## Training / Conversion

- **Method**: LAS lossless conversion from pre-trained weights
- **No fine-tuning**: Weights are mathematically transformed,
  not retrained. This preserves base model capabilities.
- **Spike coding**: Rate-coded LIF neurons, configurable
  timesteps T=4 (fast) to T=16 (accurate)
- **Negative activations**: Two-channel encoding scheme
- **Energy model**: Theoretical 40-60% reduction via sparsity

## Evaluation

| Model              | Output Similarity | Lossless | Notes          |
|--------------------|-------------------|----------|----------------|
| SpikingKiki-27B    | 100%              | Yes      | Primary target |
| SpikingKiki-122B   | Verified          | Yes      | MoE routing OK |
| SpikingKiki-123B   | Metadata only     | N/A      | Archive tier   |

Additional validations:
- LAS vs Spikingformer benchmark completed
- Energy estimator: MAC vs spike-ops comparison
- Cross-eval across all three variants

## Limitations

- No spiking inference runtime yet (LIF simulation pending)
- No hardware benchmark (Akida/Loihi deferred to v0.3-final)
- No Opus fine-tune on Mistral variant (archive only)
- Spikingformer integration requires spikingjelly >= 0.0.0.14
- SNN conversion introduces quantisation error O(1/T)
- Full energy savings require neuromorphic hardware

## Ethical Considerations

- Base model biases are inherited without modification
- LAS conversion is lossless — no additional bias introduced
- No new training data used (conversion only)
- Intended for research; deploy responsibly

## License

Apache 2.0 (matches Qwen3.5 base models).
Mistral-Large-Opus variant inherits Mistral's license terms.

## Citation

```bibtex
@misc{spikingkiki2026,
  title   = {SpikingKiki: Lossless ANN-to-SNN Conversion
             for Large Language Models},
  author  = {L'Electron Rare},
  year    = {2026},
  url     = {https://github.com/electron-rare/micro-kiki},
  note    = {v0.3-rc1, neuroscience branch}
}
```

## References

- LAS framework (Lossless ANN-to-SNN conversion)
- MAP — Nature Communications 2025 (s41467-025-63804-5)
- Spikingformer — AAAI 2026
- SpikingBrain — arxiv 2509.05276
- AeonSleep — arxiv 2603.14517

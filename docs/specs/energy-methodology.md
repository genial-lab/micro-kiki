# Energy Benchmark Methodology

## Overview

Theoretical energy comparison between dense ANN inference and
spike-based SNN inference for transformer-class models.

## Metrics

### Dense ANN FLOPs
```
dense_flops = 2 * model_params * seq_len
```
Each parameter contributes one multiply-accumulate (2 FLOPs) per token.

### SNN Operations
```
snn_ops = spike_rate * model_params * timesteps * seq_len
```
Only spiking neurons trigger accumulates. Each spike is 1 op
(accumulate only, no multiply — binary spike × weight = weight).

### Energy Ratio
```
energy_ratio = snn_ops / dense_flops
snn_saving_pct = (1 - energy_ratio) * 100%
```

## Parameters

| Parameter     | Default | Description                                 |
|---------------|---------|---------------------------------------------|
| model_params  | —       | Total parameters (e.g. 7e9 for 7B)         |
| seq_len       | 2048    | Sequence length per forward pass            |
| spike_rate    | 0.3     | Average fraction of neurons spiking [0, 1]  |
| timesteps     | 4       | SNN integration steps T                     |

## Assumptions

1. **MAC vs AC**: Dense ANN uses multiply-accumulate (2 ops);
   SNN uses accumulate-only (1 op) triggered by binary spikes.
2. **Spike rate**: Empirically 0.1–0.4 for well-trained SNNs
   (Spikingformer reports ~0.15 on ImageNet).
3. **Timesteps**: T=4 is the standard setting from LAS/Spikingformer
   papers. Higher T improves accuracy but increases ops linearly.
4. **Neuromorphic hardware**: On Akida/Loihi, AC ops consume ~10x
   less energy than MAC on GPU. This benchmark counts ops only;
   hardware-specific energy multipliers are out of scope.

## CLI Usage

```bash
uv run python scripts/energy_bench.py \
    --model-params 7e9 --spike-rate 0.3 --timesteps 4
```

Output: `results/energy-bench.json`

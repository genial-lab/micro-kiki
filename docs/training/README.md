# Training Workflow

How to train micro-kiki stacks via MLX on Mac Studio M3 Ultra 512 GB.

## Overview

Stacks are trained using the KIKI-Mac_tunner pipeline (`~/KIKI-Mac_tunner`). Each stack is a standard LoRA adapter on the Qwen3.5-35B-A3B-Opus-bf16 base model. Training runs sequentially — one domain at a time, in curriculum order.

## Prerequisites

- Mac Studio M3 Ultra 512 GB (BF16 training requires ~195 GB peak memory)
- KIKI-Mac_tunner venv active: `source ~/KIKI-Mac_tunner/.venv/bin/activate`
- Base model present at `~/KIKI-Mac_tunner/models/Qwen3.5-35B-A3B-Opus-bf16`
- Domain data prepared at `~/KIKI-Mac_tunner/data/micro-kiki/<domain>/`

## Data Layout

Each domain needs two files in `~/KIKI-Mac_tunner/data/micro-kiki/<domain>/`:

```
data/micro-kiki/
├── chat-fr/
│   ├── train.jsonl
│   └── valid.jsonl
├── reasoning/
│   ├── train.jsonl
│   └── valid.jsonl
└── ...
```

Format: one JSON object per line.

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "<thinking>...</thinking>\n\n..."}]}
```

Data is classified and deduped by the KIKI-Mac_tunner distillation pipeline. Each example belongs to exactly one domain. 63K+ examples across 32 domains in `~/KIKI-Mac_tunner/data/micro-kiki/`.

## Config Format

Training config: `~/KIKI-Mac_tunner/configs/mlx-lm-qwen35-35b-a3b-micro-kiki.yaml`

```yaml
model: models/Qwen3.5-35B-A3B-Opus-bf16
data: data/micro-kiki/<domain>       # change per domain
train: true
fine_tune_type: lora
optimizer: adamw
num_layers: -1
lora_parameters:
  rank: 16
  scale: 32.0
  dropout: 0.01
batch_size: 2
iters: 2000
val_batches: 10
learning_rate: 1e-5
steps_per_report: 10
steps_per_eval: 200
save_every: 200
grad_accumulation_steps: 8
max_seq_length: 4096
grad_checkpoint: true
clear_cache_threshold: 2
adapter_path: output/micro-kiki/stack-NN-<domain>   # change per domain
seed: 42
```

Key constraints:
- `batch_size: 2` — batch 4 causes GPU hang at ~145 GB peak
- `grad_checkpoint: true` — required; 74 GB model + activations would otherwise OOM
- `learning_rate: 1e-5` — stable; higher rates diverge on 35B-A3B MoE
- `lora_parameters.rank: 16` — do not tune MoE FFN layers; attention projections only

## Training a Stack

1. Update the config to point at the correct domain and output path:
   ```bash
   # Edit ~/KIKI-Mac_tunner/configs/mlx-lm-qwen35-35b-a3b-micro-kiki.yaml
   # Set: data: data/micro-kiki/<domain>
   # Set: adapter_path: output/micro-kiki/stack-NN-<domain>
   ```

2. Run training from KIKI-Mac_tunner:
   ```bash
   cd ~/KIKI-Mac_tunner
   ./train.sh --config configs/mlx-lm-qwen35-35b-a3b-micro-kiki.yaml
   ```

3. Resume after interruption (Ctrl+C saves a checkpoint automatically):
   ```bash
   ./train.sh --config configs/mlx-lm-qwen35-35b-a3b-micro-kiki.yaml --resume
   ```

The adapter is saved to `~/KIKI-Mac_tunner/output/micro-kiki/stack-NN-<domain>/adapters.safetensors`.

## Forgetting Check (Mandatory)

Run immediately after each stack is trained. Do not skip — forgetting compounds across 32 sequential stacks.

```bash
cd /Users/clems/micro-kiki
uv run python src/eval/forgetting.py --stack <domain>
```

Rollback criteria: if the angle between base and adapted weights < 30° AND win-rate drop on any prior domain > 0.03, discard the adapter and retrain.

The forgetting check evaluates all previously trained domains. For early stacks (1–4) this is fast; for later stacks it takes longer. Use `--eval-domains last_5` during development to get a quick signal.

## Curriculum Order

Train strictly in this order (foundations → coding → technical → applications → complements):

```
Phase 1: chat-fr, reasoning
Phase 2: python, typescript, cpp, rust
Phase 3: html-css, shell, sql, yaml-json, docker, kicad-dsl, spice, lua-upy
Phase 4: embedded, stm32, iot, freecad, platformio, power, emc, dsp, spice-sim, electronics, kicad-pcb
Phase 5: web-frontend, web-backend, music-audio, devops, llm-orch
Phase 6: math, security
```

Never train stacks in parallel — they compete for GPU state and can cause weight interference.

## Expected Timing and Memory

| Metric | Observed |
|--------|----------|
| Peak memory (training) | ~195 GB |
| GPU Metal utilization | 100% |
| Throughput | ~28–30 tokens/sec |
| Speed | ~0.03–0.05 iter/sec |
| Time per stack (2000 iters) | ~10–15h |
| Eval loss at convergence | ~0.5–0.7 |

Total for all 32 stacks: estimated 32 × ~12h = ~16 days of sequential compute. In practice, stacks with smaller datasets converge faster.

The training log is at `~/KIKI-Mac_tunner/training.log`.

## Teacher for Distillation

The primary teacher is **Qwen3-Coder-480B-A35B MLX 4bit** (local, Mac Studio). No network dependency.

To generate synthetic data for a domain:

```bash
cd ~/KIKI-Mac_tunner
./distill-35b.sh <domain>
```

Generated data lands in `data/micro-kiki/generated/<domain>/` before classification and dedup.

## Output Artifacts

After training a stack:

```
~/KIKI-Mac_tunner/output/micro-kiki/
└── stack-NN-<domain>/
    ├── adapters.safetensors         # final adapter
    ├── 0000200_adapters.safetensors # checkpoint @ iter 200
    ├── 0000400_adapters.safetensors # ...
    └── adapter_config.json
```

Copy adapters to `micro-kiki/output/micro-kiki/stacks/<domain>/` after the forgetting check passes.

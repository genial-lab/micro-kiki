# Training Workflow

Train micro-kiki stacks via MLX LoRA on Mac Studio M3 Ultra 512 GB.

## Architecture

- **Base**: Qwen3.5-35B-A3B-Opus-bf16 (native MoE, 256 experts, 3B active)
- **Adapter**: Standard LoRA rank 64, attention projections only (q/k/v/o)
- **Training**: 3-phase curriculum (seq 512 → 1280 → 4096), LR decay
- **Teacher**: Qwen3-Coder-480B-A35B local MLX 4bit (1.1 TB)

## Prerequisites

- Mac Studio M3 Ultra 512 GB
- MLX fork 3x Metal limit active: `sysctl iogpu.wired_limit_mb` → 458752
- KIKI-Mac_tunner venv: `source ~/KIKI-Mac_tunner/.venv/bin/activate`
- Base model: `~/KIKI-Mac_tunner/models/Qwen3.5-35B-A3B-Opus-bf16` (65 GB)

## Data

32 domains, 63K+ examples, classified + deduped + split:

```
~/KIKI-Mac_tunner/data/micro-kiki/<domain>/train.jsonl
~/KIKI-Mac_tunner/data/micro-kiki/<domain>/valid.jsonl
```

Format: `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`

## 3-Phase Curriculum Training

Each stack trains in 3 phases with increasing sequence length:

| Phase | Seq len | Iters | LR   | Purpose |
|-------|---------|-------|------|---------|
| 1     | 512     | 500   | 8e-6 | Foundations — learn domain patterns |
| 2     | 1280    | 1000  | 5e-6 | Medium — extend reasoning |
| 3     | 4096    | 500   | 3e-6 | Long context — full capability |

Config files in `~/KIKI-Mac_tunner/configs/`:
- `mlx-lm-micro-kiki-phase1.yaml`
- `mlx-lm-micro-kiki-phase2.yaml` (resume from phase 1)
- `mlx-lm-micro-kiki-phase3.yaml` (resume from phase 2)

### Run

```bash
cd ~/KIKI-Mac_tunner
source .venv/bin/activate

# Phase 1: foundations
python -m mlx_lm lora --config configs/mlx-lm-micro-kiki-phase1.yaml

# Phase 2: medium (resumes from phase 1 checkpoint)
python -m mlx_lm lora --config configs/mlx-lm-micro-kiki-phase2.yaml

# Phase 3: long context (resumes from phase 2)
python -m mlx_lm lora --config configs/mlx-lm-micro-kiki-phase3.yaml
```

### Switch domain

Edit `data:` and `adapter_path:` in all 3 phase configs:
```yaml
data: data/micro-kiki/<domain>
adapter_path: output/micro-kiki/stack-NN-<domain>
```

## Key Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| rank | 64 | Matches 122B Opus quality config |
| scale | 64.0 | = rank (standard scaling) |
| batch_size | 1 | Coexists with parallel processes (Brainstacks) |
| grad_accumulation | 16 | Effective batch = 16 |
| grad_checkpoint | true | Required — model + activations exceed raw memory |
| clear_cache_threshold | 4 | Aggressive Metal cache cleanup |
| dropout | 0.01 | Light regularization |

## Constraints

- **Batch size 4 causes GPU Hang** when other MLX processes run in parallel
- **Batch size 2** works solo; **batch 1** safe with parallel loads
- **LR > 2e-4 diverges** on 35B MoE (loss explodes to 36+)
- **LR 8e-6** is the sweet spot (validated on 122B Opus training)
- **Do NOT LoRA-tune MoE FFN layers** — only attention projections
- **Do NOT use QLoRA/BitsAndBytes** — known issues with MoE sparse layers

## Observed Metrics

| Metric | Phase 1 (seq=512) | Phase 2 (seq=1280) | Phase 3 (seq=4096) |
|--------|-------------------|--------------------|--------------------|
| Peak mem | ~100 GB | ~130 GB | ~195 GB |
| Throughput | ~20-30 tok/s | ~15-25 tok/s | ~10-20 tok/s |
| Val loss start | ~1.8 | (resume) | (resume) |
| Val loss converge | ~0.5-0.7 | ~0.4-0.6 | ~0.3-0.5 |
| Time | ~2h | ~6h | ~4h |

## Forgetting Check (Mandatory)

After ALL 3 phases complete for a domain:

```bash
cd ~/micro-kiki
uv run python -m src.eval.forgetting --new-stack <domain> --prior-stacks <list>
```

Rollback if: angle < 30° AND win-rate drop > 0.03.

## Curriculum Order

```
Phase I:   chat-fr, reasoning
Phase II:  python, typescript, cpp, rust
Phase III: html-css, shell, sql, yaml-json, docker, kicad-dsl, spice, lua-upy
Phase IV:  embedded, stm32, iot, freecad, platformio, power, emc, dsp,
           spice-sim, electronics, kicad-pcb
Phase V:   web-frontend, web-backend, music-audio, devops, llm-orch
Phase VI:  math, security
```

## Output

```
~/KIKI-Mac_tunner/output/micro-kiki/stack-01-chat-fr/
├── adapters.safetensors           # latest checkpoint
├── 0000100_adapters.safetensors   # phase 1 checkpoints
├── 0000200_adapters.safetensors
├── ...
└── adapter_config.json
```

## Distillation (sparse domains)

6 domains need augmentation (< 1000 examples): freecad, platformio, spice-sim, stm32, kicad-pcb, music-audio.

```bash
cd ~/micro-kiki
uv run scripts/distill_with_local_teacher.py --domain <domain>
```

Uses Qwen3-Coder-480B local as teacher. No network dependency.

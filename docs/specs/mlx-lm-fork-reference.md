# mlx-lm fork reference

A custom fork of mlx-lm lives at `studio:/Users/clems/KIKI-Mac_tunner/lib/mlx_lm_fork/`.

## Why a fork

The user's training experiments needed modifications to mlx-lm internals
that aren't yet upstream:
- Custom LoRA hot-swap for MoE adapters
- Modified perplexity computation for spike outputs
- Quant tuning for Q4_K_M GGUF export with non-standard arch

## Contents (high level)

- `lora.py`, `fuse.py`, `convert.py` — modified versions
- `models/` — custom model defs
- `quant/` — Q4 quantization with arch-specific patches
- `evaluate.py`, `perplexity.py` — eval utilities

## Access

Read-only via SSH:

```bash
ssh studio "ls /Users/clems/KIKI-Mac_tunner/lib/mlx_lm_fork/"
```

## Future

If the experimental modifications stabilize, upstream to mlx-lm or vendor
the fork into this repo (currently ~50 files, deferred for size reasons).

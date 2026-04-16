# Source Code

Python 3.11+, uv as package manager. All code under `src/`.

## Architecture pivot (2026-04-16)

Base model switched from Qwen3.5-4B + custom MoE-LoRA to **Qwen3.5-35B-A3B** (native MoE, 256 experts, 3B active) + standard LoRA. Training runs via MLX on Mac Studio M3 Ultra 512 GB using the KIKI-Mac_tunner pipeline. The custom `moe_lora.py` adapter is no longer used — stacks use PEFT `LoraConfig` directly. See `docs/specs/2026-04-16-architecture-pivot-35b.md`.

## Style

- Type hints on all public functions
- Docstrings: one-line for simple, Google style for complex
- `from __future__ import annotations` in every module

## Imports

```python
# 1. stdlib
# 2. third-party (torch, transformers, peft)
# 3. project-local (src.*)
```

## Patterns

- Immutable configs: dataclasses with `frozen=True` or Pydantic `BaseModel`
- Early returns over deep nesting
- Context managers for GPU/VRAM-sensitive resources
- Explicit device placement (`device_map`, `.to(device)`)

## Tensor Conventions

- BF16 for training, Q4_K_M for inference
- Always name tensor dimensions in comments when shape is non-obvious
- Never silently reshape — assert shapes before operations

## Anti-Patterns

- No global model state — pass model/tokenizer explicitly
- No bare `except:` — catch specific exceptions
- No `print()` for logging — use `logging` module
- No hardcoded paths — use configs or env vars

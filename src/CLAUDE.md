# Source Code

Python 3.11+, uv as package manager. All code under `src/`.

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

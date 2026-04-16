# Studio Neuroscience Environment Setup

Story 13 of the v0.3 neuroscience plan. Documents the Studio M3 Ultra
(Apple Silicon, 512 GB unified memory) environment required to
download, load, and run SpikingBrain-7B in BF16 and W8ASpike variants,
and to run the Phase N-III smoke + quantization scripts (stories 14-15).

This also covers the local minimum environment on GrosMac used for
syntactic checks before pushing work to Studio over SSH.

## 1. Target host

- **Primary**: Mac Studio, M3 Ultra, 512 GB unified memory,
  macOS >= 14 (Sonoma) or 15 (Sequoia).
- **SSH user**: `clems@studio` (key-based, per session memory).
- **Project root on Studio**: `/Users/clems/KIKI-Mac_tunner/`.
- **Venv path**: `/Users/clems/KIKI-Mac_tunner/.venv-neuroscience/`.

## 2. Python dependencies

Installed via the new `neuro` optional extra on `pyproject.toml`:

```
torch>=2.5            # MPS support on M3 Ultra
transformers>=4.45
accelerate>=0.34
safetensors
sentencepiece
tiktoken              # Qwen2.5 tokenizer family
modelscope>=1.20      # ModelScope CLI + Python SDK
spikingjelly>=0.0.0.0.14  # Native SNN runtime (optional path)
```

Rationale per dep:

- `torch>=2.5`: required for MPS fp16/bf16 matmul kernels used by
  HF transformers 4.45+. Older torch crashes on M3 Ultra on large
  BF16 tensors.
- `transformers>=4.45`: first release with GatedDeltaNet + hybrid
  linear-attention support that SpikingBrain-7B's Qwen2.5 base
  relies on via custom `modeling_*.py` shims.
- `accelerate>=0.34`: device_map auto-dispatch + offload. 7B fits in
  RAM, but smoke scripts use `device_map="auto"` for robustness.
- `safetensors`: ModelScope ships `.safetensors` shards.
- `sentencepiece` + `tiktoken`: Qwen2.5 tokenizer is a BPE+TikToken
  hybrid; both libs are required.
- `modelscope`: primary distribution channel for SpikingBrain
  checkpoints (HF unavailable, see
  `docs/specs/spikingbrain-acquisition.md`).
- `spikingjelly`: optional second runtime for native SNN execution.
  If the BICLab ModelScope weights load directly via transformers
  (they do, per story 12 probe), spikingjelly is only required for
  story 30+ (Spikingformer adapter). Kept in the extra so story 13
  yields a single deterministic install.

## 3. Disk + RAM headroom

### Disk (Studio `/Users/clems/`)

| Item                                     | Size      |
|------------------------------------------|-----------|
| `models/spikingbrain-7b/` (BF16)         | ~15 GB    |
| `models/spikingbrain-7b-w8aspike/` (opt) | ~8 GB     |
| `models/spikingbrain-7b-q4.gguf` (story 15) | ~3.5 GB |
| ModelScope cache `.cache/modelscope/`    | ~5 GB     |
| Inference scratch (activations, KV)      | ~2 GB     |
| Venv                                     | ~4 GB     |
| **Minimum free before setup**            | **20 GB** |

Studio has ~1 TB free per session memory; comfortable margin.

### RAM (Studio 512 GB unified)

| Scenario                        | Peak RAM  |
|---------------------------------|-----------|
| BF16 model load                 | ~16 GB    |
| BF16 inference @ 4 k ctx        | ~20 GB    |
| BF16 inference @ 128 k ctx      | ~28 GB    |
| W8ASpike load                   | ~9 GB     |
| W8ASpike inference @ 4 k ctx    | ~12 GB    |
| Q4_K_M GGUF load (llama.cpp)    | ~4 GB     |

Story 14 acceptance target is BF16 load <= 16 GB / peak <= 20 GB.
All cases leave >= 480 GB free on Studio — the `classify_domains.py`
curriculum job can safely co-run (it is CPU-bound and low-memory).

## 4. Setup script (`scripts/setup_neuro_env.sh`)

Idempotent, supports `--dry-run` and `--studio` flags.

```bash
# From repo root (on Studio via ssh, or locally for dry-run only)
bash scripts/setup_neuro_env.sh --dry-run             # prints plan
bash scripts/setup_neuro_env.sh                       # actual install
bash scripts/setup_neuro_env.sh --studio              # remote to Studio
```

Behaviour:

1. Detects whether `/Users/clems/KIKI-Mac_tunner/.venv-neuroscience`
   already exists. If yes, re-uses it; if no, creates it.
2. Installs / refreshes the `neuro` extra via `uv pip install -e .[neuro]`.
3. Verifies: `python -c "import spikingjelly, torch; \
   print(torch.backends.mps.is_available())"` returns `True`.
4. Checks `modelscope --version` is reachable and that a ModelScope
   login token is present at `~/.cache/modelscope/token` (if not,
   prints the public-download fallback command and the "no-login
   required for Apache-2.0 Abel2076 repo" note from story 12).
5. Prints disk + RAM estimate vs current free; warns below 20 GB
   free.
6. Exits 0 on success, non-zero if any check fails.

Dry-run mode (`--dry-run`) prints every command that would run but
performs no install and no remote action. Used to validate the
script in CI-like conditions on GrosMac without touching Studio.

## 5. Verification acceptance

Per `.claude/plans/micro-kiki-v0.3-neuroscience.md` story 13:

- Fresh clone + `uv sync --extra neuro` succeeds on Studio (no
  wheels missing, no rustc build error).
- `python -c "import spikingjelly; import torch; \
  print(torch.backends.mps.is_available())"` prints `True`.
- `bash scripts/setup_neuro_env.sh --dry-run` prints a non-empty plan
  and exits 0 with no side effects.
- `bash scripts/setup_neuro_env.sh` on Studio completes end-to-end in
  under 15 minutes on residential bandwidth (most time is torch
  wheel download ~ 800 MB).

## 6. MPS vs CPU tradeoff (M3 Ultra)

SpikingBrain-7B uses a custom `modeling_qwen2_spiking.py` that ships
with the BICLab ModelScope repo. It registers several ops that are
**not in the PyTorch MPS kernel set** as of torch 2.5:

- `torch.nn.functional.silu` on bf16 MPS: OK since torch 2.4.
- Custom spike-generation kernel: falls back to CPU tensor ops.
- GatedDeltaNet linear attention: OK on MPS with a ~15% penalty vs
  CUDA (Studio does not have CUDA; baseline is CPU).

Practical outcome: the smoke script (story 14) will run with
`device_map="auto"`, letting HF transformers map most layers to MPS
and spill the spike kernel to CPU transparently. Tokens/second on
Studio at 4 k ctx, BF16: ~18-25 tok/s (target: >= 10 tok/s). CPU-only
fallback (if MPS misbehaves): ~8-12 tok/s, still within spec.

## 7. Failure modes + remedies

| Symptom                                      | Likely cause                | Fix                                    |
|----------------------------------------------|------------------------------|----------------------------------------|
| `modelscope: command not found`              | Extra not installed          | `uv pip install -e .[neuro]`           |
| `torch.backends.mps.is_available() == False` | Intel Mac or old macOS       | Move to Studio M-series / update macOS |
| `OSError: ... .safetensors shard missing`    | Interrupted ModelScope DL    | Re-run `modelscope download` (resumes) |
| `ImportError: No module named 'spikingjelly'`| Pinning conflict with torch  | `uv pip install spikingjelly --no-deps`|
| Install hangs on `sentencepiece`             | Missing Xcode CLT            | `xcode-select --install`               |
| `Permission denied: ~/.cache/modelscope`     | Shared host, stale perms     | `chown -R $USER ~/.cache/modelscope`   |
| Studio disk > 95% full                       | Model cache accumulated      | `rm -rf ~/.cache/modelscope/hub/*.tmp` |

## 8. Out of scope for story 13

- Actual ModelScope download of the SpikingBrain weights: story 14.
- Quantization toolchain (llama.cpp `convert_hf_to_gguf.py`): story 15.
- Architecture deep-dive (layer types, spike encoding): story 16.
- LAS converter dependencies (`einops`, custom kernels): story 17.

## 9. References

- Plan: `.claude/plans/micro-kiki-v0.3-neuroscience.md` §§ 13-16.
- Acquisition spec: `docs/specs/spikingbrain-acquisition.md`.
- Session memory: `project_kiki_forge_mesh_2026_04_14.md` (Studio
  host state, tunnel map).

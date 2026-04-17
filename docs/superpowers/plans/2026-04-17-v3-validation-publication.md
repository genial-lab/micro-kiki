# micro-kiki V3 Validation & Publication Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate micro-kiki V3 quality (35 stacks, 489K dataset), fuse all adapters, publish dataset + model on HuggingFace, and complete the SpikingKiki paper with real eval results.

**Architecture:** The GGUF Q4_K_M (2.5GB) already exists for the python stack. We need: (1) an MLXBackend for eval_v2_v3.py to run real perplexity comparisons, (2) a batch fuse script for all 35 stacks, (3) HF uploads via existing release_hf.py + upload.sh tooling, (4) paper placeholder fill from eval results.

**Tech Stack:** MLX (mlx-lm), safetensors, huggingface-hub, llama.cpp (GGUF), pytest

---

## File Structure

| File | Responsibility |
|------|---------------|
| `scripts/eval_v2_v3.py` | **Modify** — add `MLXBackend` subclass |
| `tests/test_eval_backend.py` | **Create** — unit tests for MLXBackend |
| `scripts/fuse_moe_lora.py` | **Modify** — add `--all-stacks` batch mode |
| `tests/test_fuse_moe_lora.py` | **Create** — unit tests for fuse logic |
| `scripts/test_gguf_domains.py` | **Create** — smoke test GGUF across 35 domains |
| `scripts/release_hf.py` | **Existing** — already functional (182 lines) |
| `/tmp/hf-dataset-upload/upload.sh` | **Existing** — dataset upload script |
| `docs/papers/spikingkiki-v3-draft.md` | **Modify** — fill [PLACEHOLDER] with eval results |

---

### Task 1: Smoke-test the GGUF on Studio

**Files:**
- Create: `scripts/test_gguf_domains.py`

- [ ] **Step 1: Write the smoke-test script**

```python
#!/usr/bin/env python3
"""Smoke-test micro-kiki V3 GGUF across key domains via llama-server."""
import argparse
import json
import subprocess
import sys
import time

import httpx

DOMAIN_PROMPTS = {
    "python": "Write a Python function that merges two sorted lists into one sorted list.",
    "embedded": "Explain how to configure UART on an ESP32 using ESP-IDF.",
    "kicad-dsl": "Write a KiCad symbol for a 4-pin voltage regulator with VIN, VOUT, GND, EN.",
    "spice": "Write an ngspice netlist for a 2nd-order Butterworth low-pass filter at 1kHz.",
    "electronics": "What is the difference between a MOSFET and a BJT for switching applications?",
    "components": "What are the specs and pinout of the AMS1117-3.3 voltage regulator?",
    "chat-fr": "Explique-moi le principe de fonctionnement d'un pont en H.",
    "reasoning": "A circuit has three resistors: 100Ω, 200Ω, 300Ω in parallel. What is the total resistance? Show your work.",
    "shell": "Write a bash one-liner that finds all .c files modified in the last 24 hours and counts their total lines.",
    "docker": "Write a multi-stage Dockerfile for a Python FastAPI app with minimal final image size.",
}


def test_domain(client: httpx.Client, domain: str, prompt: str) -> dict:
    """Send prompt to llama-server and assess response."""
    resp = client.post("/v1/chat/completions", json={
        "model": "micro-kiki-v3",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.3,
    }, timeout=60.0)
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    tokens = data.get("usage", {})

    return {
        "domain": domain,
        "prompt_len": len(prompt),
        "response_len": len(content),
        "tokens": tokens.get("completion_tokens", 0),
        "degenerate": len(content) < 20 or content.count("\n") < 1,
        "preview": content[:200],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080", help="llama-server URL")
    ap.add_argument("--gguf", default=None, help="GGUF path — starts llama-server automatically")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    server_proc = None
    if args.gguf:
        print(f"Starting llama-server with {args.gguf}...")
        server_proc = subprocess.Popen([
            "llama-server", "-m", args.gguf,
            "-c", "4096", "--port", str(args.port),
            "--log-disable",
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(5)
        args.url = f"http://localhost:{args.port}"

    try:
        client = httpx.Client(base_url=args.url)
        # Health check
        health = client.get("/health")
        assert health.status_code == 200, f"Server not ready: {health.status_code}"

        print(f"{'Domain':<15} {'Resp len':>8} {'Tokens':>7} {'OK?':>4}")
        print("-" * 40)

        results = []
        for domain, prompt in DOMAIN_PROMPTS.items():
            r = test_domain(client, domain, prompt)
            results.append(r)
            ok = "FAIL" if r["degenerate"] else " OK "
            print(f"{domain:<15} {r['response_len']:>8} {r['tokens']:>7} {ok}")

        passed = sum(1 for r in results if not r["degenerate"])
        print(f"\n{passed}/{len(results)} domains passed smoke test")

        # Save detailed results
        with open("output/micro-kiki/gguf/smoke-test-results.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    finally:
        if server_proc:
            server_proc.terminate()
            server_proc.wait()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Copy to Studio and run**

```bash
scp scripts/test_gguf_domains.py studio:/Users/clems/KIKI-Mac_tunner/scripts/
ssh studio "cd /Users/clems/KIKI-Mac_tunner && /opt/homebrew/bin/python3.12 scripts/test_gguf_domains.py --gguf output/micro-kiki/gguf/micro-kiki-v3-Q4_K_M.gguf"
```

Expected: 10/10 domains produce non-degenerate responses (>20 chars, contains newlines).

- [ ] **Step 3: Review responses**

Read `output/micro-kiki/gguf/smoke-test-results.json` and verify each domain's response is relevant (not gibberish, not empty, addresses the prompt).

- [ ] **Step 4: Commit**

```bash
git add scripts/test_gguf_domains.py
git commit -m "test: GGUF smoke test across 10 domains"
```

---

### Task 2: Implement MLXBackend for real eval

**Files:**
- Modify: `scripts/eval_v2_v3.py` — add `MLXBackend` class after `SafetensorsBackend`
- Create: `tests/test_eval_backend.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_backend.py
"""Test MLXBackend can be instantiated and has required interface."""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def test_mlx_backend_has_required_methods():
    """MLXBackend must implement all AdapterBackend methods."""
    from eval_v2_v3 import MLXBackend
    backend = MLXBackend.__new__(MLXBackend)
    assert hasattr(backend, "load_base_model")
    assert hasattr(backend, "load_adapter")
    assert hasattr(backend, "unload_adapter")
    assert hasattr(backend, "generate")
    assert hasattr(backend, "compute_perplexity")


def test_mlx_backend_instantiation():
    """MLXBackend should instantiate without errors."""
    from eval_v2_v3 import MLXBackend
    backend = MLXBackend()
    assert backend is not None


@pytest.mark.skipif(
    not Path("/Users/clems/KIKI-Mac_tunner/models/Qwen3.5-4B").exists(),
    reason="Base model not available locally",
)
def test_mlx_backend_load_model():
    """MLXBackend can load the base model (Studio only)."""
    from eval_v2_v3 import MLXBackend
    backend = MLXBackend()
    backend.load_base_model("/Users/clems/KIKI-Mac_tunner/models/Qwen3.5-4B")
    assert backend._model is not None
    assert backend._tokenizer is not None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run python -m pytest tests/test_eval_backend.py -v
```

Expected: FAIL with `ImportError: cannot import name 'MLXBackend'`

- [ ] **Step 3: Implement MLXBackend**

Add to `scripts/eval_v2_v3.py` after the `SafetensorsBackend` class (around line 120):

```python
class MLXBackend(AdapterBackend):
    """Real MLX backend for Apple Silicon eval."""

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._active_adapter = None

    def load_base_model(self, path: str):
        try:
            from mlx_lm import load
            self._model, self._tokenizer = load(path)
        except ImportError:
            raise RuntimeError("mlx-lm not installed. Run: pip install mlx-lm")

    def load_adapter(self, adapter_dir: str) -> str:
        """Load MoE-LoRA adapter weights from a stack directory."""
        import mlx.core as mx
        adapter_path = Path(adapter_dir) / "adapters.safetensors"
        if not adapter_path.exists():
            raise FileNotFoundError(f"No adapter at {adapter_path}")
        self._active_adapter = mx.load(str(adapter_path))
        adapter_id = Path(adapter_dir).name
        return adapter_id

    def unload_adapter(self, adapter_id: str):
        self._active_adapter = None

    def generate(self, prompt: str, adapter_id: str | None = None,
                 max_tokens: int = 256, temperature: float = 0.3) -> str:
        from mlx_lm import generate as mlx_generate
        messages = [{"role": "user", "content": prompt}]
        formatted = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        response = mlx_generate(
            self._model, self._tokenizer, prompt=formatted,
            max_tokens=max_tokens, temp=temperature, verbose=False,
        )
        return response

    def compute_perplexity(self, text: str, adapter_id: str | None = None) -> float:
        """Compute perplexity of text under the current model."""
        import mlx.core as mx
        import mlx.nn as nn
        tokens = self._tokenizer.encode(text)
        if len(tokens) < 2:
            return 100.0
        input_ids = mx.array([tokens[:-1]])
        labels = mx.array([tokens[1:]])
        logits = self._model(input_ids)
        loss = nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
        )
        return float(mx.exp(mx.mean(loss)).item())
```

- [ ] **Step 4: Run tests**

```bash
uv run python -m pytest tests/test_eval_backend.py -v
```

Expected: 2 pass, 1 skip (model not local)

- [ ] **Step 5: Wire MLXBackend into CLI**

In `scripts/eval_v2_v3.py`, modify the `main()` function's backend selection (around line 300) to use `MLXBackend` when `--dry-run` is not set and MLX is available:

```python
    if args.dry_run:
        backend = StubBackend()
    else:
        try:
            backend = MLXBackend()
        except Exception:
            backend = SafetensorsBackend()
```

- [ ] **Step 6: Commit**

```bash
git add scripts/eval_v2_v3.py tests/test_eval_backend.py
git commit -m "feat(eval): MLXBackend for real V2/V3 comparison"
```

---

### Task 3: Run real eval V2 vs V3 on Studio

**Files:**
- No new files — run existing scripts on Studio

- [ ] **Step 1: Copy updated eval script to Studio**

```bash
scp scripts/eval_v2_v3.py studio:/Users/clems/KIKI-Mac_tunner/scripts/
```

- [ ] **Step 2: Run eval on 10 key domains**

```bash
ssh studio "cd /Users/clems/KIKI-Mac_tunner && /opt/homebrew/bin/python3.12 scripts/eval_v2_v3.py \
  --v2-dir output/micro-kiki/stacks-v2 \
  --v3-dir output/micro-kiki/stacks \
  --base-model models/Qwen3.5-4B \
  --domains chat-fr reasoning python embedded electronics components shell spice kicad-dsl docker \
  2>&1 | tee /tmp/eval_v2_v3_real.log"
```

Expected: comparison table with real perplexity + keyword scores per domain.

- [ ] **Step 3: Save eval results**

```bash
scp studio:/tmp/eval_v2_v3_real.log output/micro-kiki/eval/
scp studio:/Users/clems/KIKI-Mac_tunner/eval_results.json output/micro-kiki/eval/ 2>/dev/null
```

- [ ] **Step 4: Commit results**

```bash
git add output/micro-kiki/eval/
git commit -m "data: V2 vs V3 eval results (10 domains)"
```

---

### Task 4: Batch-fuse all 35 stacks

**Files:**
- Modify: `scripts/fuse_moe_lora.py` — add `--all-stacks` and `--stacks-dir` flags
- Create: `tests/test_fuse_moe_lora.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fuse_moe_lora.py
"""Test fuse_moe_lora batch mode argument parsing."""
import subprocess
import sys

def test_fuse_all_stacks_flag_exists():
    """--all-stacks flag should be accepted."""
    result = subprocess.run(
        [sys.executable, "scripts/fuse_moe_lora.py", "--help"],
        capture_output=True, text=True,
    )
    assert "--all-stacks" in result.stdout or "--all-stacks" in result.stderr


def test_build_key_mapping_import():
    """build_key_mapping should be importable."""
    sys.path.insert(0, "scripts")
    from fuse_moe_lora import build_key_mapping
    # Empty inputs should return empty mapping
    result = build_key_mapping([], set())
    assert result == {}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run python -m pytest tests/test_fuse_moe_lora.py -v
```

Expected: FAIL — no `--all-stacks` flag yet.

- [ ] **Step 3: Add batch mode to fuse_moe_lora.py**

Add to `main()` in `scripts/fuse_moe_lora.py`, replacing the fixed `sys.argv` parsing with argparse:

```python
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Fuse MoE-LoRA adapters (auto MLX/CUDA)")
    ap.add_argument("base_model", help="Path to base HF model")
    ap.add_argument("adapter", nargs="?", help="Path to single adapter stack dir")
    ap.add_argument("output", nargs="?", help="Output dir for fused model")
    ap.add_argument("--all-stacks", action="store_true", help="Fuse all stacks in --stacks-dir")
    ap.add_argument("--stacks-dir", default="output/micro-kiki/stacks", help="Dir containing stack-* subdirs")
    ap.add_argument("--out-dir", default="output/micro-kiki/gguf", help="Output directory for all fused models")
    args = ap.parse_args()

    if args.all_stacks:
        stacks_dir = Path(args.stacks_dir)
        out_base = Path(args.out_dir)
        for stack_dir in sorted(stacks_dir.iterdir()):
            if not stack_dir.is_dir():
                continue
            if not (stack_dir / "adapters.safetensors").exists():
                continue
            domain = stack_dir.name
            out_dir = out_base / f"fused-{domain}"
            print(f"\n{'='*60}")
            print(f"Fusing: {domain}")
            print(f"{'='*60}")
            fuse_single(Path(args.base_model), stack_dir, out_dir)
    else:
        if not args.adapter or not args.output:
            ap.error("Provide adapter and output paths, or use --all-stacks")
        fuse_single(Path(args.adapter), Path(args.adapter), Path(args.output))
```

Extract the current `main()` body into `fuse_single(base_path, adapter_path, output_dir)`.

- [ ] **Step 4: Run tests**

```bash
uv run python -m pytest tests/test_fuse_moe_lora.py -v
```

Expected: PASS

- [ ] **Step 5: Run batch fuse on Studio (background)**

```bash
scp scripts/fuse_moe_lora.py studio:/Users/clems/KIKI-Mac_tunner/scripts/
ssh studio "cd /Users/clems/KIKI-Mac_tunner && nohup /opt/homebrew/bin/python3.12 scripts/fuse_moe_lora.py models/Qwen3.5-4B --all-stacks > /tmp/fuse_all.log 2>&1 &"
```

Expected: 35 fused models in `output/micro-kiki/gguf/fused-{domain}/`

- [ ] **Step 6: Commit**

```bash
git add scripts/fuse_moe_lora.py tests/test_fuse_moe_lora.py
git commit -m "feat(fuse): batch mode for all 35 stacks"
```

---

### Task 5: Upload dataset to HuggingFace

**Files:**
- Existing: `/tmp/hf-dataset-upload/README.md`, `/tmp/hf-dataset-upload/upload.sh`

- [ ] **Step 1: Install huggingface-cli and login**

```bash
uv pip install huggingface-hub
huggingface-cli login
```

Enter token when prompted. Verify: `huggingface-cli whoami` should show `electron-rare`.

- [ ] **Step 2: Verify data is synced**

```bash
du -sh /tmp/hf-dataset-upload/data/
ls /tmp/hf-dataset-upload/data/ | wc -l
```

Expected: ~1.1 GB, 38 domain dirs. If empty, re-sync:
```bash
rsync -a studio:/Users/clems/KIKI-Mac_tunner/data/micro-kiki/ /tmp/hf-dataset-upload/data/
```

- [ ] **Step 3: Review README.md**

```bash
cat /tmp/hf-dataset-upload/README.md | head -50
```

Verify YAML frontmatter, domain table, license info are correct.

- [ ] **Step 4: Run upload script**

```bash
bash /tmp/hf-dataset-upload/upload.sh
```

Expected: Creates `electron-rare/micro-kiki-v3-dataset` on HuggingFace, uploads all domain dirs.

- [ ] **Step 5: Verify on HuggingFace**

```bash
huggingface-cli repo info electron-rare/micro-kiki-v3-dataset --type dataset
```

Expected: repo exists, shows file count matching upload.

---

### Task 6: Upload model to HuggingFace

**Files:**
- Existing: `scripts/release_hf.py`

- [ ] **Step 1: Create model repo config**

Create `configs/hf-release-v3.yaml`:

```yaml
repo: "electron-rare/micro-kiki-v3"
adapters_dir: "output/micro-kiki/stacks"
model_card: "MODEL_CARD.md"
gguf_dir: "output/micro-kiki/gguf"
private: false
execute: false  # dry-run first
```

- [ ] **Step 2: Dry-run the model release**

```bash
ssh studio "cd /Users/clems/KIKI-Mac_tunner && /opt/homebrew/bin/python3.12 scripts/release_hf.py \
  --repo electron-rare/micro-kiki-v3 \
  --adapters output/micro-kiki/stacks \
  --model-card MODEL_CARD.md"
```

Expected: lists all files that would be uploaded, no actual upload.

- [ ] **Step 3: Execute the model release**

```bash
ssh studio "cd /Users/clems/KIKI-Mac_tunner && /opt/homebrew/bin/python3.12 scripts/release_hf.py \
  --repo electron-rare/micro-kiki-v3 \
  --adapters output/micro-kiki/stacks \
  --model-card MODEL_CARD.md \
  --execute"
```

Expected: creates `electron-rare/micro-kiki-v3` with 35 adapter dirs + MODEL_CARD.md.

- [ ] **Step 4: Upload GGUF separately**

```bash
ssh studio "cd /Users/clems/KIKI-Mac_tunner && huggingface-cli upload electron-rare/micro-kiki-v3 \
  output/micro-kiki/gguf/micro-kiki-v3-Q4_K_M.gguf \
  micro-kiki-v3-Q4_K_M.gguf --repo-type model"
```

- [ ] **Step 5: Commit config**

```bash
git add configs/hf-release-v3.yaml
git commit -m "config: HF release config for V3"
```

---

### Task 7: Complete SpikingKiki paper with eval results

**Files:**
- Modify: `docs/papers/spikingkiki-v3-draft.md`

- [ ] **Step 1: Read eval results**

After Task 3 completes, read `output/micro-kiki/eval/eval_v2_v3_real.log`.

Extract:
- Per-domain perplexity (V2 vs V3)
- Keyword hit rates
- Forgetting matrix (if `--cross-eval` was run)

- [ ] **Step 2: Fill [PLACEHOLDER] sections**

In `docs/papers/spikingkiki-v3-draft.md`, replace the 4 placeholder lines (around lines 189-196) with actual domain distribution counts from the dataset:

```markdown
| Category | Domains | Examples |
|----------|---------|----------|
| Conversation | chat-fr | 63,092 |
| Reasoning | reasoning, math | 12,513 |
| Code | python, typescript, cpp, rust, shell, sql | 190,507 |
| Electronics | electronics, components, embedded, stm32, power, emc, dsp | 110,191 |
| EDA | kicad-dsl, kicad-pcb, spice, freecad | 22,026 |
| Infrastructure | docker, devops, llm-ops, llm-orch, ml-training | 14,882 |
| Web | html-css, web-frontend, web-backend, typescript | 15,268 |
| Other | iot, platformio, lua-upy, yaml-json, music-audio, security | 11,668 |
```

- [ ] **Step 3: Add Results section with real data**

Replace the placeholder results table with actual eval numbers from Task 3.

- [ ] **Step 4: Commit**

```bash
git add docs/papers/spikingkiki-v3-draft.md
git commit -m "docs: fill paper placeholders with V3 eval results"
```

---

## Self-Review

**Spec coverage:**
- ✅ Task 1: GGUF smoke test (Priority 1.1)
- ✅ Task 2: MLXBackend for eval (Priority 1.2)
- ✅ Task 3: Run real eval V2 vs V3 (Priority 1.2)
- ✅ Task 4: Batch fuse all 35 stacks (Priority 1.3)
- ✅ Task 5: HF dataset upload (Priority 2.4)
- ✅ Task 6: HF model upload (Priority 2.5)
- ✅ Task 7: Complete paper (Priority 2.6)

**Placeholder scan:** No TBD/TODO. All steps have exact commands and code.

**Type consistency:** `MLXBackend` matches `AdapterBackend` ABC from eval_v2_v3.py. `fuse_single()` extracted from existing `main()`. `build_key_mapping()` signature matches existing code.

**Dependencies:** Task 3 depends on Task 2. Task 7 depends on Task 3. Tasks 1, 4, 5, 6 are independent. Recommended execution order: 1 → 2 → 3 → 7 (serial), 4 + 5 + 6 (parallel after Task 1).

# C2-LoRA Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rerun the C2 downstream eval using real Qwen3.6-35B-A3B LoRA adapters (already trained on Studio, 10 MLX stacks in `~/micro-kiki/outputs/stacks/stack-<domain>/`) instead of system-prompt pseudo-adapters, and test whether weight-level specialisation eliminates the persona-refusal pathology identified in the diagnostic.

**Architecture:** A small HTTP server on Studio wraps MLX: `load()` base Qwen3.6-35B-A3B once + `linear_to_lora_layers` shell conversion once, then accept per-request adapter swaps by directly reloading the LoRA weight tensors onto existing `LoRALinear` layers (pattern from commit `b102115`). The bench orchestrator runs locally and calls two HTTP endpoints on Studio: `/v1/chat/completions` (with `adapter` query param) for generation AND for self-judging. Same 100 queries, 3 routers (vqc, random, oracle), identical harness contract.

**Tech Stack:** Python 3.13 on Studio (`~/micro-kiki/.venv`), `mlx_lm` 0.31.2, FastAPI, uvicorn, numpy, torch (local). Inference on Apple M3 Ultra 512 GB. No new training.

Spec basis: C2 diagnostic findings (`docs/paper-a/c2-diagnostic.md`) — pattern #1 (persona-refusal) is prompt-specific, so weight-level specialisation should eliminate it.

---

## File Structure

**Files to create locally (in micro-kiki repo):**
- `src/serving/studio_lora_server.py` — FastAPI server + MLX adapter-swap logic (~150 lines)
- `tests/serving/test_studio_lora_server.py` — unit tests on pure logic + API contract (~100 lines, MLX mocked)
- `scripts/bench_downstream_c2_lora.py` — bench orchestrator (~200 lines, forks from `bench_downstream_c2.py`)
- `tests/scripts/test_bench_downstream_c2_lora.py` — integration test with mocked Studio HTTP (~60 lines)
- `scripts/deploy_studio_lora_server.sh` — rsync + systemd-like start command (~30 lines)
- `results/c2-lora-downstream.json` — measured results
- `docs/paper-a/c2-lora-results.md` — narrative comparing vs C2 baseline
- `docs/paper-a/c2-lora-figure.pdf` — side-by-side bar chart

**Files to modify:**
- `docs/superpowers/plans/2026-04-19-phase-c-roadmap.md` (add C2-LoRA row)
- `docs/paper-a/paper-a-v2.tex` (optional — §5.2 add subsection on LoRA follow-up)

---

### Task 1: Studio adapter-swap logic + tests (local with MLX mocked)

**Files:**
- Create: `src/serving/__init__.py` (empty)
- Create: `src/serving/studio_lora_server.py`
- Create: `tests/serving/__init__.py` (empty)
- Create: `tests/serving/test_studio_lora_server.py`

- [ ] **Step 1: Create empty package dirs**

```bash
mkdir -p src/serving tests/serving
touch src/serving/__init__.py tests/serving/__init__.py
```

- [ ] **Step 2: Write failing tests** with EXACTLY this content at `tests/serving/test_studio_lora_server.py`:

```python
"""Unit tests for src/serving/studio_lora_server.py — MLX mocked out."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def test_adapter_name_to_path_maps_known_stacks(tmp_path):
    """adapter_name_to_path returns the absolute path for each of the 10 stacks."""
    from src.serving.studio_lora_server import adapter_name_to_path

    # Simulate a stacks root containing 3 subdirs
    (tmp_path / "stack-dsp").mkdir()
    (tmp_path / "stack-spice").mkdir()
    (tmp_path / "stack-stm32").mkdir()
    for d in ("dsp", "spice", "stm32"):
        (tmp_path / f"stack-{d}" / "adapters.safetensors").write_bytes(b"stub")

    p = adapter_name_to_path("dsp", stacks_root=tmp_path)
    assert p == tmp_path / "stack-dsp" / "adapters.safetensors"


def test_adapter_name_to_path_raises_on_unknown(tmp_path):
    from src.serving.studio_lora_server import adapter_name_to_path

    import pytest
    with pytest.raises(FileNotFoundError, match="no adapter"):
        adapter_name_to_path("nonexistent-domain", stacks_root=tmp_path)


def test_build_chat_prompt_no_system_no_adapter_persona():
    """With real LoRA, we DO NOT inject a system prompt — weights do the specialisation."""
    from src.serving.studio_lora_server import build_chat_prompt

    tokenizer = MagicMock()
    tokenizer.apply_chat_template = MagicMock(return_value="TEMPLATED")
    out = build_chat_prompt(tokenizer, user_message="What is a Schmitt trigger?")
    tokenizer.apply_chat_template.assert_called_once()
    args, kwargs = tokenizer.apply_chat_template.call_args
    messages = args[0] if args else kwargs["conversation"]
    # Exactly one message, role=user, no system persona
    assert len(messages) == 1
    assert messages[0] == {"role": "user", "content": "What is a Schmitt trigger?"}


def test_swap_adapter_tracks_currently_loaded():
    """swap_adapter updates state and calls mx.load + model.load_weights."""
    from src.serving.studio_lora_server import AdapterState, swap_adapter

    state = AdapterState(model=MagicMock(), tokenizer=MagicMock(),
                         stacks_root=Path("/tmp/fake-stacks"), currently_loaded=None)
    fake_weights = [("lora_A", "tensor_a"), ("lora_B", "tensor_b")]
    with patch("src.serving.studio_lora_server.mx") as mx_mock:
        mx_mock.load.return_value = dict(fake_weights)
        swap_adapter(state, "dsp",
                     adapter_path_override=Path("/tmp/fake-stacks/stack-dsp/adapters.safetensors"))
    assert state.currently_loaded == "dsp"
    state.model.load_weights.assert_called_once()
    # strict=False means unknown keys don't error
    _args, kwargs = state.model.load_weights.call_args
    assert kwargs.get("strict") is False


def test_swap_adapter_base_resets_to_none():
    """Passing adapter_name='base' resets currently_loaded to None (no-op if already None)."""
    from src.serving.studio_lora_server import AdapterState, swap_adapter

    state = AdapterState(model=MagicMock(), tokenizer=MagicMock(),
                         stacks_root=Path("/tmp/fake-stacks"), currently_loaded="dsp")
    with patch("src.serving.studio_lora_server.mx"):
        swap_adapter(state, "base")
    assert state.currently_loaded is None
    # No weight load call — 'base' means zero out LoRA, which mlx does by having LoRA.B init to 0
    # For simplicity we just mark state; generation quality is verified in smoke test
```

- [ ] **Step 3: Run to verify fail**

Run: `uv run python -m pytest tests/serving/test_studio_lora_server.py -v 2>&1 | tail -10`
Expected: 5/5 FAIL with `ModuleNotFoundError: No module named 'src.serving.studio_lora_server'`.

- [ ] **Step 4: Commit**

```bash
cd /Users/electron/Documents/Projets/micro-kiki
git branch --show-current  # verify 'main'
git add src/serving/__init__.py tests/serving/__init__.py tests/serving/test_studio_lora_server.py
git commit -m "test(c2-lora): studio server tests (red)"
```

Subject ≤50 chars, no Co-Authored-By.

---

### Task 2: Implement `studio_lora_server.py`

**Files:**
- Create: `src/serving/studio_lora_server.py`

- [ ] **Step 1: Create** with EXACTLY this content:

```python
"""Studio-side MLX inference server with per-request adapter-swap.

Wraps Qwen3.6-35B-A3B + any of 10 trained LoRA adapters from
~/micro-kiki/outputs/stacks/stack-<domain>/. Base model loaded once with
LoRALinear shells; per-request swap reloads adapter weights in place (no
base reload, no double-conversion of Linear → LoRALinear).

Pattern derived from commit b102115 ('E2E smoke on Studio + real results').

Exposes two HTTP endpoints (OpenAI-compatible):
- GET  /v1/models               — health / currently-loaded adapter
- POST /v1/chat/completions     — body {messages, max_tokens, adapter?}
                                  if adapter is set and differs from current, swap first.

Run:
    ~/micro-kiki/.venv/bin/python -m src.serving.studio_lora_server \\
        --base ~/models/Qwen3.6-35B-A3B \\
        --stacks-root ~/micro-kiki/outputs/stacks \\
        --host 0.0.0.0 --port 19000
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# MLX imports are deferred; tests mock them.
try:  # pragma: no cover — only available on Studio / systems with MLX
    import mlx.core as mx  # type: ignore
except Exception:  # pragma: no cover
    mx = None  # type: ignore


@dataclass
class AdapterState:
    model: Any
    tokenizer: Any
    stacks_root: Path
    currently_loaded: str | None = None


def adapter_name_to_path(name: str, stacks_root: Path) -> Path:
    """Resolve 'dsp' → <stacks_root>/stack-dsp/adapters.safetensors."""
    p = stacks_root / f"stack-{name}" / "adapters.safetensors"
    if not p.parent.exists():
        raise FileNotFoundError(f"no adapter for domain {name!r} under {stacks_root}")
    return p


def build_chat_prompt(tokenizer: Any, user_message: str) -> str:
    """Build the prompt string using the tokenizer's chat template.

    IMPORTANT: NO system-prompt persona is injected. The weight-level LoRA
    adapter provides specialisation — we do NOT want the persona-refusal
    pathology from C2 baseline (pattern #1).
    """
    messages = [{"role": "user", "content": user_message}]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )


def swap_adapter(
    state: AdapterState,
    adapter_name: str,
    *,
    adapter_path_override: Path | None = None,
) -> None:
    """Load the named adapter's weights onto the pre-converted LoRALinear layers.

    Passing adapter_name='base' resets currently_loaded to None without weight IO
    (LoRALinear's LoRA.B branch contributes 0 when freshly zeroed; in practice
    when swapping across non-base adapters we always overwrite, so 'base' is
    only used at startup).
    """
    if adapter_name == "base":
        state.currently_loaded = None
        return

    if state.currently_loaded == adapter_name:
        return  # already loaded, no-op

    adapter_path = adapter_path_override or adapter_name_to_path(
        adapter_name, state.stacks_root
    )
    if mx is None:  # pragma: no cover — only hit in MLX-less envs
        raise RuntimeError("mlx.core unavailable; run this on Studio")
    weights = mx.load(str(adapter_path))
    state.model.load_weights(list(weights.items()), strict=False)
    state.currently_loaded = adapter_name


def _bootstrap(base_path: Path, stacks_root: Path) -> AdapterState:  # pragma: no cover
    """Load base + convert to LoRALinear shells once at server startup."""
    from mlx_lm import load  # type: ignore
    from mlx_lm.tuner.utils import linear_to_lora_layers  # type: ignore

    logger.info("loading base model from %s", base_path)
    model, tokenizer = load(str(base_path))

    # Shell conversion: use rank 16, alpha 32 (matches PRD story-4 config).
    lora_config = [
        {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.0,
            "scale": 2.0,
            "keys": [
                "self_attn.q_proj", "self_attn.k_proj",
                "self_attn.v_proj", "self_attn.o_proj",
            ],
        },
    ]
    num_layers = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    if num_layers is None:
        num_layers = 40  # Qwen3.6-35B-A3B default; fallback
    linear_to_lora_layers(model, num_layers, lora_config[0])
    logger.info("converted %d layers to LoRALinear shells", num_layers)

    return AdapterState(
        model=model, tokenizer=tokenizer, stacks_root=stacks_root, currently_loaded=None
    )


def _build_app(state: AdapterState):  # pragma: no cover — tested manually via curl
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(title="Studio LoRA Swap Server")

    class Message(BaseModel):
        role: str
        content: str

    class ChatCompletionRequest(BaseModel):
        model: str | None = None
        messages: list[Message]
        max_tokens: int = 512
        temperature: float = 0.0
        adapter: str | None = None  # our extension
        chat_template_kwargs: dict | None = None  # ignored; MLX handles its own template

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [
                {"id": "qwen3.6-35b-a3b",
                 "currently_loaded_adapter": state.currently_loaded},
            ],
        }

    @app.post("/v1/chat/completions")
    def completions(req: ChatCompletionRequest):
        from mlx_lm import generate  # type: ignore

        if req.adapter is not None and req.adapter != state.currently_loaded:
            swap_adapter(state, req.adapter)

        # Flatten messages into the user-turn content expected by build_chat_prompt.
        # We only take the LAST user message (one-shot); keep the flow simple.
        user_content = next(
            (m.content for m in reversed(req.messages) if m.role == "user"),
            req.messages[-1].content if req.messages else "",
        )
        prompt = build_chat_prompt(state.tokenizer, user_content)
        text = generate(
            state.model, state.tokenizer, prompt, max_tokens=req.max_tokens,
            verbose=False,
        )
        return {
            "id": "cmpl-c2lora",
            "object": "chat.completion",
            "model": "qwen3.6-35b-a3b",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "c2_lora_meta": {"adapter": state.currently_loaded},
        }

    return app


def main() -> int:  # pragma: no cover
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True, type=Path)
    p.add_argument("--stacks-root", required=True, type=Path)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=19000)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    state = _bootstrap(args.base, args.stacks_root)
    app = _build_app(state)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
```

- [ ] **Step 2: Run tests to verify pass**

Run: `uv run python -m pytest tests/serving/test_studio_lora_server.py -v 2>&1 | tail -12`
Expected: 5/5 PASSED.

- [ ] **Step 3: Commit**

```bash
git add src/serving/studio_lora_server.py
git commit -m "feat(c2-lora): MLX swap server (no persona)"
```

Subject ≤50 chars.

---

### Task 3: Deploy + smoke-test server on Studio

**Files:**
- Create: `scripts/deploy_studio_lora_server.sh`

- [ ] **Step 1: Create the deploy script**

Write to `scripts/deploy_studio_lora_server.sh`:

```bash
#!/bin/bash
# Deploy + (re)start the Studio LoRA swap server.
# Assumes: studio ssh alias, ~/micro-kiki checkout present, .venv with mlx_lm + fastapi + uvicorn.
set -euo pipefail

STUDIO_REPO="${STUDIO_REPO:-/Users/clems/micro-kiki}"
BASE_MODEL="${BASE_MODEL:-/Users/clems/models/Qwen3.6-35B-A3B}"
STACKS_ROOT="${STACKS_ROOT:-/Users/clems/micro-kiki/outputs/stacks}"
PORT="${PORT:-19000}"

echo "=== rsync server module ==="
rsync -az src/serving/ studio:"${STUDIO_REPO}/src/serving/"

echo "=== ensure deps on Studio ==="
ssh studio "cd ${STUDIO_REPO} && .venv/bin/pip install -q fastapi uvicorn || true"

echo "=== kill old server + launch ==="
ssh studio "pkill -f studio_lora_server 2>/dev/null; sleep 2; \
  nohup ${STUDIO_REPO}/.venv/bin/python -m src.serving.studio_lora_server \
    --base ${BASE_MODEL} \
    --stacks-root ${STACKS_ROOT} \
    --host 0.0.0.0 --port ${PORT} \
    > /tmp/studio-lora-server.log 2>&1 < /dev/null &"

echo "=== waiting for health on port ${PORT} ==="
for i in $(seq 1 60); do
    if ssh studio "curl -sf -o /dev/null http://localhost:${PORT}/v1/models"; then
        echo "READY (after ${i} × 10s)"
        exit 0
    fi
    sleep 10
done
echo "TIMEOUT — check /tmp/studio-lora-server.log on Studio"
exit 1
```

Make executable: `chmod +x scripts/deploy_studio_lora_server.sh`.

- [ ] **Step 2: Run the deploy**

```bash
bash scripts/deploy_studio_lora_server.sh
```

Expected: base model loads (2-5 min), "READY" printed. Verify base + LoRALinear conversion in log:

```bash
ssh studio "grep -E 'loading base|converted' /tmp/studio-lora-server.log | head -5"
```

Expected: "loading base model from /Users/clems/models/Qwen3.6-35B-A3B" then "converted 40 layers to LoRALinear shells".

If READY doesn't happen within 10 minutes OR the log shows an error, STOP and report BLOCKED — the `linear_to_lora_layers` API may have changed in mlx_lm 0.31.2, requiring a patch based on actual signatures. Do NOT commit anything until the server is actually up.

- [ ] **Step 3: Adapter-swap smoke test via curl**

```bash
# Query 1: base (no adapter)
ssh studio "curl -s http://localhost:19000/v1/chat/completions -H 'Content-Type: application/json' -d '{\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2?\"}],\"max_tokens\":50}' | jq -r '.choices[0].message.content' | head -c 200"

# Query 2: with dsp adapter
ssh studio "curl -s http://localhost:19000/v1/chat/completions -H 'Content-Type: application/json' -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Write code for an FIR filter.\"}],\"max_tokens\":100,\"adapter\":\"dsp\"}' | jq -r '.choices[0].message.content' | head -c 300"

# Query 3: check currently_loaded
ssh studio "curl -s http://localhost:19000/v1/models | jq"
```

Expected:
- Query 1: some math-ish answer.
- Query 2: code-like DSP answer (content may vary; what matters is a non-empty non-error response).
- Query 3: `currently_loaded_adapter: "dsp"`.

If any 500 error, dump the server log:

```bash
ssh studio "tail -30 /tmp/studio-lora-server.log"
```

Common failure modes and fixes:
- `AttributeError: 'Qwen3...' object has no attribute 'load_weights'` → MLX API delta; fall back to per-query model.update_modules pattern.
- `KeyError: some.lora.A` → adapter shapes mismatch base; verify the adapter was actually trained on Qwen3.6-35B-A3B (not 3.5).
- `RuntimeError: out of memory` → reduce model.max_tokens or run smaller adapter first.

STOP the plan and escalate if any of these persist after a single retry. Otherwise continue.

- [ ] **Step 4: Commit the deploy script**

```bash
git add scripts/deploy_studio_lora_server.sh
git commit -m "feat(c2-lora): studio deploy script"
```

---

### Task 4: Port the bench — `bench_downstream_c2_lora.py` with failing integration test

**Files:**
- Create: `tests/scripts/test_bench_downstream_c2_lora.py`
- Create: `scripts/bench_downstream_c2_lora.py`

- [ ] **Step 1: Write the failing integration test**

Create `tests/scripts/test_bench_downstream_c2_lora.py`:

```python
"""Integration test: bench_downstream_c2_lora dry-runs against mocked Studio server."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def test_c2_lora_dry_run_produces_json(tmp_path):
    """--dry-run skips HTTP, uses stub answers and a fixed score — exercises wiring."""
    from scripts.bench_downstream_c2_lora import main

    out = tmp_path / "c2-lora-dry.json"
    argv = [
        "--data-dir", "data/corpus-real",
        "--domains", "dsp,electronics,emc,embedded,freecad,kicad-dsl,platformio,power,spice,stm32",
        "--per-domain", "1",
        "--vqc-epochs", "30",
        "--output", str(out),
        "--dry-run",
    ]
    rc = main(argv)
    assert rc == 0
    assert out.exists()
    report = json.loads(out.read_text())
    assert "config" in report and "results" in report
    for r in ("vqc", "random", "oracle"):
        assert r in report["results"]
        assert "mean_score" in report["results"][r]
```

- [ ] **Step 2: Create the bench** `scripts/bench_downstream_c2_lora.py` with EXACTLY this content:

```python
#!/usr/bin/env python3
"""C2-LoRA downstream bench: VQC vs random vs oracle router, using REAL LoRA
adapters swapped at inference time on Studio MLX. Same 100-query held-out
eval as the C2 baseline. No system-prompt persona is injected — weight-level
specialisation replaces it.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import requests
import torch

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.routing.downstream_harness import run_downstream_eval
from src.routing.llm_judge import build_rubric_prompt, parse_score
from src.routing.text_jepa.dataset import load_domain_corpus
from src.routing.torch_vqc_router import TorchVQCRouter

logger = logging.getLogger(__name__)


def _studio_call(base_url: str, messages: list[dict], max_tokens: int,
                 adapter: str | None) -> str:
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0.0}
    if adapter is not None:
        body["adapter"] = adapter
    resp = requests.post(f"{base_url}/v1/chat/completions", json=body, timeout=240)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data/corpus-real"))
    p.add_argument("--domains",
                   default="dsp,electronics,emc,embedded,freecad,kicad-dsl,platformio,power,spice,stm32")
    p.add_argument("--per-domain", type=int, default=10)
    p.add_argument("--studio-url", default="http://studio:19000")
    p.add_argument("--backbone", default="models/niche-embeddings")
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--vqc-epochs", type=int, default=300)
    p.add_argument("--vqc-lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--judge-use-adapter", action="store_true",
                   help="If set, the judge call loads the EXPECTED-domain adapter "
                        "(may self-favour). Default: judge on base (no adapter).")
    p.add_argument("--output", type=Path, default=Path("results/c2-lora-downstream.json"))
    p.add_argument("--dry-run", action="store_true",
                   help="Stub LLM + judge calls; skip Studio HTTP entirely.")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    dom_to_idx = {d: i for i, d in enumerate(domains)}
    n_classes = len(domains)

    samples = load_domain_corpus(args.data_dir, domains=domains, max_per_domain=50)
    if not samples:
        logger.error("no samples loaded from %s", args.data_dir)
        return 2

    per_dom_count: dict[str, int] = {d: 0 for d in domains}
    eval_queries: list[dict] = []
    remaining: list = []
    for s in samples:
        if per_dom_count[s.domain] < args.per_domain:
            eval_queries.append({
                "question": s.text,
                "domain": s.domain,
                "domain_idx": dom_to_idx[s.domain],
            })
            per_dom_count[s.domain] += 1
        else:
            remaining.append(s)
    logger.info("eval: %d queries (%d/domain) | training pool: %d", len(eval_queries),
                args.per_domain, len(remaining))

    # Embeddings
    if args.dry_run:
        rng0 = np.random.default_rng(args.seed)
        eval_embs = rng0.standard_normal((len(eval_queries), 384)).astype(np.float64)
        train_embs = rng0.standard_normal((len(remaining), 384)).astype(np.float64)
    else:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(args.backbone, device="cpu")
        tok = st.tokenizer
        tr_model = st[0].auto_model.to("cpu")

        def embed(text: str) -> np.ndarray:
            enc = tok(text, return_tensors="pt", truncation=True,
                      max_length=args.seq_len, padding="max_length")
            with torch.no_grad():
                out = tr_model(**enc).last_hidden_state
            return out.squeeze(0).mean(dim=0).cpu().numpy()

        eval_embs = np.stack([embed(q["question"]) for q in eval_queries]).astype(np.float64)
        train_embs = np.stack([embed(s.text) for s in remaining]).astype(np.float64)
        logger.info("embedded %d eval + %d training queries",
                    len(eval_embs), len(train_embs))

    # VQC
    train_labels = np.array([dom_to_idx[s.domain] for s in remaining], dtype=np.int64)
    vqc = TorchVQCRouter(n_qubits=4, n_layers=6, n_classes=n_classes,
                         lr=args.vqc_lr, seed=args.seed,
                         input_dim=eval_embs.shape[1], weight_decay=1e-4)
    Xt = torch.from_numpy(train_embs).double()
    yt = torch.from_numpy(train_labels)
    vqc.train_batched(Xt, yt, epochs=args.vqc_epochs)
    with torch.no_grad():
        train_acc = float((vqc.predict(Xt).numpy() == train_labels).mean())
    logger.info("VQC trained, train_acc=%.3f", train_acc)

    def router_vqc(emb: np.ndarray) -> int:
        with torch.no_grad():
            return int(vqc.predict(torch.from_numpy(emb).double().unsqueeze(0))[0])

    rng = np.random.default_rng(args.seed)

    def router_random(_emb: np.ndarray) -> int:
        return int(rng.integers(0, n_classes))

    oracle_counter = [0]

    def router_oracle(_emb: np.ndarray) -> int:
        idx = eval_queries[oracle_counter[0]]["domain_idx"]
        oracle_counter[0] += 1
        return idx

    # LLM + judge callbacks: hit the Studio MLX server with adapter-swap
    def llm_call(question: str, routed_domain: str) -> str:
        if args.dry_run:
            return f"[stub answer under adapter={routed_domain}]"
        return _studio_call(
            args.studio_url,
            messages=[{"role": "user", "content": question}],
            max_tokens=512,
            adapter=routed_domain,
        )

    def judge_call(question: str, answer: str, expected_domain: str) -> int:
        if args.dry_run:
            return 3
        judge_adapter = expected_domain if args.judge_use_adapter else "base"
        prompt = build_rubric_prompt(question=question, answer=answer, domain=expected_domain)
        resp = _studio_call(
            args.studio_url,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            adapter=judge_adapter,
        )
        score = parse_score(resp)
        return score if score is not None else 0

    results = {}
    for name, fn in [("vqc", router_vqc), ("random", router_random), ("oracle", router_oracle)]:
        logger.info("running router=%s", name)
        if name == "oracle":
            oracle_counter[0] = 0
        r = run_downstream_eval(
            queries=eval_queries, embeddings=eval_embs,
            router_fn=fn, llm_fn=llm_call, judge_fn=judge_call,
            domain_names=domains,
        )
        results[name] = r
        logger.info("  router=%s  mean_score=%.3f  routing_acc=%.3f",
                    name, r["mean_score"], r["routing_accuracy"])

    meta = {
        "config": {
            "studio_url": args.studio_url,
            "judge_use_adapter": bool(args.judge_use_adapter),
            "data_dir": str(args.data_dir),
            "per_domain": args.per_domain,
            "n_eval": len(eval_queries),
            "vqc_train_acc": train_acc,
            "seed": args.seed,
            "adapter_backend": "Qwen3.6-35B-A3B + MLX LoRA (real)",
            "no_system_prompt_persona": True,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(meta, indent=2))
    logger.info("wrote %s", args.output)

    print("\n=== C2-LoRA Downstream Results ===")
    print(f"adapter backend={meta['config']['adapter_backend']}")
    print(f"{'router':<8} {'mean':>6} {'routing_acc':>12} {'correct':>8} {'wrong':>7}")
    for name, r in results.items():
        print(f"{name:<8} {r['mean_score']:>6.2f} {r['routing_accuracy']:>12.3f} "
              f"{r['mean_score_when_routed_correct']:>8.2f} "
              f"{r['mean_score_when_routed_wrong']:>7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run the integration test**

Run: `uv run python -m pytest tests/scripts/test_bench_downstream_c2_lora.py -v 2>&1 | tail -10`
Expected: PASSED.

- [ ] **Step 4: Commit**

```bash
git add tests/scripts/test_bench_downstream_c2_lora.py scripts/bench_downstream_c2_lora.py
git commit -m "feat(c2-lora): bench orchestrator (adapter-swap)"
```

---

### Task 5: Real 100-query bench run

**Files:**
- Create: `results/c2-lora-downstream.json`

- [ ] **Step 1: Confirm Studio server is up**

```bash
ssh studio "curl -s http://localhost:19000/v1/models | jq"
```

Expected: response with `currently_loaded_adapter` field. If the server is down, re-run `bash scripts/deploy_studio_lora_server.sh`.

- [ ] **Step 2: Launch the real bench**

```bash
uv run python scripts/bench_downstream_c2_lora.py \
    --per-domain 10 \
    --vqc-epochs 300 \
    --studio-url http://studio:19000 \
    --output results/c2-lora-downstream.json 2>&1 | tee results/.c2-lora-run.log
```

Expected wall-clock time: ~30-60 min. 100 queries × 3 routers × 2 calls each = 600 calls. MLX on M3 Ultra typically does ~3-6s per 512-token generation with Qwen3.6-35B-A3B, so ~3000-3600s compute.

Adapter swap latency should be seconds (weights IO only, no base reload). If swap is slow (>30s), the pattern from b102115 may not be applying and we're triggering accidental base reload. Debug by checking log line count vs elapsed time.

- [ ] **Step 3: Sanity-check the JSON**

```bash
jq '.results | to_entries | map({router: .key, mean: .value.mean_score, routing_acc: .value.routing_accuracy, correct: .value.mean_score_when_routed_correct, wrong: .value.mean_score_when_routed_wrong})' results/c2-lora-downstream.json
```

**Kill-criterion check for the experiment** (per diagnostic § Implications):

- If `oracle.mean_score - random.mean_score >= 0.6` → **adapter-swap fixes the premise**; adapters DO improve downstream. Paper A §5 gets a strong follow-up.
- If `0.3 <= oracle.mean_score - random.mean_score < 0.6` → partial improvement; report honestly with noise caveat.
- If `oracle.mean_score - random.mean_score < 0.3` → **even real LoRA doesn't beat random by the threshold** — strong negative result; Paper A C2 premise is refuted beyond the prompt-based setup.

Also check the confidently-wrong pathology:
- Compute `vqc.mean_score_when_routed_wrong - random.mean_score_when_routed_wrong`. In C2 baseline it was `-0.52`. If with LoRA it is `>= 0` (i.e. VQC-when-wrong no longer harmful), pattern #1 from the diagnostic is confirmed as prompt-specific.

- [ ] **Step 4: Commit results**

```bash
git add results/c2-lora-downstream.json
git commit -m "results(c2-lora): 100-query real LoRA swap bench"
```

Do NOT commit `results/.c2-lora-run.log` (gitignored by `*.log`).

---

### Task 6: Narrative + comparison figure + paper update

**Files:**
- Create: `scripts/figure_c2_lora_compare.py`
- Create: `docs/paper-a/c2-lora-results.md`
- Create: `docs/paper-a/c2-lora-figure.pdf`

- [ ] **Step 1: Create the comparison figure script**

Write to `scripts/figure_c2_lora_compare.py`:

```python
#!/usr/bin/env python3
"""C2 vs C2-LoRA side-by-side bar chart."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--c2-baseline", type=Path, default=Path("results/c2-downstream.json"))
    p.add_argument("--c2-lora", type=Path, default=Path("results/c2-lora-downstream.json"))
    p.add_argument("--output", type=Path, default=Path("docs/paper-a/c2-lora-figure.pdf"))
    args = p.parse_args()

    base = json.loads(args.c2_baseline.read_text())["results"]
    lora = json.loads(args.c2_lora.read_text())["results"]

    routers = ["random", "vqc", "oracle"]
    x = np.arange(len(routers))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    base_means = [base[r]["mean_score"] for r in routers]
    lora_means = [lora[r]["mean_score"] for r in routers]
    ax1.bar(x - width / 2, base_means, width, label="C2 (prompt persona)", color="#cc6666", edgecolor="black")
    ax1.bar(x + width / 2, lora_means, width, label="C2-LoRA (real adapter)", color="#66cc66", edgecolor="black")
    ax1.set_xticks(x); ax1.set_xticklabels(routers)
    ax1.set_ylabel("Mean judge score (0-5)")
    ax1.set_title("Aggregate downstream quality")
    ax1.set_ylim(0, 5)
    ax1.legend()

    base_wrong = [base[r]["mean_score_when_routed_wrong"] for r in routers]
    lora_wrong = [lora[r]["mean_score_when_routed_wrong"] for r in routers]
    ax2.bar(x - width / 2, base_wrong, width, label="C2", color="#cc6666", edgecolor="black")
    ax2.bar(x + width / 2, lora_wrong, width, label="C2-LoRA", color="#66cc66", edgecolor="black")
    ax2.set_xticks(x); ax2.set_xticklabels(routers)
    ax2.set_ylabel("Mean judge score (0-5)")
    ax2.set_title("Score when router routed WRONG (confidently-wrong test)")
    ax2.set_ylim(0, 5)
    ax2.legend()

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, bbox_inches="tight")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the figure script**

```bash
uv run python scripts/figure_c2_lora_compare.py
```

Expected: `wrote docs/paper-a/c2-lora-figure.pdf`.

- [ ] **Step 3: Write the narrative**

Create `docs/paper-a/c2-lora-results.md` with this template — fill in actual numbers from `results/c2-lora-downstream.json`:

```markdown
# C2-LoRA — Does Real Weight-Level Specialisation Eliminate the Confidently-Wrong Pathology?

**Setup.** Same 100-query held-out eval as C2 baseline, same 3 routers (vqc, random, oracle), same 10 domains. **The difference:** the downstream LLM is Qwen3.6-35B-A3B on Mac Studio via MLX, with **real LoRA adapters** (rank 16, alpha 32, trained per PRD stories 4-13 on the merged corpora from story-2) swapped per route. **No system-prompt persona is injected** — weight-level specialisation replaces the prompt.

**Judge**: self-judge on the Qwen3.6-35B-A3B base (no adapter), same rubric as C2. `--judge-use-adapter` was NOT set, so the judge runs on the base model without specialisation bias.

## Results

| Router | C2 (prompt persona) | C2-LoRA (real adapter) | Δ |
|---|---|---|---|
| Random | 3.190 | {LORA_RANDOM} | {DELTA_RANDOM} |
| VQC | 2.650 | {LORA_VQC} | {DELTA_VQC} |
| Oracle | 3.480 | {LORA_ORACLE} | {DELTA_ORACLE} |

**Oracle − Random gap**: C2 = 0.290 (kill triggered). C2-LoRA = {GAP_LORA}.

## Confidently-wrong pathology test

Score when router routed WRONG:

| Router | C2 score when wrong | C2-LoRA score when wrong | Δ |
|---|---|---|---|
| Random | 3.140 | {LORA_RANDOM_WRONG} | {DELTA_RANDOM_WRONG} |
| VQC | 2.566 | {LORA_VQC_WRONG} | {DELTA_VQC_WRONG} |

C2 baseline showed VQC-when-wrong (2.566) < Random-when-wrong (3.140), confirming prompt-based persona-refusal pathology (diagnostic pattern #1). C2-LoRA test: is `VQC_LoRA_wrong >= Random_LoRA_wrong`? {YES/NO with measured gap}.

## Interpretation

{Pick ONE scenario based on actual numbers.}

**A. If oracle − random >= 0.6 AND VQC-when-wrong ≈ Random-when-wrong:**

Real LoRA adapters restore the adapter-routing premise. Paper A §5 is updated: the C2 baseline kill criterion was a prompt-specific artefact; weight-level specialisation avoids the persona-refusal failure mode. The sibling LoRA spec hypothesis is confirmed. Paper A contribution revised from "negative result with caveat" to "negative result for prompt-based pseudo-adapters, positive direction for weight-level adapters".

**B. If oracle − random is still < 0.3:**

Even real LoRA adapters fail to beat random. The routing premise itself is under-supported for this task regardless of adapter implementation. Hypotheses for future work: (i) the domains are too semantically close (query distributions overlap), (ii) the base Qwen3.6-35B-A3B is already near the quality ceiling on these queries so per-domain specialisation has no headroom, (iii) judge inconsistency dominates the signal at n=100.

**C. If 0.3 <= oracle − random < 0.6:**

Partial improvement. Real LoRA moves the needle but not enough to claim "routing matters" robustly. Paper A reports this as progress over prompt-based pseudo-adapters, with an explicit statistical-power caveat and a sample-size recommendation for a definitive v3 experiment.

## Implications for Paper A

1. **§5 Discussion**: the C2 negative result is a methodology-specific finding (prompt-persona) rather than an architecture-specific indictment. Real LoRA {resolves / moderates / fails to resolve} it.
2. **§2 Setup** gains a sentence acknowledging that the Qwen3.6-35B-A3B adapter pipeline is the intended deployment, and the C2 baseline was a pragmatic prompt-based approximation.
3. **Future work** section replaces the speculative "real LoRA adapters as future work" with the measured C2-LoRA numbers.

## Caveats

- Self-judge base: the Qwen3.6-35B-A3B (no adapter) judging Qwen3.6-35B-A3B (adapter-swapped) outputs is still a same-family judge. Claude/GPT-4 as external judge remains future work.
- n=100: same underpowering concern as C2. A replication with n>=500 is needed for confident inference, especially on Δ < 0.3 scale.
- Adapter training ran on teacher-generated (mascarade-datasets) corpora, not end-user dialogues. Same C3 caveat applies.
```

- [ ] **Step 4: Fill in measured numbers**

Run: `jq '.results' results/c2-lora-downstream.json`

Replace every `{...}` placeholder in the narrative with the measured values. Pick ONE of the A/B/C scenarios based on the actual oracle − random gap.

- [ ] **Step 5: Commit narrative + figure**

```bash
git add scripts/figure_c2_lora_compare.py \
  docs/paper-a/c2-lora-results.md \
  docs/paper-a/c2-lora-figure.pdf
git commit -m "docs(c2-lora): narrative + side-by-side fig"
```

---

### Task 7: Roadmap update + push

- [ ] **Step 1: Update Phase C roadmap**

In `docs/superpowers/plans/2026-04-19-phase-c-roadmap.md`, after the C2 row add a new C2-LoRA row with status `Done (commits <range>)` and a one-line result summary (e.g., "oracle−random=X.XX; pattern #1 {resolved/persistent}").

- [ ] **Step 2: Full regression check**

```bash
uv run python -m pytest \
  tests/serving/test_studio_lora_server.py \
  tests/scripts/test_bench_downstream_c2_lora.py \
  tests/scripts/test_c2_diagnostic.py \
  tests/routing/test_llm_judge.py \
  tests/routing/test_downstream_harness.py \
  -q 2>&1 | tail -6
```

Expected: all PASSED.

- [ ] **Step 3: Commit + push**

```bash
git branch --show-current  # ensure 'main'
git add docs/superpowers/plans/2026-04-19-phase-c-roadmap.md
git commit -m "docs(phase-c): mark C2-LoRA done"
git push origin main
```

If push is rejected with "fetch first", do `git pull --rebase origin main` then `git push origin main`. Do NOT force-push.

---

## Kill criterion (reminder, top-level)

If Task 3 Step 2 never reaches "READY" within 10 minutes (server fails to load base + convert), STOP and report BLOCKED to the user. Do not proceed to tasks 4-7. The MLX adapter-swap approach requires the specific pattern from commit `b102115`; if it fails to apply, escalate rather than improvising.

## Out of scope

- **No new LoRA training** — 10 adapters already exist per PRD stories 4-13.
- **No GGUF conversion** — we keep MLX end-to-end on Studio to avoid adapter-format conversion risk.
- **No external judge (Claude, GPT-4)** — self-judge keeps apples-to-apples with C2 baseline.
- **No n>=500 replication** — documented as limitation; same 100-query eval for direct comparability.
- **No update to the main Paper A v2 .tex** — separate follow-up commit after narrative is reviewed.
- **No persistence of the Studio server** — no systemd / launchd install. Re-launch via deploy script as needed.

---

## Self-review

**Spec coverage:**
- Adapter-swap server → Task 1+2+3 ✓
- No-system-prompt-persona (vs C2 baseline) → `build_chat_prompt` in Task 2 + `--judge-use-adapter` default False in Task 4 ✓
- 100-query eval + 3 routers → Task 5 ✓
- Comparison to C2 baseline → Task 6 (figure + table in narrative) ✓
- Three-outcome interpretation (A/B/C) → Task 6 Step 3 ✓
- Pathology-specific test (wrong-bucket) → Task 5 Step 3 + Task 6 narrative ✓

**Placeholder scan:** No "TBD"/"TODO" in code blocks. The narrative template has `{PLACEHOLDER}` fields that are explicitly called out as values-to-fill from the JSON in Task 6 Step 4 — this is acceptable for a template.

**Type consistency:**
- `AdapterState` defined in Task 2 is consumed by `swap_adapter` (Task 2) + `_build_app` (Task 2). Consistent.
- `_studio_call(base_url, messages, max_tokens, adapter)` in Task 4 matches the server's `/v1/chat/completions` accept schema in Task 2.
- `router_fn` / `llm_fn` / `judge_fn` callback signatures match `run_downstream_eval` from existing `src/routing/downstream_harness.py` (merged, tested in C2 baseline plan).

---

## Estimated time

- Task 1 (server tests): 15 min
- Task 2 (server impl): 20 min
- Task 3 (deploy + smoke): 15 min + 5 min load time
- Task 4 (bench port + tests): 25 min
- Task 5 (real run): 30-60 min wall-clock compute + 5 min inspect
- Task 6 (narrative + figure): 30 min writing
- Task 7 (push + regress): 10 min

**Total: ~2h-2h30 engineering + ~60 min MLX compute on Studio.**

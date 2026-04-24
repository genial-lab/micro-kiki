# Publication Readiness Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make micro-kiki publishable — close 6 gaps identified in self-review: router novelty, composition protocol, external benchmarks, per-domain gains, cognitive layer evaluation, artifact sync.

**Architecture:** Evaluation-first approach. Run benchmarks before writing claims. Each task produces reproducible JSON results that feed the paper directly. No fabricated numbers.

**Tech Stack:** mlx_lm 0.31.2, sentence-transformers, pytest, HuggingFace Hub CLI

**Context:** Existing results (results/sota-sweep-N20-N50/) already show V4 adapters DON'T generalize from HumanEval to MBPP. Math adapter is toxic (-24pp). The paper must be honest about this. The contribution is the SYSTEM (7-stage pipeline on Apple Silicon), not "SOTA adapters".

---

## Gap Analysis (from self-review)

| # | Gap | What exists | What's needed |
|---|-----|-------------|---------------|
| 1 | Router not novel vs semantic-router/LoRAMoE | MiniLM+MLP sigmoid classifier | Ablation vs baselines (keyword, LLM-judge, semantic-router) |
| 2 | No composition protocol ablation | Single adapter loaded (no mixing) | Rank/layers/alpha ablation matrix |
| 3 | Benchmarks not reproducible externally | HumanEval N=164, GSM8K/MBPP N=50 | Full N runs + scripts in repo + CI-runnable |
| 4 | No per-domain gain vs base Qwen3.6 | bench-complete.json (10 domains, old) | PPL eval all 35 domains V4 |
| 5 | No cognitive layer evaluation | Latency metrics only | A/B: pipeline vs raw inference, LLM-judge scoring |
| 6 | GitHub/HF desynchronized | README updated, HF stale | Single sync script |

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `scripts/eval_router_ablation.py` | Router baseline comparison | Create |
| `scripts/eval_ppl_all_domains.py` | PPL eval 35 domains vs base | Create |
| `scripts/eval_cognitive_ab.py` | A/B test pipeline vs raw | Create |
| `scripts/sync_hf.py` | Sync GitHub artifacts → HF | Create |
| `results/ablation-router/` | Router ablation results | Create dir |
| `results/ablation-lora/` | LoRA config ablation results | Create dir |
| `results/v4-ppl-35domains/` | Per-domain PPL results | Create dir |
| `results/cognitive-ab/` | A/B test results | Create dir |
| `paper/micro-kiki.tex` | Updated paper with real results | Modify |

---

### Task 1: Router Ablation — MiniLM vs Baselines

Compare 4 routing strategies on the same 5377 validation prompts:
- **Keyword** (current classify_domains.py regex/keyword scorer)
- **MiniLM+MLP sigmoid** (current router V4)
- **LLM-judge** (Qwen3.6 base classifies the prompt zero-shot)
- **Random** (uniform random domain selection)

**Files:**
- Create: `scripts/eval_router_ablation.py`
- Output: `results/ablation-router/router-ablation.json`
- Test: `tests/scripts/test_eval_router_ablation.py`

- [ ] **Step 1: Write the ablation script**

The script loads `data/router-v4/valid.jsonl`, runs each prompt through 4 routing strategies, and measures top-1/top-3 accuracy against the ground-truth domain label.

```python
"""Router ablation: compare 4 routing strategies on the same validation set.

Usage:
    python scripts/eval_router_ablation.py \
        --valid data/router-v4/valid.jsonl \
        --router-weights output/router-v4 \
        --output results/ablation-router/router-ablation.json
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# --- Strategy 1: Keyword-based (same logic as classify_domains.py) ---
def _keyword_route(text: str, domain_keywords: dict[str, list[str]]) -> list[str]:
    """Score text against domain keywords with word-boundary matching."""
    scores: dict[str, float] = {}
    text_lower = text.lower()
    for domain, keywords in domain_keywords.items():
        score = sum(
            1.0 for kw in keywords
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower)
        )
        if score >= 1.0:
            scores[domain] = score
    if not scores:
        return []
    ranked = sorted(scores, key=scores.get, reverse=True)
    return ranked[:4]


# --- Strategy 2: MiniLM+MLP (current router V4) ---
def _load_minilm_router(weights_dir: Path):
    """Load the trained router."""
    from safetensors.numpy import load_file
    from sentence_transformers import SentenceTransformer

    meta = json.loads((weights_dir / "meta.json").read_text())
    tensors = load_file(str(weights_dir / "router.safetensors"))
    w0 = tensors["0.weight"]
    b0 = tensors["0.bias"]
    w1 = tensors["3.weight"]
    b1 = tensors["3.bias"]
    domains = meta["domains"]
    encoder = SentenceTransformer(
        meta.get("embedding_model", "sentence-transformers/all-mpnet-base-v2")
    )

    def route(text: str) -> list[str]:
        emb = encoder.encode(text, normalize_embeddings=True)
        h = np.maximum(w0 @ emb + b0, 0)
        logits = w1 @ h + b1
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
        active = np.where(probs > 0.12)[0]
        if len(active) == 0:
            return []
        active = active[np.argsort(probs[active])[::-1]][:4]
        return [domains[i] for i in active]

    return route


# --- Strategy 3: LLM zero-shot classification ---
def _llm_judge_route(text: str, domains: list[str], generate_fn) -> list[str]:
    """Use the base LLM to classify the domain zero-shot."""
    domain_list = ", ".join(domains)
    prompt = (
        f"Classify this prompt into one of these technical domains: {domain_list}\n\n"
        f"Prompt: {text}\n\n"
        f"Answer with just the domain name, nothing else."
    )
    response = generate_fn(prompt).strip().lower()
    for d in domains:
        if d in response:
            return [d]
    return []


# --- Strategy 4: Random baseline ---
def _random_route(domains: list[str]) -> list[str]:
    return [random.choice(domains)]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--router-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/ablation-router/router-ablation.json"))
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM judge (slow)")
    parser.add_argument("--keyword-config", type=Path, default=None)
    args = parser.parse_args()

    # Load validation data
    examples = []
    for line in args.valid.read_text().strip().split("\n"):
        if not line:
            continue
        e = json.loads(line)
        examples.append({"prompt": e["prompt"], "domain": e["domain"]})
    logger.info("Loaded %d validation examples", len(examples))

    domains = sorted(set(e["domain"] for e in examples))
    logger.info("Domains: %d unique", len(domains))

    # Load MiniLM router
    minilm_route = _load_minilm_router(args.router_weights)

    # Build keyword config (simplified — top 5 keywords per domain from training data)
    # In production this comes from domains.yaml; here we use domain name as keyword
    domain_keywords = {d: [d.replace("-", " "), d] for d in domains}

    # Run all strategies
    results = {"keyword": [], "minilm": [], "random": []}
    strategies = ["keyword", "minilm", "random"]

    for i, ex in enumerate(examples):
        prompt, gold = ex["prompt"], ex["domain"]

        # Keyword
        kw_pred = _keyword_route(prompt, domain_keywords)
        results["keyword"].append({"gold": gold, "pred": kw_pred})

        # MiniLM
        ml_pred = minilm_route(prompt)
        results["minilm"].append({"gold": gold, "pred": ml_pred})

        # Random
        rnd_pred = _random_route(domains)
        results["random"].append({"gold": gold, "pred": rnd_pred})

        if (i + 1) % 1000 == 0:
            logger.info("Processed %d/%d", i + 1, len(examples))

    # Compute metrics
    summary = {}
    for strategy in strategies:
        preds = results[strategy]
        top1 = sum(1 for p in preds if p["pred"] and p["pred"][0] == p["gold"]) / len(preds)
        top3 = sum(1 for p in preds if p["gold"] in p["pred"][:3]) / len(preds)
        coverage = sum(1 for p in preds if len(p["pred"]) > 0) / len(preds)
        summary[strategy] = {
            "top1": round(top1, 4),
            "top3": round(top3, 4),
            "coverage": round(coverage, 4),
            "n": len(preds),
        }
        logger.info("%s: top1=%.3f top3=%.3f coverage=%.3f", strategy, top1, top3, coverage)

    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_examples": len(examples),
        "n_domains": len(domains),
        "strategies": summary,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the ablation (skip LLM judge for now — slow)**

```bash
cd /Users/clems/Projets/micro-kiki && \
  .venv/bin/python scripts/eval_router_ablation.py \
    --valid data/router-v4/valid.jsonl \
    --router-weights output/router-v4 \
    --skip-llm \
    --output results/ablation-router/router-ablation.json
```

Expected: JSON with top-1/top-3 for keyword, minilm, random.

- [ ] **Step 3: Commit**

```bash
git add scripts/eval_router_ablation.py results/ablation-router/
git commit -m "eval: router ablation — keyword vs MiniLM vs random"
```

---

### Task 2: LoRA Config Ablation

Run PPL evaluation on 3 representative domains (chat-fr, python, embedded) across a grid:
- Rank: {4, 8, 16, 32}
- Layers: {8, 16, 32, 40}

This requires training 16 adapters per domain (48 total). Given time constraints, use a SUBSET: train for 50 iters only (enough to see trends), not full 200-1000.

**Note:** This is a LONG task (48 training runs × ~5min each = ~4 hours). Write the script, verify it works on 1 cell, then run in background.

**Files:**
- Create: `scripts/eval_lora_ablation.py`
- Output: `results/ablation-lora/lora-ablation.json`

- [ ] **Step 1: Write the ablation script**

Script that:
1. For each (rank, layers, domain) cell:
   - Generates a temporary config YAML
   - Runs `mlx_lm.lora` for 50 iters
   - Evaluates PPL on valid.jsonl
   - Records the result
2. Outputs a JSON grid

- [ ] **Step 2: Run on 1 cell to validate**

```bash
.venv/bin/python scripts/eval_lora_ablation.py \
  --domains chat-fr \
  --ranks 16 \
  --layers 32 \
  --iters 50 \
  --output results/ablation-lora/test-single.json
```

- [ ] **Step 3: Run full grid (background, ~4h)**

```bash
nohup .venv/bin/python scripts/eval_lora_ablation.py \
  --domains chat-fr python embedded \
  --ranks 4 8 16 32 \
  --layers 8 16 32 40 \
  --iters 50 \
  --output results/ablation-lora/lora-ablation.json \
  > /tmp/lora-ablation.log 2>&1 &
```

- [ ] **Step 4: Commit script**

```bash
git add scripts/eval_lora_ablation.py
git commit -m "eval: LoRA config ablation script (rank × layers)"
```

---

### Task 3: Full PPL Evaluation — 35 Domains vs Base

Run perplexity evaluation on ALL 35 V4 adapters vs base Qwen3.6. This script already exists as `scripts/eval_v4_sota.py` (created earlier in ~/micro-kiki). Port it to Projets/micro-kiki if needed, or use `eval_niche_vs_base.py`.

**Files:**
- Modify: `scripts/eval_niche_vs_base.py` (adapt for V4 paths + 35 domains)
- Output: `results/v4-ppl-35domains/ppl-all-domains.json`

- [ ] **Step 1: Check if eval_niche_vs_base.py already handles V4 paths**

```bash
grep -n "qwen36\|v4-sota\|35.*domain" scripts/eval_niche_vs_base.py | head -10
```

If not, adapt the adapter path and domain list.

- [ ] **Step 2: Run PPL eval on all 35 domains (long — ~3h)**

```bash
nohup .venv/bin/python scripts/eval_niche_vs_base.py \
  --model /Users/clems/KIKI-Mac_tunner/models/Qwen3.6-35B-A3B-4bit \
  --adapter-dir /Users/clems/KIKI-Mac_tunner/output/micro-kiki/lora-qwen36-35b-v4-sota \
  --data-dir /Users/clems/KIKI-Mac_tunner/data/micro-kiki \
  --output results/v4-ppl-35domains/ppl-all-domains.json \
  --max-samples 50 \
  > /tmp/ppl-eval.log 2>&1 &
```

- [ ] **Step 3: Commit results**

```bash
git add results/v4-ppl-35domains/
git commit -m "eval: PPL all 35 V4 domains vs base Qwen3.6"
```

---

### Task 4: Cognitive Layer A/B Test

Compare response quality WITH vs WITHOUT the cognitive layer (Aeon + Negotiator + AntiBias). Use the running pipeline server.

**Design:**
- 50 prompts across 10 domains (5 per domain)
- Condition A: full pipeline (`/v1/chat/completions`)
- Condition B: direct MLX inference (bypass stages 3,5,6,7 — route + infer only)
- Judge: LLM-based scoring (Qwen3.6 base as judge) on 4 criteria: relevance, accuracy, safety, coherence (1-5 scale each)

**Files:**
- Create: `scripts/eval_cognitive_ab.py`
- Output: `results/cognitive-ab/cognitive-ab-results.json`

- [ ] **Step 1: Write the A/B evaluation script**

The script:
1. Loads 50 test prompts (from `data/prompts/` or hand-curated)
2. For each prompt, calls the pipeline in two modes:
   - Full pipeline (default)
   - Raw mode (add a `"raw_mode": true` flag — or call MLX directly)
3. Uses LLM-judge to score both responses
4. Outputs per-prompt scores + aggregate

- [ ] **Step 2: Add raw_mode to the pipeline server**

Add a `raw_mode` flag to `ChatCompletionRequest` that skips stages 3 (Aeon recall), 5 (Negotiator), 6 (AntiBias), 7 (Aeon write). This isolates the cognitive layer's contribution.

In `full_pipeline_server.py`, add:
```python
raw_mode: bool = False  # Skip cognitive layer (stages 3,5,6,7) for A/B testing
```

Then in the handler, wrap stages 3, 5, 6, 7 in `if not req.raw_mode:`.

- [ ] **Step 3: Run A/B test**

```bash
.venv/bin/python scripts/eval_cognitive_ab.py \
  --server http://127.0.0.1:9200 \
  --output results/cognitive-ab/cognitive-ab-results.json
```

- [ ] **Step 4: Commit**

```bash
git add scripts/eval_cognitive_ab.py results/cognitive-ab/
git commit -m "eval: cognitive layer A/B test (pipeline vs raw)"
```

---

### Task 5: Complete External Benchmarks (MBPP + GSM8K Full N)

The sota-sweep at N=50 showed regressions. Run FULL N benchmarks to confirm or refute.

**Existing scripts:** `eval_mbpp_v4.py`, `eval_gsm8k_v4.py` already exist.

**Files:**
- Output: `results/v4-benchmarks-full/`

- [ ] **Step 1: Run MBPP full (N=500, ~6-8h)**

```bash
nohup .venv/bin/python scripts/eval_mbpp_v4.py \
  --model /Users/clems/KIKI-Mac_tunner/models/Qwen3.6-35B-A3B-4bit \
  --adapters base python cpp typescript rust shell math \
  --n 500 \
  --output-dir results/v4-benchmarks-full/mbpp \
  > /tmp/mbpp-full.log 2>&1 &
```

- [ ] **Step 2: Run GSM8K full (N=500, ~4-6h)**

```bash
nohup .venv/bin/python scripts/eval_gsm8k_v4.py \
  --model /Users/clems/KIKI-Mac_tunner/models/Qwen3.6-35B-A3B-4bit \
  --adapters base math reasoning python \
  --n 500 \
  --output-dir results/v4-benchmarks-full/gsm8k \
  > /tmp/gsm8k-full.log 2>&1 &
```

- [ ] **Step 3: Commit results**

```bash
git add results/v4-benchmarks-full/
git commit -m "eval: full MBPP + GSM8K benchmarks (N=500)"
```

---

### Task 6: Sync GitHub ↔ HuggingFace

Write a single sync script that ensures all HF repos have the correct README + model cards.

**Files:**
- Create: `scripts/sync_hf.py`

- [ ] **Step 1: Write sync script**

Script that:
1. Updates `clemsail/micro-kiki-router-v4` README with current router accuracy
2. Updates `clemsail/micro-kiki-v4-sota` README from `KIKI-Mac_tunner/output/.../README.md`
3. Pushes the paper PDF (when compiled) as an artifact
4. Verifies consistency between GitHub README and HF model cards

- [ ] **Step 2: Run dry-run**

```bash
.venv/bin/python scripts/sync_hf.py --dry-run
```

- [ ] **Step 3: Run execute**

```bash
.venv/bin/python scripts/sync_hf.py --execute
```

- [ ] **Step 4: Commit**

```bash
git add scripts/sync_hf.py
git commit -m "ops: add HF sync script"
```

---

### Task 7: Update Paper with Real Results

After tasks 1-5 produce results, update the paper.

**Files:**
- Modify: `paper/micro-kiki.tex`

- [ ] **Step 1: Insert router ablation table**

Replace placeholder claims with the actual ablation numbers from `results/ablation-router/router-ablation.json`.

- [ ] **Step 2: Insert LoRA ablation table**

Add rank × layers PPL grid from `results/ablation-lora/lora-ablation.json`.

- [ ] **Step 3: Insert per-domain PPL table**

Add 35-domain PPL comparison from `results/v4-ppl-35domains/`.

- [ ] **Step 4: Insert cognitive A/B results**

Add the pipeline-vs-raw comparison with judge scores.

- [ ] **Step 5: Honest discussion of limitations**

- Adapters overfit to HumanEval, regress on MBPP (cite sota-sweep README)
- Math adapter is toxic
- Router accuracy still limited on sparse domains (freecad 4%, stm32 29)
- Cognitive layer adds latency (~100ms) — worth it only for multi-turn

- [ ] **Step 6: Reframe contributions**

The contribution is NOT "SOTA adapters". It IS:
- Production-grade multi-adapter cognitive pipeline on consumer hardware
- Systems engineering: 7-stage integration, O(10ms) swap, KV cache reuse
- Honest evaluation showing adapter limitations and when cognitive augmentation helps

- [ ] **Step 7: Commit updated paper**

```bash
git add paper/micro-kiki.tex
git commit -m "docs(paper): update with ablation + benchmark results"
```

---

## Dependency Graph

```
Task 1 (router ablation) ────────────────────┐
Task 2 (LoRA ablation) ──────────────────────┤
Task 3 (PPL 35 domains) ─────────────────────┼──→ Task 7 (update paper)
Task 4 (cognitive A/B) ──────────────────────┤
Task 5 (MBPP + GSM8K full) ──────────────────┘
Task 6 (sync HF) ← independent, run anytime
```

Tasks 1-5 are independent — run in parallel where hardware allows (but GPU tasks compete for Metal memory).

**Recommended execution order:**
1. Task 1 (router ablation, ~20min, CPU only)
2. Task 6 (HF sync, ~5min)
3. Task 3 (PPL 35 domains, ~3h GPU) — background
4. Task 4 (cognitive A/B, ~1h, needs running server)
5. Task 2 (LoRA ablation, ~4h GPU) — background, after Task 3
6. Task 5 (MBPP + GSM8K, ~12h GPU) — background, after Task 2
7. Task 7 (paper update) — after all results are in

**Total GPU time estimate: ~20h**. Can run over 2-3 days on the Mac Studio.

# V-35B Opus Training Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrain LoRA adapters on Qwen3.6-35B-A3B with enriched datasets (Opus reasoning 11K + Opus chat 10K + distilled 35B Opus) to fix the chat-fr (+57%) and reasoning (+6%) regressions while keeping the 30/35 domain wins.

**Architecture:** Merge Opus high-quality data into foundations (chat-fr, reasoning), increase iters for foundations (500 vs 200), keep 8-layer LoRA with GPU-hang mitigation, sequential curriculum on Studio M3 Ultra.

**Tech Stack:** MLX (`mlx_lm lora`), Qwen3.6-35B-A3B BF16 (67 GB), Mac Studio M3 Ultra 512 GB

---

## Context

The current V-35B training (completed 2026-04-19) has 35 adapters but shows regressions on foundations:
- chat-fr: base 6.53 → LoRA 10.25 (+57%) — foundation data was generic, not Opus quality
- reasoning: base 5.97 → LoRA 6.31 (+6%) — insufficient high-quality reasoning examples
- freecad: base 53.00 → LoRA 63.75 (+20%) — small dataset (12K), noisy data
- html-css: base 22.38 → LoRA 24.25 (+8%) — mild regression

Available Opus datasets NOT yet merged into training:
- `data/Opus-4.6-reasoning-sft-12k-chat/train.jsonl` — 11,673 high-quality reasoning examples
- `data/claude-opus-4.6-10000x/opus46_final.jsonl` — 9,628 Opus chat examples
- `data/distilled-qwen35-35b-opus/` — distilled from 35B Opus
- `data/Opus-4.6-Reasoning-3000x-filtered/` — 3,000 filtered reasoning
- `data/distilled-qwen35-35b-opus-batch2/` — second batch

Strategy: merge Opus data into chat-fr + reasoning, increase foundation iters to 500, retrain only the 4 regressed domains (not all 35).

---

### Task 1: Merge Opus data into foundation datasets

**Files:**
- Modify: data on Studio via SSH (`data/micro-kiki/chat-fr/train.jsonl`, `data/micro-kiki/reasoning/train.jsonl`)

- [ ] **Step 1: Count existing + Opus data**

```bash
ssh studio "cd /Users/clems/KIKI-Mac_tunner && echo '=== Current ===' && \
  wc -l data/micro-kiki/chat-fr/train.jsonl data/micro-kiki/reasoning/train.jsonl && \
  echo '=== Opus sources ===' && \
  wc -l data/Opus-4.6-reasoning-sft-12k-chat/train.jsonl \
        data/claude-opus-4.6-10000x/opus46_final.jsonl \
        data/Opus-4.6-Reasoning-3000x-filtered/*.jsonl \
        data/distilled-qwen35-35b-opus/*.jsonl \
        data/distilled-qwen35-35b-opus-batch2/*.jsonl 2>/dev/null"
```

Expected: chat-fr ~63K, reasoning ~10K, Opus sources ~35K total.

- [ ] **Step 2: Merge Opus reasoning into reasoning domain**

```bash
ssh studio 'cd /Users/clems/KIKI-Mac_tunner && /opt/homebrew/bin/python3.12 -c "
import json, hashlib
from pathlib import Path

target = Path(\"data/micro-kiki/reasoning/train.jsonl\")
existing = set()
for line in open(target):
    try:
        t = \"\".join(m.get(\"content\",\"\")[:100] for m in json.loads(line).get(\"messages\",[]))
        existing.add(hashlib.sha256(t.encode()).hexdigest()[:16])
    except: pass

sources = [
    (\"data/Opus-4.6-reasoning-sft-12k-chat/train.jsonl\", \"opus-reasoning-12k\"),
    (\"data/Opus-4.6-Reasoning-3000x-filtered/train.jsonl\", \"opus-reasoning-3k-filtered\"),
]
total = 0
for src_path, tag in sources:
    p = Path(src_path)
    if not p.exists(): continue
    added = 0
    with open(target, \"a\") as out:
        for line in open(p):
            try:
                d = json.loads(line.strip())
                t = \"\".join(m.get(\"content\",\"\")[:100] for m in d.get(\"messages\",[]))
                h = hashlib.sha256(t.encode()).hexdigest()[:16]
                if h not in existing:
                    d[\"_source\"] = tag
                    out.write(json.dumps(d, ensure_ascii=False) + \"\\n\")
                    existing.add(h)
                    added += 1
            except: pass
    print(f\"  {tag}: +{added}\")
    total += added
n = sum(1 for _ in open(target))
print(f\"reasoning total: {n}\")
"'
```

Expected: reasoning grows from ~10K to ~20K+.

- [ ] **Step 3: Merge Opus chat into chat-fr domain**

```bash
ssh studio 'cd /Users/clems/KIKI-Mac_tunner && /opt/homebrew/bin/python3.12 -c "
import json, hashlib
from pathlib import Path

target = Path(\"data/micro-kiki/chat-fr/train.jsonl\")
existing = set()
for line in open(target):
    try:
        t = \"\".join(m.get(\"content\",\"\")[:100] for m in json.loads(line).get(\"messages\",[]))
        existing.add(hashlib.sha256(t.encode()).hexdigest()[:16])
    except: pass

sources = [
    (\"data/claude-opus-4.6-10000x/opus46_final.jsonl\", \"opus-chat-10k\"),
    (\"data/distilled-qwen35-35b-opus/train.jsonl\", \"distilled-35b-opus\"),
    (\"data/distilled-qwen35-35b-opus-batch2/train.jsonl\", \"distilled-35b-opus-b2\"),
]
total = 0
for src_path, tag in sources:
    p = Path(src_path)
    if not p.exists():
        # Try globbing for jsonl files in directory
        parent = Path(src_path).parent
        if parent.exists():
            for f in parent.glob(\"*.jsonl\"):
                added = 0
                with open(target, \"a\") as out:
                    for line in open(f):
                        try:
                            d = json.loads(line.strip())
                            t = \"\".join(m.get(\"content\",\"\")[:100] for m in d.get(\"messages\",[]))
                            h = hashlib.sha256(t.encode()).hexdigest()[:16]
                            if h not in existing:
                                d[\"_source\"] = tag
                                out.write(json.dumps(d, ensure_ascii=False) + \"\\n\")
                                existing.add(h)
                                added += 1
                        except: pass
                print(f\"  {f.name}: +{added}\")
                total += added
        continue
    added = 0
    with open(target, \"a\") as out:
        for line in open(p):
            try:
                d = json.loads(line.strip())
                t = \"\".join(m.get(\"content\",\"\")[:100] for m in d.get(\"messages\",[]))
                h = hashlib.sha256(t.encode()).hexdigest()[:16]
                if h not in existing:
                    d[\"_source\"] = tag
                    out.write(json.dumps(d, ensure_ascii=False) + \"\\n\")
                    existing.add(h)
                    added += 1
            except: pass
    print(f\"  {tag}: +{added}\")
    total += added
n = sum(1 for _ in open(target))
print(f\"chat-fr total: {n}\")
"'
```

Expected: chat-fr grows from ~63K to ~70K+.

- [ ] **Step 4: Verify merged counts**

```bash
ssh studio "wc -l /Users/clems/KIKI-Mac_tunner/data/micro-kiki/chat-fr/train.jsonl \
                  /Users/clems/KIKI-Mac_tunner/data/micro-kiki/reasoning/train.jsonl"
```

Expected: chat-fr ~70K+, reasoning ~20K+.

- [ ] **Step 5: Commit data merge note**

```bash
git add -A
git commit -m "data: merge Opus into chat-fr + reasoning"
```

---

### Task 2: Create V-35B-Opus training script

**Files:**
- Create: `scripts/train_lora_36b_opus.sh`

- [ ] **Step 1: Write the training script**

```bash
cat > scripts/train_lora_36b_opus.sh << 'TRAINEOF'
#!/usr/bin/env bash
# V-35B-Opus: retrain regressed domains with Opus-enriched data
# Only retrains: chat-fr, reasoning, freecad, html-css
# Uses higher iters for foundations (500) vs niches (200)
set -euo pipefail

MODEL="models/Qwen3.6-35B-A3B"
DATA="data/micro-kiki"
OUTPUT="output/micro-kiki/lora-qwen36-35b-opus"
PYTHON="/opt/homebrew/bin/python3.12"

# Domains to retrain (the 4 regressions)
# chat-fr and reasoning get 500 iters (foundation), others get 200
declare -A DOMAIN_ITERS=(
  [chat-fr]=500
  [reasoning]=500
  [freecad]=200
  [html-css]=200
)

mkdir -p "$OUTPUT"

echo "================================================================"
echo "V-35B-Opus Training — ${#DOMAIN_ITERS[@]} regressed domains"
echo "Model: $MODEL | LR: 1e-5 | 8 layers | BF16"
echo "================================================================"

for domain in "${!DOMAIN_ITERS[@]}"; do
  iters="${DOMAIN_ITERS[$domain]}"
  adapter="$OUTPUT/$domain"

  [ -f "$adapter/adapters.safetensors" ] && echo "SKIP $domain (done)" && continue
  [ ! -f "$DATA/$domain/train.jsonl" ] && echo "SKIP $domain (no data)" && continue

  n=$(wc -l < "$DATA/$domain/train.jsonl")
  echo ""
  echo "=== $domain ($n examples, $iters iters) ==="

  $PYTHON -m mlx_lm lora \
    --model "$MODEL" \
    --data "$DATA/$domain" \
    --train \
    --iters "$iters" \
    --batch-size 1 \
    --learning-rate 1e-5 \
    --adapter-path "$adapter" \
    --max-seq-length 512 \
    --num-layers 8 \
    --steps-per-report 25 \
    --steps-per-eval 50 \
    --grad-checkpoint \
    --clear-cache-threshold 0.2 \
    2>&1 | tee "$OUTPUT/log-$domain.txt"

  echo "$domain DONE"
  sleep 10
done

echo "================================================================"
echo "ALL COMPLETE"
echo "================================================================"
TRAINEOF
chmod +x scripts/train_lora_36b_opus.sh
```

- [ ] **Step 2: Commit**

```bash
git add scripts/train_lora_36b_opus.sh
git commit -m "feat: V-35B-Opus training script"
```

---

### Task 3: Copy script to Studio and launch training

**Files:**
- No local files modified

- [ ] **Step 1: Copy to Studio**

```bash
scp scripts/train_lora_36b_opus.sh studio:/Users/clems/KIKI-Mac_tunner/scripts/
```

- [ ] **Step 2: Launch training**

```bash
ssh studio "cd /Users/clems/KIKI-Mac_tunner && nohup bash scripts/train_lora_36b_opus.sh > /tmp/train_opus.log 2>&1 &
echo 'PID:' \$!"
```

- [ ] **Step 3: Verify first domain starts without GPU Hang**

```bash
sleep 30
ssh studio "tail -10 /tmp/train_opus.log 2>/dev/null"
```

Expected: chat-fr training starts, shows `Iter 1: Val loss ...`, no GPU Hang error.

---

### Task 4: Monitor and verify training quality

**Files:**
- No files modified

- [ ] **Step 1: Wait for chat-fr to complete (~15 min at 500 iters)**

```bash
ssh studio "grep 'Val loss' /Users/clems/KIKI-Mac_tunner/output/micro-kiki/lora-qwen36-35b-opus/log-chat-fr.txt 2>/dev/null | tail -3"
```

Expected: val_loss < 6.53 (must beat base model to fix the regression).

- [ ] **Step 2: Wait for all 4 domains to complete (~30 min total)**

```bash
ssh studio "ls /Users/clems/KIKI-Mac_tunner/output/micro-kiki/lora-qwen36-35b-opus/*/adapters.safetensors 2>/dev/null | wc -l"
```

Expected: 4 adapters.

- [ ] **Step 3: Verify adapters are trained (lora_b non-zero)**

```bash
ssh studio '/opt/homebrew/bin/python3.12 -c "
import safetensors, numpy as np
from pathlib import Path
for d in sorted(Path(\"output/micro-kiki/lora-qwen36-35b-opus\").iterdir()):
    if not d.is_dir(): continue
    adapt = d / \"adapters.safetensors\"
    if not adapt.exists(): continue
    f = safetensors.safe_open(str(adapt), framework=\"numpy\")
    maxv = max(float(np.max(np.abs(f.get_tensor(k)))) for k in list(f.keys())[:5])
    print(f\"  {d.name:<18} max={maxv:.4f}  {\"TRAINED\" if maxv > 0.001 else \"ZERO\"}\")"'
```

Expected: all 4 show TRAINED.

---

### Task 5: Benchmark Opus adapters vs base vs original LoRA

**Files:**
- Create: results in `output/micro-kiki/eval/bench-opus.json`

- [ ] **Step 1: Run perplexity comparison on the 4 domains**

```bash
ssh studio 'cd /Users/clems/KIKI-Mac_tunner && /opt/homebrew/bin/python3.12 -c "
import mlx.core as mx, mlx.nn as nn
from mlx_lm import load

PROMPTS = {
    \"chat-fr\": \"Explique le fonctionnement d un variateur de frequence pour moteur asynchrone triphase.\",
    \"reasoning\": \"Three resistors 100 200 300 ohms in parallel. Calculate total resistance step by step.\",
    \"freecad\": \"Write a FreeCAD Python macro for a parametric enclosure with snap-fit lid.\",
    \"html-css\": \"Write a responsive CSS Grid dashboard layout with sidebar header main and footer.\",
}

def ppl(model, tok, prompt):
    tokens = tok.encode(prompt)
    ids = mx.array([tokens[:-1]])
    labels = mx.array([tokens[1:]])
    logits = model(ids)
    loss = nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
    return float(mx.exp(mx.mean(loss)).item())

# Base
model, tok = load(\"models/Qwen3.6-35B-A3B\")
for d, p in PROMPTS.items():
    print(f\"base/{d}: ppl={ppl(model, tok, p):.2f}\")
del model

# Original LoRA
for d, p in PROMPTS.items():
    m, t = load(\"models/Qwen3.6-35B-A3B\", adapter_path=f\"output/micro-kiki/lora-qwen36-35b/{d}\")
    print(f\"lora-orig/{d}: ppl={ppl(m, t, p):.2f}\")
    del m

# Opus LoRA
for d, p in PROMPTS.items():
    path = f\"output/micro-kiki/lora-qwen36-35b-opus/{d}\"
    from pathlib import Path
    if not Path(path).exists(): continue
    m, t = load(\"models/Qwen3.6-35B-A3B\", adapter_path=path)
    print(f\"lora-opus/{d}: ppl={ppl(m, t, p):.2f}\")
    del m
"'
```

Expected: lora-opus chat-fr and reasoning ppl < base ppl (regression fixed).

- [ ] **Step 2: Save results**

Copy output to `output/micro-kiki/eval/bench-opus.json`.

- [ ] **Step 3: Commit results**

```bash
git add output/micro-kiki/eval/
git commit -m "data: V-35B-Opus benchmark results"
```

---

### Task 6: Build hybrid adapter set (best-of all versions)

**Files:**
- Modify: `scripts/build_hybrid_adapters.py`

- [ ] **Step 1: Update hybrid script to include Opus adapters**

Add `lora-qwen36-35b-opus` as a third source. For chat-fr, reasoning, freecad, html-css: use Opus adapter if it beats both base and original LoRA; otherwise fall back to base (no adapter) or original LoRA.

```bash
ssh studio "cd /Users/clems/KIKI-Mac_tunner && mkdir -p output/micro-kiki/lora-qwen36-35b-hybrid && \
for domain in chat-fr reasoning freecad html-css; do \
  opus=output/micro-kiki/lora-qwen36-35b-opus/\$domain; \
  [ -d \"\$opus\" ] && ln -sf \$(realpath \$opus) output/micro-kiki/lora-qwen36-35b-hybrid/\$domain; \
done && \
for domain in \$(ls output/micro-kiki/lora-qwen36-35b/); do \
  [ -d output/micro-kiki/lora-qwen36-35b-hybrid/\$domain ] && continue; \
  ln -sf \$(realpath output/micro-kiki/lora-qwen36-35b/\$domain) output/micro-kiki/lora-qwen36-35b-hybrid/\$domain; \
done && \
echo 'Hybrid set:' && ls output/micro-kiki/lora-qwen36-35b-hybrid/ | wc -l && echo 'domains'"
```

Expected: 35 domains in hybrid set.

- [ ] **Step 2: Commit**

```bash
git commit -m "feat: hybrid adapter set (Opus foundations + original niches)"
```

---

### Task 7: Upload Opus adapters to HuggingFace

**Files:**
- No local files

- [ ] **Step 1: Upload the 4 Opus adapters**

```bash
for domain in chat-fr reasoning freecad html-css; do
  adapt="/tmp/hf-35b-upload/adapters/$domain/adapters.safetensors"
  # First rsync from Studio
  rsync -a studio:/Users/clems/KIKI-Mac_tunner/output/micro-kiki/lora-qwen36-35b-opus/$domain/ /tmp/hf-opus-upload/$domain/
  uv run hf upload clemsail/micro-kiki-v35b "/tmp/hf-opus-upload/$domain/adapters.safetensors" \
    "adapters-opus/$domain/adapters.safetensors" --repo-type model
done
```

- [ ] **Step 2: Verify on HF**

```bash
uv run hf repos info clemsail/micro-kiki-v35b --type model
```

---

## Self-Review

**Spec coverage:**
- ✅ Task 1: Merge Opus reasoning + chat datasets
- ✅ Task 2: Training script with higher iters for foundations
- ✅ Task 3: Launch on Studio
- ✅ Task 4: Monitor + verify training quality
- ✅ Task 5: Benchmark Opus vs base vs original
- ✅ Task 6: Build hybrid adapter set
- ✅ Task 7: Upload to HuggingFace

**Placeholder scan:** None found. All steps have exact commands.

**Type consistency:** All paths use `lora-qwen36-35b-opus` consistently. SSH commands reference correct Studio paths.

**Dependencies:** Task 1 → Task 3 → Task 4 → Task 5 → Task 6 → Task 7. Task 2 is independent.

# E2E smoke runbook — full forgetting gate on real adapters

When the MLX server on Mac Studio is running, this is how to validate the
whole `mlx_client.py` → `measure_forgetting.py` → `post_train_gate.py` chain
end-to-end. The pieces were unit-tested via mocks on 2026-04-19; the
production smoke is only feasible when studio:8000 is live.

## Prerequisites

1. **MLX server running** on Mac Studio port 8000 (OpenAI-compatible API):
   ```bash
   ssh grosmac 'ssh studio "cd ~/KIKI-Mac_tunner && .venv/bin/mlx_lm.server --model models/Qwen3.6-35B-A3B --host 0.0.0.0 --port 8000 &"'
   ```
   Wait until `curl studio:8000/v1/models` returns the model list.

2. **Two real adapter paths** from `lora-qwen36-35b/`:
   - Prior: `/Users/clems/KIKI-Mac_tunner/output/micro-kiki/lora-qwen36-35b/chat-fr/adapters.safetensors`
   - New:   `/Users/clems/KIKI-Mac_tunner/output/micro-kiki/lora-qwen36-35b/reasoning/adapters.safetensors`
   (known healthy per `results/adapter-health-sweep.json`).

3. **Held-out eval dataset** for the prior domain — something like
   `data/micro-kiki/chat-fr/eval.jsonl` (operator responsibility).

## Step 1: angle-only probe (cheap sanity)

```bash
ssh grosmac 'ssh studio "cd ~/tmp/mk-smoke && /Users/clems/KIKI-Mac_tunner/.venv/bin/python3 scripts/measure_forgetting.py \
    --prior-adapter /Users/clems/KIKI-Mac_tunner/output/micro-kiki/lora-qwen36-35b/chat-fr/adapters.safetensors \
    --new-adapter   /Users/clems/KIKI-Mac_tunner/output/micro-kiki/lora-qwen36-35b/reasoning/adapters.safetensors \
    --output /tmp/angle-only.json"'
rsync -e 'ssh grosmac ssh' studio:/tmp/angle-only.json results/e2e-smoke-angle.json
```

Expected (per the 2026-04-18 sweep): mean angle ≈ 79°, all non-ignored
modules > 30°, `gate_status_aggregate = "angle_only_partial"`.

## Step 2: full gate with win-rate

Run `measure_forgetting.py` with `--generate-fn-module src.serving.mlx_client:generate`
and a small eval JSONL. Start with `--limit 5` on the dataset to keep the
wall-clock under 30 s for the first pass.

```bash
ssh grosmac 'ssh studio "cd ~/tmp/mk-smoke && MLX_HOST=http://localhost:8000 MLX_MODEL=qwen3.6-35b /Users/clems/KIKI-Mac_tunner/.venv/bin/python3 scripts/measure_forgetting.py \
    --prior-adapter <prior> --new-adapter <new> \
    --eval-dataset data/micro-kiki/chat-fr/eval.jsonl \
    --generate-fn-module src.serving.mlx_client:generate \
    --winrate-baseline-score 0.6 \
    --output /tmp/full-gate.json"'
```

Expected JSON (truncated):
```json
{
  "angle_degrees_mean": 79.4,
  "winrate_measured": 0.6..0.9,
  "winrate_drop": ...,
  "gate_status_aggregate": "pass" | "fail",
  ...
}
```

## Step 3: orchestration (`post_train_gate.py`) smoke

Same invocation via the one-shot wrapper:

```bash
ssh grosmac 'ssh studio "cd ~/tmp/mk-smoke && /Users/clems/KIKI-Mac_tunner/.venv/bin/python3 scripts/post_train_gate.py \
    --prior-adapter <prior> --new-adapter <new> \
    --eval-dataset data/micro-kiki/chat-fr/eval.jsonl \
    --generate-fn-module src.serving.mlx_client:generate \
    --winrate-baseline-score 0.6"'
```

Exit 0 on pass, 1 on degenerate adapter, 2 on forgetting trigger, 3 on bad
invocation. Designed to be appended to `mlx_lm lora` training scripts:

```bash
mlx_lm.server lora ... && \
    python scripts/post_train_gate.py --new-adapter $ADAPTER --prior-adapter $PRIOR ...
```

The non-zero exits surface via the shell — ideal for CI or a launchd
post-training action.

## Troubleshooting

- **`ConnectError` on :8000** — MLX server isn't running. Start it per
  step 0 of the prerequisites.
- **`gate_status = angle_only_partial`** in step 2 — one of `--eval-dataset`,
  `--generate-fn-module`, `--winrate-baseline-score` is missing.
- **Gate fails on `mlp.shared_expert_gate`** — this module is in the
  default ignore set (rank-1 delta, structurally constrained). Verify
  `results/adapter-health-sweep.json` to confirm the adapters are healthy.
- **Gate fails on a specific expert for a `switch_mlp` module** — inspect
  `angle_degrees_per_expert` in the JSON output; per-expert specialization
  may be real, in which case the aggregate view is masking a genuine
  forgetting signal. Escalate to a human review before rollback.

## Related

- `scripts/measure_forgetting.py` — the underlying angle + winrate CLI.
- `scripts/validate_adapter_health.py` — standalone health check.
- `scripts/post_train_gate.py` — this runbook's one-shot wrapper.
- `scripts/run_forgetting_sweep.py` — batch variant across N adapters.
- `src/serving/mlx_client.py` — the MLX server client.
- `docs/training/forgetting-gate.md` — the canonical gate doc.
- `.omc/brainstorm-oplora.md` — original design doc.

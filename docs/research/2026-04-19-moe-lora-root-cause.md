# Pre-pivot MoE-LoRA root cause — 2026-04-19

Forensic read of `scripts/micro_kiki/moe_lora.py` plus its training
harness to identify why every pre-pivot adapter had `lora_b` at
machine-zero (see `2026-04-19-prepivot-moe-lora-audit.md`).

## Summary

The pre-pivot `apply_moe_lora()` mounts each MoE-LoRA module **twice**
in the MLX parameter tree: once as a child inside `MoELoRALinear`
(in the forward path) and once as a sibling attribute `{target}_moe_lora`
on the parent submodule (out of the forward path). Gradients flow
only to the child copy; the state-dict filter (`"moe_lora" in name`)
and the on-disk key pattern that `src/serving/moe_lora_runtime.py`
expects at load time both target the sibling copy. The saved `lora_b`
is therefore always the initial-zero tensor, and the fix — archive,
not repair — was already taken by the 2026-04-16 pivot to standard
LoRA (204/204 `lora_b` non-zero on post-pivot `chat-fr`). This file
documents the forensic trail so future investigators do not have to
re-derive it.

## Forward pass analysis

`MoELoRALayer.__call__` (`scripts/micro_kiki/moe_lora.py:L136-L172`)
is structurally fine for autograd: it computes soft weights over all
experts (not a hard top-k), stacks per-expert outputs, and reduces
with a weighted sum. Every operation is differentiable:

```python
# scripts/micro_kiki/moe_lora.py:L155-L168
h = nn.gelu(self.router_w1(x))         # (..., router_hidden)
logits = self.router_w2(h)             # (..., num_experts)
weights = mx.softmax(logits, axis=-1)  # (..., num_experts)

expert_outputs = mx.stack(
    [expert(x) for expert in self.experts], axis=-2
)  # (..., num_experts, out_features)

w = mx.expand_dims(weights, axis=-1)
delta = mx.sum(expert_outputs * w, axis=-2)  # (..., out_features)
```

Each `expert(x)` at `L45-L52` is `(x @ A) @ B * scale`. With `B = 0`
at init (`L43`) every `expert_outputs` entry is zero on step 1 — but
the backward graph is still intact: `∂delta/∂B_i = w_i · (x @ A_i)^T`
is non-zero once the router produces any non-uniform weighting.

So the single-module forward/backward math is *not* the bug.

## Backward pass / mounting analysis

The bug is in the **mount topology** chosen by `apply_moe_lora`:

```python
# scripts/micro_kiki/moe_lora.py:L255-L273
wrapped = MoELoRALinear(linear, moe)
setattr(sub_module, target, wrapped)              # <-- child mount
# Also store sibling ref for null-space extraction
attr_name = f"{target}_moe_lora"
setattr(sub_module, attr_name, moe)               # <-- sibling mount
```

`moe` is the *same Python object* in both mounts, but MLX tree-flatten
walks `nn.Module` attributes by name, so each mount produces a
distinct parameter path:

- child (in forward path, inside `MoELoRALinear.__call__` at `L200-L204`):
  `...layers.L.sub.target.moe_lora.experts.E.lora_b`
- sibling (out of forward path, kept "for null-space extraction"):
  `...layers.L.sub.target_moe_lora.experts.E.lora_b`

### Why gradients never reach the sibling's `lora_b`

`scripts/micro_kiki/train_stack.py:L285-L293` unfreezes by name
substring match (`"moe_lora" in name`), which hits *both* mounts,
and the trainer's parameter collection at `L87-L105`
(`extract_moe_lora_state_dict`, filter `"moe_lora" in name`) likewise
pulls from both. But the **forward pass only ever evaluates the
child** (the sibling is just a Python reference stashed for the
null-space helper at `scripts/micro_kiki/null_space.py:L148-L175`).
Autograd therefore populates `∂L/∂B` on the *child* path only;
the sibling path's `B` parameter participates in no graph node and
its gradient is `None` / zero.

The optimizer step in MLX walks the tree and applies updates by
path. The sibling `lora_b` is seen as a trainable parameter (it was
unfrozen and filtered in), receives a zero update every step, and
stays at its initial `mx.zeros((rank, out_features))` value
(`L43`).

### Why the saved file contains the sibling, not the child

`extract_moe_lora_state_dict` at `scripts/micro_kiki/train_stack.py:L87-L105`
iterates `tree_flatten(model.parameters())` and keeps every key
containing the substring `moe_lora`. Both paths match, so both end
up in the saved `.safetensors`. At load time,
`src/serving/moe_lora_runtime.py:L15-L17,L312-L351` *only* recognises
the sibling pattern (`...{proj}_moe_lora.experts.E.lora_b`) — see
the literal `f"{sub}.{proj}_moe_lora"` match at `L347`. The child
path `...{proj}.moe_lora...` is silently ignored.

Net effect: the child copy trains correctly but is discarded at
serving time; the sibling copy is served but was never in the
forward graph. The symptom observed in the 2026-04-19 audit
(`lora_b` norm = 0 on 35/35 adapters) falls directly out of this.

## Specific line references

| Location | Role |
|----------|------|
| `scripts/micro_kiki/moe_lora.py:L43` | `lora_b = mx.zeros((rank, out_features))` init |
| `scripts/micro_kiki/moe_lora.py:L45-L52` | Expert forward `(x @ A) @ B * scale` |
| `scripts/micro_kiki/moe_lora.py:L136-L172` | `MoELoRALayer.__call__` — soft-weighted sum (math is fine) |
| `scripts/micro_kiki/moe_lora.py:L196-L204` | `MoELoRALinear.__call__` — wraps base + child moe_lora |
| `scripts/micro_kiki/moe_lora.py:L255-L273` | **Root cause** — dual mount (child + sibling) |
| `scripts/micro_kiki/train_stack.py:L87-L105` | `extract_moe_lora_state_dict` saves *both* paths |
| `scripts/micro_kiki/train_stack.py:L285-L293` | Unfreeze-by-name-substring hits *both* paths |
| `src/serving/moe_lora_runtime.py:L15-L17,L347` | Load-time matches only the sibling `{proj}_moe_lora` key |

## Diagnosis (definitive)

`∂L/∂B_sibling = 0` because the sibling module is not reachable from
any `loss(model(x))` evaluation — no path in the forward graph
touches it. `∂L/∂B_child > 0` and the child trains, but its
parameters are serialized under a key pattern
(`...{target}.moe_lora.experts.N.lora_b`) that the MLX inference
runtime discards. What the runtime loads is the sibling's persistent
zero tensor, and all 35 pre-pivot adapters inherit that zero from the
`mx.zeros` initializer at `L43`.

## Empirical validation plan (not executed)

Should anyone ever need to confirm this instead of taking the
post-hoc analysis on faith:

1. Load any pre-pivot `stacks-v3-r16/*/adapters.safetensors` and
   confirm *both* key families are present:
   `...{proj}.moe_lora.experts.N.{lora_a,lora_b}` and
   `...{proj}_moe_lora.experts.N.{lora_a,lora_b}`. The child family
   should have non-zero `lora_b`; the sibling family should not.
2. Minimal repro with `mlx.core` + `mlx.nn`: build a two-layer toy
   with one `Linear` wrapped by `MoELoRALinear` + sibling mount,
   train one step on a constant target, then inspect
   `tree_flatten(model.parameters())`. The child `lora_b` will have
   moved off zero; the sibling `lora_b` will still be zero.
3. MLX gradient probe: `mx.grad(loss_fn)(params)` and grep the
   returned tree — sibling `lora_b` entries will be zero arrays,
   child entries non-zero.

None of this is worth doing: the post-pivot path already serves
answers (see Decision).

## Decision: ARCHIVE

Fixing would require either (a) dropping the sibling mount and
teaching `src/serving/moe_lora_runtime.py` + `null_space.py` to
consume the child path, or (b) hoisting the child out and keeping
only the sibling in the forward graph (rewriting `MoELoRALinear`).
Both rewrites touch a code path that was obsoleted by the 2026-04-16
pivot to standard LoRA on Qwen3.5-35B-A3B. Post-pivot adapters
produced by `scripts/train_niches_mlxtune.py` + `python -m mlx_lm lora`
store 204/204 non-zero `lora_b` on `chat-fr` (empirical sanity
check from the same audit), so the architecture that needed the
pre-pivot MoE-LoRA path no longer exists.

Archiving is therefore the correct action. Post-pivot guardrail is
already in place via `scripts/validate_adapter_health.py`
(added same day), which fails CI / training runs if any produced
adapter regresses to the pre-pivot `all-lora_b-zero` state.

## What the archive means

- `scripts/micro_kiki/moe_lora.py` moves to `scripts/legacy/moe_lora.py`
  via `git mv` (history preserved).
- The ~35 pre-pivot adapters in
  `/Users/clems/KIKI-Mac_tunner/output/micro-kiki/stacks-v3-r16/`
  remain on disk but are documented as **known-unusable** (audit
  result: `A @ B = A @ 0 = 0` → zero delta contribution).
- Intra-package callers (`scripts/micro_kiki/train_stack.py`,
  `eval_stack.py`, `residual_boost.py`, `null_space.py`) are updated
  to import from `scripts.legacy.moe_lora` so the package imports
  still resolve for anyone running the archived pipeline
  end-to-end against the archived artifacts. No new runs should
  produce new pre-pivot adapters.
- The external caller `scripts/eval_v2_v3.py` (MLXBackend) is
  updated to locate the module at its new `scripts/legacy/` path;
  it remains opt-in and is only exercised against archived
  adapters.
- `scripts/micro_kiki/AGENTS.md` and `scripts/legacy/AGENTS.md` are
  updated to point at this document as the canonical explanation.
- No test changes: `tests/test_moe_lora.py` targets
  `src.stacks.moe_lora` (a legitimate post-pivot stub, different
  file — see `src/stacks/AGENTS.md:L27`), and
  `tests/test_moe_lora_runtime.py` targets
  `src.serving.moe_lora_runtime` (the inference loader, also a
  different file).

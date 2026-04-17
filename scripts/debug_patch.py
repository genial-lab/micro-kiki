#!/usr/bin/env python3
"""Debug: check lora_b weights across all projections."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
from src.serving.moe_lora_runtime import MoELoRAConfig, load_adapter_projections

config = MoELoRAConfig()
projections = load_adapter_projections(
    "output/micro-kiki/stacks-v3-r16/python", config
)

# Count zero vs non-zero lora_b
zero_b = 0
nonzero_b = 0
zero_a = 0
nonzero_a = 0

for key, proj in projections.items():
    for i in range(4):
        a_max = mx.max(mx.abs(proj.lora_a[i])).item()
        b_max = mx.max(mx.abs(proj.lora_b[i])).item()
        if a_max == 0:
            zero_a += 1
        else:
            nonzero_a += 1
        if b_max == 0:
            zero_b += 1
        else:
            nonzero_b += 1

print(f"Total projections: {len(projections)}")
print(f"lora_a: {nonzero_a} non-zero, {zero_a} zero")
print(f"lora_b: {nonzero_b} non-zero, {zero_b} zero")

# Also check the raw safetensors to see if lora_b was saved correctly
d = mx.load("output/micro-kiki/stacks-v3-r16/python/adapters.safetensors")
b_keys = [k for k in sorted(d.keys()) if k.endswith(".lora_b")]
print(f"\nRaw safetensors lora_b keys: {len(b_keys)}")
nonzero_raw = 0
for k in b_keys[:10]:
    v = d[k]
    mx.eval(v)
    m = mx.max(mx.abs(v)).item()
    print(f"  {k}: shape={v.shape} max={m:.8f}")
    if m > 0:
        nonzero_raw += 1

# Check if ALL lora_b are zero
all_zero = True
for k in b_keys:
    v = d[k]
    mx.eval(v)
    if mx.max(mx.abs(v)).item() > 0:
        all_zero = False
        print(f"  FOUND non-zero: {k}")
        break

if all_zero:
    print(f"\nALL {len(b_keys)} lora_b tensors are zero!")
    print("This means the adapter was NOT trained (B initialized to 0, never updated)")

# Also check embedded domain
print("\n--- Checking embedded domain ---")
d2 = mx.load("output/micro-kiki/stacks-v3-r16/embedded/adapters.safetensors")
b_keys2 = [k for k in sorted(d2.keys()) if k.endswith(".lora_b")]
for k in b_keys2[:5]:
    v = d2[k]
    mx.eval(v)
    m = mx.max(mx.abs(v)).item()
    print(f"  {k}: max={m:.8f}")

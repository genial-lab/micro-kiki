# SpikingKiki v0.3 Cookbook

Three worked examples for the v0.3 neuroscience branch.

## Example A: AeonSleep Standalone

Write and recall episodic memories using the Atlas SIMD vector index.

```python
from src.memory.atlas import AtlasIndex

# Create a 64-d index
index = AtlasIndex(dim=64)

# Add memories
index.add("morning-coffee", [0.1] * 64, payload={"mood": "alert"})
index.add("afternoon-nap",  [0.9] * 64, payload={"mood": "sleepy"})
index.add("evening-code",   [0.5] * 64, payload={"mood": "focused"})

# Recall similar memories
query = [0.5] * 64
hits = index.search(query, k=2)
for hit in hits:
    print(f"{hit.key}: score={hit.score:.3f}, payload={hit.payload}")
# => evening-code: score=1.000, payload={'mood': 'focused'}
# => morning-coffee: score=0.xxx, payload={'mood': 'alert'}
```

**Key points:**
- Vectors are unit-normalised internally (cosine similarity).
- `use_numpy=False` forces the pure-Python fallback (no deps).
- The `stats()` method returns index metadata for monitoring.

## Example B: SNN CPU Inference

Convert a small MLP to a spiking equivalent and run inference
on CPU using only numpy.

```python
import numpy as np
from src.spiking.las_converter import LASConverter

# Build a 2-layer ANN
rng = np.random.default_rng(0)
layers = [
    {"weight": rng.standard_normal((32, 16)) * 0.2, "bias": np.zeros(32)},
    {"weight": rng.standard_normal((8, 32)) * 0.2,  "bias": np.zeros(8)},
]

# Convert to SNN (T=64 timesteps for low quantisation error)
converter = LASConverter(timesteps=64, max_rate=1.0)
snn = converter.convert_model(layers)

# Run inference
x = np.clip(rng.standard_normal((4, 16)) * 0.3, -0.5, 0.5)
output = snn(x)
print(f"Output shape: {output.shape}")  # (4, 8)
print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

# Verify against ANN
def ann_forward(x):
    h = np.maximum(x @ layers[0]["weight"].T + layers[0]["bias"], 0)
    return np.maximum(h @ layers[1]["weight"].T + layers[1]["bias"], 0)

ann_out = ann_forward(x)
rel_err = np.linalg.norm(output - ann_out) / (np.linalg.norm(ann_out) + 1e-12)
print(f"Relative error: {rel_err:.4f}")  # < 0.05
```

**Key points:**
- Higher `timesteps` = lower error but slower inference.
- `max_rate` clips activations; set it to match your model's range.
- The converter is numpy-only: no torch needed for inference.

## Example C: Akida Routing (MoE-aware SNN)

Convert a Mixture-of-Experts layer preserving expert selection.

```python
import numpy as np
from src.spiking.las_converter import LASConverter

rng = np.random.default_rng(42)
dim, num_experts, out_dim = 128, 4, 64

# Build router + experts
router = {
    "weight": rng.standard_normal((num_experts, dim)) * 0.1,
    "bias": np.zeros(num_experts),
}
experts = [
    {"weight": rng.standard_normal((out_dim, dim)) * 0.1,
     "bias": np.zeros(out_dim)}
    for _ in range(num_experts)
]

# Convert with MoE-aware method
converter = LASConverter(timesteps=256, max_rate=1.0)
snn_moe = converter.convert_moe_layer(
    router=router, experts=experts, top_k=2
)

# Run inference
x = rng.standard_normal((8, dim)) * 0.3
output = snn_moe(x)
print(f"Output shape: {output.shape}")  # (8, 64)

# Check which experts were selected
indices = snn_moe.selected_experts(x)
print(f"Expert selections (first 3 samples):")
for i in range(3):
    print(f"  sample {i}: experts {indices[i].tolist()}")

# Energy comparison
from scripts.energy_bench import compute_energy
result = compute_energy(
    model_params=dim * out_dim * num_experts,
    spike_rate=0.3, timesteps=4,
)
print(f"Energy saving: {result.snn_energy_saving_pct:.1f}%")
```

**Key points:**
- Router uses ANN logits for selection (preserves ordering).
- Experts use spiking forward (energy efficient).
- Top-k softmax combination weights from router logits.
- On Akida hardware, the spiking experts map to neuromorphic cores.

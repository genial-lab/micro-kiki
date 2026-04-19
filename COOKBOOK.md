# micro-kiki Cookbook

Seven runnable examples — five end-user inference flows and two operator gates (forgetting + CI-local). Base model: Qwen3.6-35B-A3B; adapters cover 17 module kinds per layer with MLX `scale = 20.0`.

## 1. Basic Chat (chat-fr)

French conversational AI with natural tone and cultural awareness.

```python
from src.routing.router import DomainRouter
from src.serving.mlx_server import MLXServer

server = MLXServer(model_path="models/qwen3.6-35b-a3b", quantization="Q4_K_M")
router = DomainRouter()

prompt = "Explique-moi le fonctionnement d'un condensateur comme si j'avais 10 ans."
domain = router.classify(prompt)  # -> "chat-fr"

response = server.generate(prompt, adapters=[domain])
print(response)
```

**Expected output:**
```
Imagine un condensateur comme un petit réservoir d'électricité. Quand tu branches
une pile, l'électricité remplit le réservoir. Quand tu débranches la pile, le
réservoir peut redonner l'électricité qu'il a stockée...
```

## 2. Code Generation (python)

Python code generation with type hints and idiomatic patterns.

```python
from src.routing.router import DomainRouter
from src.serving.mlx_server import MLXServer

server = MLXServer(model_path="models/qwen3.6-35b-a3b", quantization="Q4_K_M")
router = DomainRouter()

prompt = """Write a Python async context manager that limits concurrent API calls
to a configurable maximum, with timeout support."""

domain = router.classify(prompt)  # -> "python"
response = server.generate(prompt, adapters=[domain])
print(response)
```

**Expected output:**
```python
import asyncio
from contextlib import asynccontextmanager

class ConcurrencyLimiter:
    def __init__(self, max_concurrent: int = 10, timeout: float = 30.0) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._timeout = timeout

    @asynccontextmanager
    async def limit(self):
        async with asyncio.timeout(self._timeout):
            async with self._semaphore:
                yield
```

## 3. Electronics (kicad-dsl)

KiCad schematic DSL generation for PCB design.

```python
from src.routing.router import DomainRouter
from src.serving.mlx_server import MLXServer

server = MLXServer(model_path="models/qwen3.6-35b-a3b", quantization="Q4_K_M")
router = DomainRouter()

prompt = """Generate a KiCad schematic snippet for an ESP32-S3 with:
- USB-C connector (CC resistors, ESD protection)
- 3.3V LDO (AMS1117-3.3)
- Decoupling capacitors (100nF + 10uF)"""

domain = router.classify(prompt)  # -> "kicad-dsl"
response = server.generate(prompt, adapters=[domain])
print(response)
```

**Expected output:**
```
(kicad_sch
  (symbol (lib_id "Connector:USB_C_Receptacle") (at 25 50 0)
    (property "Reference" "J1") ...)
  (symbol (lib_id "Regulator_Linear:AMS1117-3.3") (at 75 50 0)
    (property "Reference" "U2") ...)
  (wire (pts (xy 40 50) (xy 60 50)))  ; VBUS to LDO input
  ...)
```

## 4. Multi-Domain Routing

Automatic domain detection and multi-stack routing for complex queries.

```python
from src.routing.router import DomainRouter
from src.routing.dispatcher import Dispatcher
from src.serving.mlx_server import MLXServer

server = MLXServer(model_path="models/qwen3.6-35b-a3b", quantization="Q4_K_M")
router = DomainRouter()
dispatcher = Dispatcher()

prompt = """I need a Python script that reads INA226 current sensor data over I2C,
computes a rolling average, and stores results in a SQLite database.
Include proper error handling for I2C communication failures."""

# Router detects multiple domains
domains = router.classify_multi(prompt, top_k=4)
# -> ["python", "firmware", "electronics", "sql"]

# Dispatcher selects meta-intent
intent = dispatcher.dispatch(domains, prompt)
# -> "code_generation" with stacks ["python", "firmware"]

response = server.generate(prompt, adapters=intent.stacks)
print(f"Domains: {domains}")
print(f"Intent: {intent.name}, Stacks: {intent.stacks}")
print(response)
```

**Expected output:**
```
Domains: ['python', 'firmware', 'electronics', 'sql']
Intent: code_generation, Stacks: ['python', 'firmware']

import smbus2
import sqlite3
from collections import deque
...
```

## 5. Memory-Augmented Conversation (Aeon)

Long-term memory with episodic recall for context-aware conversations.

```python
from datetime import datetime

from src.memory.aeon import AeonPalace
from src.routing.router import DomainRouter
from src.serving.mlx_server import MLXServer

server = MLXServer(model_path="models/qwen3.6-35b-a3b", quantization="Q4_K_M")
router = DomainRouter()
palace = AeonPalace(dim=3072)

# Write conversation episodes to memory
palace.write(
    content="User asked about ESP32 deep sleep current consumption. "
            "Answered: ~10uA with ULP, ~150uA with timer wakeup.",
    domain="electronics",
    timestamp=datetime(2026, 4, 1, 10, 0),
)
palace.write(
    content="User designed a battery monitor with INA226. "
            "Recommended 0.01 ohm shunt for 3A max.",
    domain="electronics",
    timestamp=datetime(2026, 4, 2, 14, 30),
    links=[],  # could link to previous episode
)

# Later: recall relevant memory for new query
query = "What shunt resistor should I use for my battery project?"
memories = palace.recall(query, top_k=3, domain="electronics")

# Build context-augmented prompt
context = "\n".join(f"[Memory] {m.content}" for m in memories)
augmented_prompt = f"{context}\n\nUser: {query}"

domain = router.classify(query)
response = server.generate(augmented_prompt, adapters=[domain])
print(f"Retrieved {len(memories)} memories")
print(response)
```

**Expected output:**
```
Retrieved 2 memories

Based on our previous discussion about your battery monitor with the INA226,
I recommended a 0.01 ohm shunt resistor for up to 3A. If your requirements
have changed, here are the considerations:
- 0.01 ohm: good for 1-5A range, ~30mV drop at 3A
- 0.1 ohm: better resolution for <500mA, but higher power loss
...
```

## 6. Forgetting gate (operator — after training a new stack)

Chain health-check, angle measurement, and win-rate evaluation so a freshly-trained
LoRA adapter cannot enter the curriculum if it would damage a prior stack.

```bash
# (1) Health: all lora_B matrices non-zero (catches the pre-pivot MoE-LoRA bug).
python scripts/validate_adapter_health.py \
    output/stacks/stack-04-rust/adapter_model.safetensors

# (2) Pairwise per-module angle vs. the immediately prior stack (informational).
python scripts/measure_forgetting.py \
    --prior-adapter output/stacks/stack-03-cpp/adapter_model.safetensors \
    --new-adapter   output/stacks/stack-04-rust/adapter_model.safetensors \
    --output        results/forgetting-stack04-vs-stack03.json

# (3) Or the one-shot orchestrator — health + angle + win-rate in one call.
# Exit codes: 0 PASS, 1 angle-fail, 2 winrate-fail, 3 health-fail.
python scripts/post_train_gate.py \
    output/stacks/stack-04-rust/ \
    --prior-dir output/stacks/ \
    --output    results/gate-stack04.json
```

Bulk sweeps:

```bash
# Every pair in a directory (matrix dump).
python scripts/run_forgetting_sweep.py output/stacks/ \
    --output results/forgetting-matrix.json

# Every adapter's lora_B health (catches dead-weight adapters in bulk).
python scripts/sweep_adapter_health.py output/stacks/ \
    --output results/adapter-health-sweep.json
```

Empirical baseline: `results/smoke-gate.json` (chat-fr ↔ reasoning, mean 79.4°,
winrate_drop −0.04, gate PASS). Canonical gate doc:
`docs/training/forgetting-gate.md`. Dual-server real-adapter runbook:
`docs/training/e2e-smoke-runbook.md`.

## 7. Run CI gates locally (before pushing)

Reproduce both CI jobs (`.github/workflows/validators.yml`) on your laptop.

```bash
# Four config invariants — fast, no deps beyond pyyaml/safetensors/numpy.
python scripts/validate_domains.py            # 34-domain list across 3 mirrors
python scripts/validate_rank_schema.py        # rank ∈ {4,8,12,16,32}, alpha = 2·rank
python scripts/validate_curriculum_order.py   # foundations before niches
python scripts/validate_no_pre_pivot.py       # no Qwen3.5-4B leaks in src/

# Validator unit tests (mirrors the CI `config-invariants` job).
python -m pytest tests/test_validate_*.py -q

# Forgetting tests (mirrors the CI `forgetting-tests` job; needs CPU torch).
python -m pytest tests/scripts/test_measure_forgetting.py \
                 tests/test_forgetting_check_all_previous.py -v
```

Any of the four validators failing is a blocking CI failure. If you add a
34th-list or rank-tier change, update all three config mirrors in the same
commit — `validate_domains.py` will reject partial updates.

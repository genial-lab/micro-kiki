# micro-kiki Cookbook

Five runnable examples demonstrating micro-kiki capabilities.

## 1. Basic Chat (chat-fr)

French conversational AI with natural tone and cultural awareness.

```python
from src.routing.router import DomainRouter
from src.serving.mlx_server import MLXServer

server = MLXServer(model_path="models/qwen3.5-35b-a3b", quantization="Q4_K_M")
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

server = MLXServer(model_path="models/qwen3.5-35b-a3b", quantization="Q4_K_M")
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

server = MLXServer(model_path="models/qwen3.5-35b-a3b", quantization="Q4_K_M")
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

server = MLXServer(model_path="models/qwen3.5-35b-a3b", quantization="Q4_K_M")
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

server = MLXServer(model_path="models/qwen3.5-35b-a3b", quantization="Q4_K_M")
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

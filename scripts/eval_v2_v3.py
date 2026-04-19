#!/usr/bin/env python3
"""Benchmark V2 vs V3 trained MoE-LoRA stacks on Qwen3.5-4B.

Compares per-domain adapters across two stack generations:
  - V2: 32 domains, no null-space
  - V3: 35 domains, null-space enabled

Metrics: perplexity, domain keyword hit rate, response length, forgetting.

Usage:
    uv run python scripts/eval_v2_v3.py \\
        --v2-dir /path/to/stacks-v2 \\
        --v3-dir /path/to/stacks \\
        --base-model /path/to/qwen3.5-4b

    # Evaluate specific domains only:
    uv run python scripts/eval_v2_v3.py \\
        --v2-dir ./stacks-v2 --v3-dir ./stacks \\
        --base-model ./models/qwen3.5-4b \\
        --domains chat-fr python embedded

    # Dry run (no model loading, uses stubs):
    uv run python scripts/eval_v2_v3.py \\
        --v2-dir ./stacks-v2 --v3-dir ./stacks \\
        --base-model ./models/qwen3.5-4b \\
        --dry-run

    # With cross-eval forgetting matrix:
    uv run python scripts/eval_v2_v3.py \\
        --v2-dir ./stacks-v2 --v3-dir ./stacks \\
        --base-model ./models/qwen3.5-4b \\
        --cross-eval
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Domain test prompts — 5+ per key domain
# ---------------------------------------------------------------------------

DOMAIN_PROMPTS: dict[str, list[str]] = {
    "chat-fr": [
        "Explique de manière accessible la différence entre l'intelligence "
        "artificielle générative et discriminative, avec un exemple concret.",
        "Rédige un email professionnel pour demander un report de réunion "
        "en préservant un ton courtois et en proposant deux créneaux.",
        "Comment expliquer le concept de 'prompt engineering' à un chef de "
        "projet non-technique ? Donne une analogie claire.",
        "Traduis et adapte culturellement cette phrase anglaise en français "
        "professionnel : 'Let's circle back on this offline and touch base EOD.'",
        "Résume en 3 points clés le principe du consensus de Nakamoto dans "
        "Bitcoin, en termes simples pour un public non-informaticien.",
    ],
    "reasoning": [
        "A snail climbs a 10-meter pole. Each day it climbs 3 meters, each "
        "night it slips back 2 meters. On which day does it reach the top? "
        "Show your reasoning step by step.",
        "Given: All A are B. Some B are C. No C are D. Can we conclude that "
        "some A are not D? Construct a validity proof or counterexample.",
        "You have 12 coins, one is counterfeit (lighter or heavier). With "
        "exactly 3 weighings on a balance scale, identify the counterfeit and "
        "whether it is lighter or heavier.",
        "A factory produces widgets. Machine A: 60% output, 2% defect. "
        "Machine B: 40% output, 5% defect. A widget is defective — what is "
        "the probability it came from Machine B? (Bayes theorem)",
        "Prove by induction that the sum of the first n odd numbers equals n². "
        "State the base case, inductive hypothesis, and inductive step.",
    ],
    "python": [
        "Write a Python context manager that wraps a sqlite3 connection: "
        "commits on success, rolls back on exception, and closes the connection.",
        "Implement a generic LRUCache class using collections.OrderedDict with "
        "get(key) and put(key, value) in O(1) time.",
        "Write a decorator @retry(max_attempts=3, delay=1.0) that retries a "
        "function with exponential backoff on specified exceptions.",
        "Using asyncio, write a coroutine that fetches 10 URLs concurrently "
        "with aiohttp, collects results, and returns them sorted by response time.",
        "Explain Python's GIL: why does it exist, when does it NOT prevent "
        "parallelism, and how does multiprocessing work around it?",
    ],
    "embedded": [
        "Write an interrupt-safe ring buffer in C for bare-metal ARM Cortex-M, "
        "ensuring head/tail pointer updates are atomic using critical sections.",
        "Explain the startup sequence of a bare-metal ARM Cortex-M4: from reset "
        "vector through stack pointer load, .data copy, .bss zero-fill, to main().",
        "A Cortex-M hard fault handler must preserve crash context. Write the "
        "assembly stub that saves registers and calls a C fault logger.",
        "Compare I2C, SPI, and UART for a sensor hub: arbitration, clock "
        "stretching, full-duplex, typical throughput on a 72 MHz MCU.",
        "Write a linker script fragment for STM32F4 placing .fastcode in CCMRAM "
        "at 0x10000000 and .data in SRAM at 0x20000000.",
    ],
    "kicad-dsl": [
        "Write a KiCad schematic symbol DSL definition for a 16-pin STM32F103C8T6 "
        "with pin types (power, I/O, bidirectional) and multi-unit split.",
        "In KiCad 8 footprint scripting (Python API), write a function to create "
        "a THT resistor footprint from pitch, pad_size, and drill parameters.",
        "Explain B.Cu, In1.Cu, and F.Mask layer meanings in a KiCad PCB file "
        "(.kicad_pcb) and when each is referenced in an s-expression.",
        "Given a KiCad netlist in IPC-D-356A format, write a parser extracting "
        "net names, component references, and pin numbers into a Python dict.",
        "Describe the (net_tie_pad_groups ...) directive in a KiCad footprint: "
        "what does it do, when is it needed? Provide an example for a 0-Ohm jumper.",
    ],
    "spice-sim": [
        "In a SPICE simulation of a buck converter at 400 kHz, write the .TRAN "
        "and .MEASURE commands to extract Vout_avg, Vripple_pp, and efficiency.",
        "How do you model parasitic inductance of a PCB trace in SPICE? Write a "
        "subcircuit for 10 nH trace inductance with 50 mOhm series resistance.",
        "Explain Monte Carlo analysis in ngspice: write a .MC command block "
        "running 100 trials varying R1 +/-5% and C1 +/-10% with normal distribution.",
        "A simulation shows ringing at 50 MHz on a gate drive. What SPICE "
        "parameters would you vary to damp it, and how would you verify with .MEAS?",
        "Write a Verilog-A model (ngspice-compatible) for a temperature-dependent "
        "resistor with TCR1=3000 ppm/K and TCR2=0.5 ppm/K².",
    ],
    "electronics": [
        "A common-emitter BJT amplifier: Vcc=12V, RC=4.7kOhm, RE=1kOhm, beta=100. "
        "Calculate the Q-point (ICQ, VCEQ) and small-signal voltage gain.",
        "Explain the Miller effect in a MOSFET amplifier: how does Cgd create an "
        "effective input capacitance Miller_C = Cgd*(1+Av)?",
        "An RC charge pump doubles 3.3V to 6.6V at 1 mA. Calculate the flying "
        "capacitor value, output ripple, and efficiency loss at 1 MHz.",
        "Compare NMOS vs PMOS for high-side switching: gate drive requirements, "
        "body diode direction, Rds(on), and why NMOS with bootstrap is preferred.",
        "A Wheatstone bridge uses four 10kOhm resistors, one changes to 10.1kOhm. "
        "Calculate the differential output for Vexcitation=5V.",
    ],
    "components": [
        "Compare LDO vs switching regulator for 5V to 3.3V at 500mA: efficiency, "
        "noise, cost, PCB area, and when to choose each.",
        "Explain the key parameters of a MOSFET datasheet: Vgs(th), Rds(on), Qg, "
        "Ciss, and how each affects switching speed and losses.",
        "Select a TVS diode for USB-C ESD protection: working voltage, clamping "
        "voltage, response time, and junction capacitance requirements.",
        "What are the differences between X5R, X7R, C0G/NP0 MLCC ceramic "
        "capacitors? When would you choose each for a power supply design?",
        "Design a current sense circuit using a 10 mOhm shunt resistor and INA219: "
        "calculate the ADC resolution, max current, and power dissipation.",
    ],
    "shell": [
        "Write a bash script that monitors a directory for new .log files using "
        "inotifywait and sends an alert if 'ERROR' appears.",
        "Explain set -e, set -u, set -o pipefail, and set -x in bash. Write a "
        "script header that enables safe defaults.",
        "Write a POSIX-compatible shell function retry(n, delay, cmd) that "
        "re-runs cmd up to n times with delay between attempts.",
        "In bash, how do you process a CSV file with quoted fields containing "
        "commas? Write a solution using awk that handles edge cases.",
        "Write a shell pipeline that counts the 10 most frequent words in a text "
        "file, excluding stop words from a list file, sorted by frequency.",
    ],
    "docker": [
        "Write a multi-stage Dockerfile for a Python FastAPI app: build stage "
        "with uv, final stage python:3.12-slim, non-root user, health check.",
        "Explain Docker overlay2 storage driver: layer stacking, whiteout files, "
        "and what docker history shows for intermediate layer sizes.",
        "Write a Docker Compose file for FastAPI + PostgreSQL + Redis with named "
        "volumes, health checks, and a custom network 172.20.0.0/24.",
        "What is the difference between CMD and ENTRYPOINT in a Dockerfile? "
        "Give an example where ENTRYPOINT + CMD enables default + override.",
        "Explain Docker BuildKit cache mounts (--mount=type=cache) vs layer "
        "caching. Write a Dockerfile using cache mounts for pip/uv install.",
    ],
}

# ---------------------------------------------------------------------------
# Domain keywords for keyword hit rate scoring
# ---------------------------------------------------------------------------

DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "chat-fr": [
        "bonjour", "merci", "cordialement", "expliquer", "résumé",
        "exemple", "professionnel", "courtois", "différence", "analogie",
    ],
    "reasoning": [
        "therefore", "proof", "hypothesis", "induction", "base case",
        "probability", "bayes", "conclude", "counterexample", "step",
    ],
    "python": [
        "def", "class", "import", "async", "await", "decorator",
        "context manager", "yield", "exception", "return",
    ],
    "embedded": [
        "cortex", "interrupt", "register", "dma", "gpio", "uart",
        "spi", "i2c", "linker", "volatile", "irq", "arm",
    ],
    "kicad-dsl": [
        "kicad", "footprint", "schematic", "symbol", "layer",
        "pad", "net", "pcb", "s-expression", "module",
    ],
    "spice-sim": [
        "spice", "ngspice", "tran", "measure", "netlist", "subcircuit",
        "simulation", "monte carlo", "voltage", "current",
    ],
    "electronics": [
        "resistor", "capacitor", "mosfet", "bjt", "amplifier",
        "gain", "voltage", "current", "impedance", "transistor",
    ],
    "components": [
        "ldo", "regulator", "mosfet", "capacitor", "diode",
        "datasheet", "esd", "tvs", "mlcc", "shunt",
    ],
    "shell": [
        "bash", "script", "pipe", "awk", "grep", "set -e",
        "posix", "exit", "function", "variable",
    ],
    "docker": [
        "dockerfile", "container", "image", "volume", "compose",
        "layer", "build", "stage", "entrypoint", "health",
    ],
}

# ---------------------------------------------------------------------------
# Adapter backend interface
# ---------------------------------------------------------------------------


class AdapterBackend(ABC):
    """Abstract interface for loading adapters and running inference.

    Implement a concrete subclass for your platform (MLX, vLLM, HF, etc.).
    """

    @abstractmethod
    def load_base_model(self, model_path: str) -> None:
        """Load the base model (Qwen3.5-4B)."""

    @abstractmethod
    def load_adapter(self, adapter_dir: str) -> str:
        """Load a LoRA adapter from a directory containing adapters.safetensors.

        Returns an adapter handle/ID string.
        """

    @abstractmethod
    def unload_adapter(self, adapter_id: str) -> None:
        """Unload a previously loaded adapter to free memory."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        adapter_id: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """Generate a response, optionally with a loaded adapter active."""

    @abstractmethod
    def compute_perplexity(
        self,
        text: str,
        adapter_id: str | None = None,
    ) -> float:
        """Compute perplexity of a text under the (optionally adapted) model.

        Returns a float >= 1.0. Lower is better.
        """


class StubBackend(AdapterBackend):
    """Stub backend for dry-run / CI testing. No real model loading."""

    def __init__(self) -> None:
        self._adapters: dict[str, Path] = {}
        self._counter = 0

    def load_base_model(self, model_path: str) -> None:
        logger.info("[STUB] load_base_model(%s)", model_path)

    def load_adapter(self, adapter_dir: str) -> str:
        self._counter += 1
        aid = f"stub-adapter-{self._counter}"
        self._adapters[aid] = Path(adapter_dir)
        logger.debug("[STUB] load_adapter(%s) -> %s", adapter_dir, aid)
        return aid

    def unload_adapter(self, adapter_id: str) -> None:
        self._adapters.pop(adapter_id, None)

    def generate(
        self,
        prompt: str,
        adapter_id: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        adapter_name = ""
        if adapter_id and adapter_id in self._adapters:
            adapter_name = self._adapters[adapter_id].name
        # Produce a plausible-length stub response
        return (
            f"[STUB response | adapter={adapter_name}] "
            f"This is a placeholder response for the given prompt. "
            f"The model would produce domain-relevant content here. "
            f"Prompt snippet: {prompt[:60]}..."
        )

    def compute_perplexity(
        self,
        text: str,
        adapter_id: str | None = None,
    ) -> float:
        # Return a plausible perplexity in [3.0, 15.0] range
        import hashlib
        h = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        base_ppl = 5.0 + (h % 1000) / 100.0
        # Adapter slightly lowers perplexity in stub mode
        if adapter_id:
            base_ppl *= 0.85
        return round(base_ppl, 2)


class SafetensorsBackend(AdapterBackend):
    """Backend that loads adapters.safetensors via the safetensors library.

    This provides the adapter weight loading. You must subclass or extend
    this to wire in actual model inference for your platform.

    For real inference, override generate() and compute_perplexity() with
    calls to your model framework (transformers, vLLM, llama.cpp, etc.).
    """

    def __init__(self) -> None:
        try:
            from safetensors import safe_open  # noqa: F401
        except ImportError:
            raise ImportError(
                "safetensors package required. Install with: pip install safetensors"
            )
        self._adapters: dict[str, dict[str, Any]] = {}
        self._counter = 0

    def load_base_model(self, model_path: str) -> None:
        logger.info("Loading base model from %s (safetensors backend)", model_path)
        # In a real implementation, load the base model weights here
        self._model_path = model_path

    def load_adapter(self, adapter_dir: str) -> str:
        from safetensors import safe_open

        adapter_path = Path(adapter_dir) / "adapters.safetensors"
        if not adapter_path.exists():
            raise FileNotFoundError(f"No adapters.safetensors in {adapter_dir}")

        self._counter += 1
        aid = f"sf-adapter-{self._counter}"

        # Load adapter metadata
        meta_path = Path(adapter_dir) / "stack_meta.json"
        meta = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())

        # Load adapter tensors (just the keys/shapes for now)
        tensor_info = {}
        with safe_open(str(adapter_path), framework="numpy") as f:
            for key in f.keys():
                tensor_info[key] = f.get_tensor(key).shape

        self._adapters[aid] = {
            "path": adapter_path,
            "meta": meta,
            "tensor_info": tensor_info,
        }
        n_params = sum(
            1
            for _ in tensor_info
        )
        logger.info(
            "Loaded adapter %s: %d tensors from %s",
            aid, n_params, adapter_path,
        )
        return aid

    def unload_adapter(self, adapter_id: str) -> None:
        self._adapters.pop(adapter_id, None)

    def generate(
        self,
        prompt: str,
        adapter_id: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        # Subclass this for real inference
        raise NotImplementedError(
            "SafetensorsBackend.generate() must be overridden for your "
            "inference framework (transformers, vLLM, llama.cpp, etc.). "
            "Use --dry-run for stub mode."
        )

    def compute_perplexity(
        self,
        text: str,
        adapter_id: str | None = None,
    ) -> float:
        # Subclass this for real perplexity computation
        raise NotImplementedError(
            "SafetensorsBackend.compute_perplexity() must be overridden. "
            "Use --dry-run for stub mode."
        )


class _MoELoRAWrappedLinear:
    """Wrapper that replaces a nn.Linear + MoE-LoRA pair during inference.

    MLX resolves __call__ through the class MRO, so monkey-patching
    __call__ on an instance doesn't work. Instead we replace the module
    attribute on the parent with this wrapper that acts as a callable
    nn.Module-like object, combining base linear + MoE-LoRA delta.
    """

    def __init__(self, base_linear, moe_lora_layer):
        self._base = base_linear
        self._moe = moe_lora_layer
        # Forward all attribute access to the base linear so the model
        # can still access .weight, .bias, etc.
        self.weight = base_linear.weight
        if hasattr(base_linear, "bias"):
            self.bias = base_linear.bias

    def __call__(self, x):
        base_out = self._base(x)
        # MoE-LoRA expects 3D input (batch, seq, features)
        if x.ndim == 2:
            x_3d = x.reshape(1, *x.shape)
            delta = self._moe(x_3d).reshape(x.shape[0], -1)
        else:
            delta = self._moe(x)
        return base_out + delta

    def __getattr__(self, name):
        # Delegate any attribute access not found here to the base linear
        return getattr(self._base, name)


def _patch_moe_lora_forward(model, target_modules: list[str] | None = None) -> int:
    """Replace linear projections with wrapped versions that add MoE-LoRA deltas.

    After apply_moe_lora() attaches MoELoRALayer as sibling attributes
    (e.g. mlp.gate_proj_moe_lora), the model's standard forward pass
    does NOT use them. This function replaces each Linear module with
    a _MoELoRAWrappedLinear that combines base + MoE-LoRA.

    Note: MLX Module's dir() does NOT return dynamically setattr'd
    submodules, so we use the known submodule/target names directly.

    Returns the number of projections patched.
    """
    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    patched = 0

    # Find the layers list (model.model.layers or model.language_model.model.layers)
    if hasattr(model, "language_model"):
        layers = model.language_model.model.layers
    elif hasattr(model, "model"):
        layers = model.model.layers
    else:
        layers = model.layers

    for layer in layers:
        for sub_name in ["self_attn", "mlp"]:
            sub = getattr(layer, sub_name, None)
            if sub is None:
                continue
            for target in target_modules:
                moe_attr = f"{target}_moe_lora"
                moe_layer = getattr(sub, moe_attr, None)
                if moe_layer is None:
                    continue
                base_proj = getattr(sub, target, None)
                if base_proj is None:
                    continue

                # Skip if already wrapped
                if isinstance(base_proj, _MoELoRAWrappedLinear):
                    continue

                # Replace with wrapped version
                wrapped = _MoELoRAWrappedLinear(base_proj, moe_layer)
                setattr(sub, target, wrapped)
                patched += 1

    return patched


def _unpatch_moe_lora_forward(model, target_modules: list[str] | None = None) -> int:
    """Reverse the wrapping done by _patch_moe_lora_forward."""
    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    unpatched = 0

    if hasattr(model, "language_model"):
        layers = model.language_model.model.layers
    elif hasattr(model, "model"):
        layers = model.model.layers
    else:
        layers = model.layers

    for layer in layers:
        for sub_name in ["self_attn", "mlp"]:
            sub = getattr(layer, sub_name, None)
            if sub is None:
                continue
            for target in target_modules:
                proj = getattr(sub, target, None)
                if isinstance(proj, _MoELoRAWrappedLinear):
                    # Restore the original linear
                    setattr(sub, target, proj._base)
                    unpatched += 1
    return unpatched


class MLXBackend(AdapterBackend):
    """Real MLX backend for Apple Silicon eval with MoE-LoRA support.

    Uses the archived MoE-LoRA module from scripts/legacy/moe_lora.py
    to properly apply per-expert LoRA adapters during inference.
    Handles Qwen3.5 thinking mode by stripping <think>...</think> tags.
    """

    # MoE-LoRA config matching brainstacks.yaml
    MOE_LORA_CONFIG = {
        "num_experts": 4,
        "rank": 16,
        "alpha": 32.0,
        "top_k": 2,
        "router_hidden": 64,
    }

    # Target projections for MoE-LoRA
    TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._base_model_path: str | None = None
        self._current_adapter_id: str | None = None
        self._moe_lora_attached = False

    def _strip_thinking(self, text: str) -> str:
        """Strip Qwen3.5 <think>...</think> blocks from response."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def _get_apply_moe_lora(self):
        """Import apply_moe_lora from the archived legacy module.

        Archived 2026-04-19 — see docs/research/2026-04-19-moe-lora-root-cause.md.
        The module moved from scripts/micro_kiki/moe_lora.py to
        scripts/legacy/moe_lora.py because its dual-mount topology
        stranded every ``lora_b`` at zero on pre-pivot adapters.
        """
        try:
            # Try importing from the project's scripts directory
            import importlib.util
            for scripts_dir in [
                Path(self._base_model_path).parent.parent / "scripts",
                PROJECT_ROOT / "scripts",
            ]:
                # Post-archival location
                moe_path = scripts_dir / "legacy" / "moe_lora.py"
                # Fallback to pre-archival location for stale checkouts
                if not moe_path.exists():
                    moe_path = scripts_dir / "micro_kiki" / "moe_lora.py"
                if moe_path.exists():
                    spec = importlib.util.spec_from_file_location(
                        "legacy.moe_lora", str(moe_path)
                    )
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    return mod.apply_moe_lora
        except Exception as exc:
            logger.warning("Failed to import moe_lora: %s", exc)

        # Fallback: try sys.path
        try:
            from legacy.moe_lora import apply_moe_lora  # type: ignore[import-not-found]
            return apply_moe_lora
        except ImportError:
            raise RuntimeError(
                "Cannot import apply_moe_lora. Ensure scripts/legacy/moe_lora.py "
                "exists in the project or is on PYTHONPATH."
            )

    def load_base_model(self, path: str) -> None:
        try:
            from mlx_lm import load
            self._base_model_path = path
            logger.info("Loading base model from %s ...", path)
            self._model, self._tokenizer = load(path)
            self._model.freeze()
            self._current_adapter_id = None
            self._moe_lora_attached = False
            logger.info("Base model loaded successfully")
        except ImportError:
            raise RuntimeError("mlx-lm not installed. Run: pip install mlx-lm")

    def _attach_moe_lora(self) -> None:
        """Attach MoE-LoRA layer structure to the model (once)."""
        if self._moe_lora_attached:
            return

        apply_moe_lora = self._get_apply_moe_lora()

        # Find the right submodel to attach to
        if hasattr(self._model, "language_model"):
            target = self._model.language_model
        else:
            target = self._model

        cfg = self.MOE_LORA_CONFIG
        n = apply_moe_lora(
            target,
            target_modules=self.TARGET_MODULES,
            num_experts=cfg["num_experts"],
            rank=cfg["rank"],
            alpha=cfg["alpha"],
            top_k=cfg["top_k"],
            router_hidden=cfg["router_hidden"],
        )
        logger.info("Attached %d MoE-LoRA layers to model", n)
        self._moe_lora_attached = True

    def load_adapter(self, adapter_dir: str) -> str:
        """Load MoE-LoRA adapter weights and patch the forward pass."""
        import mlx.core as mx

        adapter_path = Path(adapter_dir) / "adapters.safetensors"
        if not adapter_path.exists():
            raise FileNotFoundError(f"No adapter at {adapter_path}")

        # Ensure MoE-LoRA structure is attached
        self._attach_moe_lora()

        # Load adapter weights (only _moe_lora keys will match)
        logger.info("Loading adapter weights from %s ...", adapter_dir)
        self._model.load_weights(str(adapter_path), strict=False)

        # Patch forward pass to use MoE-LoRA deltas
        n_patched = _patch_moe_lora_forward(self._model)
        logger.info("Patched %d projections with MoE-LoRA forward", n_patched)

        adapter_id = Path(adapter_dir).name
        self._current_adapter_id = adapter_id
        return adapter_id

    def unload_adapter(self, adapter_id: str) -> None:
        """Unpatch MoE-LoRA forward and reload base model to clear weights."""
        if self._current_adapter_id is not None:
            n = _unpatch_moe_lora_forward(self._model)
            logger.debug("Unpatched %d projections", n)

            # Reload base model to clear adapter weights
            from mlx_lm import load
            logger.info("Reloading base model to clear adapter weights ...")
            self._model, self._tokenizer = load(self._base_model_path)
            self._model.freeze()
            self._moe_lora_attached = False
            self._current_adapter_id = None

    def generate(
        self,
        prompt: str,
        adapter_id: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.3,
    ) -> str:
        from mlx_lm import generate as mlx_generate
        messages = [{"role": "user", "content": prompt}]
        try:
            formatted = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # Fallback if enable_thinking is not supported
            formatted = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        response = mlx_generate(
            self._model, self._tokenizer, prompt=formatted,
            max_tokens=max_tokens, verbose=False,
        )
        # Strip any residual thinking tags
        return self._strip_thinking(response)

    def compute_perplexity(
        self,
        text: str,
        adapter_id: str | None = None,
    ) -> float:
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


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------


def score_keyword_hits(response: str, domain: str) -> float:
    """Compute fraction of domain keywords found in the response.

    Returns a float in [0.0, 1.0].
    """
    keywords = DOMAIN_KEYWORDS.get(domain, [])
    if not keywords:
        return 0.0
    response_lower = response.lower()
    hits = sum(1 for kw in keywords if kw.lower() in response_lower)
    return round(hits / len(keywords), 4)


def score_response_length(response: str) -> dict[str, Any]:
    """Score response length: flag degenerate (too short/too long) responses.

    Returns dict with length, token_estimate, and a quality flag.
    """
    length = len(response)
    # Rough token estimate (1 token ~ 4 chars for English)
    token_est = length / 4.0

    if token_est < 10:
        quality = "degenerate_short"
    elif token_est < 30:
        quality = "too_short"
    elif token_est > 2000:
        quality = "too_long"
    elif token_est > 1000:
        quality = "verbose"
    else:
        quality = "ok"

    return {
        "char_length": length,
        "token_estimate": round(token_est),
        "quality": quality,
    }


# ---------------------------------------------------------------------------
# Stack discovery
# ---------------------------------------------------------------------------


def discover_stacks(stack_dir: Path) -> dict[str, Path]:
    """Discover domain stacks in a directory.

    Looks for subdirectories containing adapters.safetensors.
    Returns a dict mapping domain name -> stack directory path.
    """
    stacks: dict[str, Path] = {}
    if not stack_dir.exists():
        logger.warning("Stack directory does not exist: %s", stack_dir)
        return stacks

    for sub in sorted(stack_dir.iterdir()):
        if not sub.is_dir():
            continue
        adapter_file = sub / "adapters.safetensors"
        if adapter_file.exists():
            # Domain name is the directory name
            domain = sub.name
            stacks[domain] = sub
            logger.debug("Found stack: %s -> %s", domain, sub)

    logger.info("Discovered %d stacks in %s", len(stacks), stack_dir)
    return stacks


# ---------------------------------------------------------------------------
# Load prompts from data/prompts/ files
# ---------------------------------------------------------------------------


def load_file_prompts(domain: str, max_prompts: int = 5) -> list[str]:
    """Load prompts from data/prompts/<domain>.jsonl or prompts-expanded/.

    Falls back to built-in DOMAIN_PROMPTS if no file exists.
    """
    for subdir in ("prompts-expanded", "prompts"):
        path = PROJECT_ROOT / "data" / subdir / f"{domain}.jsonl"
        if path.exists():
            prompts = []
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = obj.get("prompt", "")
                    if not text:
                        msgs = obj.get("messages", [])
                        user_msgs = [m["content"] for m in msgs if m.get("role") == "user"]
                        text = user_msgs[0] if user_msgs else ""
                    if text:
                        prompts.append(text)
                    if len(prompts) >= max_prompts:
                        break
            if prompts:
                logger.debug("Loaded %d prompts from %s", len(prompts), path)
                return prompts

    # Fall back to built-in prompts
    builtin = DOMAIN_PROMPTS.get(domain, [])
    if builtin:
        logger.debug("Using %d built-in prompts for %s", len(builtin), domain)
        return builtin[:max_prompts]

    logger.warning("No prompts available for domain '%s'", domain)
    return []


# ---------------------------------------------------------------------------
# Per-domain evaluation
# ---------------------------------------------------------------------------


@dataclass
class DomainResult:
    domain: str
    version: str  # "v2" or "v3"
    prompts_evaluated: int = 0
    avg_perplexity: float = 0.0
    avg_keyword_hits: float = 0.0
    length_stats: dict[str, Any] = field(default_factory=dict)
    degenerate_count: int = 0
    per_prompt: list[dict[str, Any]] = field(default_factory=list)
    adapter_loaded: bool = False
    error: str | None = None


def evaluate_domain(
    domain: str,
    version: str,
    adapter_dir: Path | None,
    backend: AdapterBackend,
    prompts: list[str],
    max_tokens: int = 256,
) -> DomainResult:
    """Evaluate a single domain+version combination."""
    result = DomainResult(domain=domain, version=version)

    if not prompts:
        result.error = "no_prompts"
        return result

    adapter_id = None
    if adapter_dir is not None:
        try:
            adapter_id = backend.load_adapter(str(adapter_dir))
            result.adapter_loaded = True
        except (FileNotFoundError, ImportError) as exc:
            result.error = f"adapter_load_failed: {exc}"
            logger.warning("[%s/%s] Failed to load adapter: %s", domain, version, exc)
            return result

    perplexities = []
    keyword_scores = []
    lengths = []
    degenerate = 0

    try:
        for i, prompt in enumerate(prompts):
            t0 = time.monotonic()

            # Generate response
            try:
                response = backend.generate(
                    prompt, adapter_id=adapter_id, max_tokens=max_tokens
                )
            except NotImplementedError:
                response = f"[NOT_IMPLEMENTED] Backend does not support generation."

            gen_time = time.monotonic() - t0

            # Perplexity
            try:
                ppl = backend.compute_perplexity(response, adapter_id=adapter_id)
            except NotImplementedError:
                ppl = float("nan")

            # Keyword hits
            kw_score = score_keyword_hits(response, domain)

            # Length analysis
            len_info = score_response_length(response)
            if len_info["quality"] in ("degenerate_short",):
                degenerate += 1

            perplexities.append(ppl)
            keyword_scores.append(kw_score)
            lengths.append(len_info["token_estimate"])

            result.per_prompt.append({
                "prompt_idx": i,
                "prompt_snippet": prompt[:80],
                "response_snippet": response[:120],
                "perplexity": ppl,
                "keyword_hits": kw_score,
                "length": len_info,
                "gen_time_s": round(gen_time, 3),
            })

    finally:
        if adapter_id:
            backend.unload_adapter(adapter_id)

    result.prompts_evaluated = len(prompts)
    valid_ppls = [p for p in perplexities if not math.isnan(p)]
    result.avg_perplexity = (
        round(sum(valid_ppls) / len(valid_ppls), 4) if valid_ppls else float("nan")
    )
    result.avg_keyword_hits = (
        round(sum(keyword_scores) / len(keyword_scores), 4) if keyword_scores else 0.0
    )
    result.degenerate_count = degenerate
    result.length_stats = {
        "mean": round(sum(lengths) / len(lengths)) if lengths else 0,
        "min": min(lengths) if lengths else 0,
        "max": max(lengths) if lengths else 0,
    }

    return result


# ---------------------------------------------------------------------------
# Cross-evaluation (forgetting metric)
# ---------------------------------------------------------------------------


def cross_evaluate(
    domains: list[str],
    version: str,
    stacks: dict[str, Path],
    backend: AdapterBackend,
    prompts_per_domain: dict[str, list[str]],
    max_cross_prompts: int = 2,
) -> dict[str, dict[str, float]]:
    """Cross-evaluate: test domain X's adapter on domain Y's prompts.

    Returns a matrix: forgetting_matrix[adapter_domain][prompt_domain] = keyword_score.
    A well-specialized adapter should score high on its own domain and
    lower on unrelated domains (good specialization without catastrophic forgetting).
    """
    matrix: dict[str, dict[str, float]] = {}

    for adapter_domain in domains:
        if adapter_domain not in stacks:
            continue

        adapter_id = None
        try:
            adapter_id = backend.load_adapter(str(stacks[adapter_domain]))
        except Exception as exc:
            logger.warning(
                "[cross-eval] Failed to load adapter for %s: %s",
                adapter_domain, exc,
            )
            continue

        matrix[adapter_domain] = {}

        try:
            for prompt_domain in domains:
                test_prompts = prompts_per_domain.get(prompt_domain, [])[:max_cross_prompts]
                if not test_prompts:
                    continue

                kw_scores = []
                for prompt in test_prompts:
                    try:
                        response = backend.generate(
                            prompt, adapter_id=adapter_id, max_tokens=128
                        )
                    except NotImplementedError:
                        response = "[NOT_IMPLEMENTED]"
                    kw_scores.append(score_keyword_hits(response, prompt_domain))

                avg_kw = sum(kw_scores) / len(kw_scores) if kw_scores else 0.0
                matrix[adapter_domain][prompt_domain] = round(avg_kw, 4)

        finally:
            if adapter_id:
                backend.unload_adapter(adapter_id)

    return matrix


def compute_forgetting_rate(
    cross_matrix: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Compute forgetting statistics from the cross-eval matrix.

    Forgetting rate = how much an adapter degrades on other domains
    relative to its own domain.
    """
    if not cross_matrix:
        return {"forgetting_rate": None, "specialization": None}

    self_scores = []
    other_scores = []
    per_adapter = {}

    for adapter_domain, prompt_scores in cross_matrix.items():
        self_score = prompt_scores.get(adapter_domain, 0.0)
        others = [
            s for d, s in prompt_scores.items() if d != adapter_domain
        ]
        avg_other = sum(others) / len(others) if others else 0.0

        self_scores.append(self_score)
        other_scores.append(avg_other)

        per_adapter[adapter_domain] = {
            "self_score": self_score,
            "avg_other_score": round(avg_other, 4),
            "specialization": round(self_score - avg_other, 4),
        }

    avg_self = sum(self_scores) / len(self_scores) if self_scores else 0.0
    avg_other = sum(other_scores) / len(other_scores) if other_scores else 0.0

    return {
        "avg_self_domain_score": round(avg_self, 4),
        "avg_cross_domain_score": round(avg_other, 4),
        "specialization_gap": round(avg_self - avg_other, 4),
        "per_adapter": per_adapter,
    }


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------


def composite_score(result: DomainResult) -> float:
    """Compute a composite score from perplexity, keywords, and length.

    Score in [0, 100]:
      - Perplexity: 40% weight (lower is better, mapped to 0-40)
      - Keyword hits: 40% weight (higher is better, mapped to 0-40)
      - Length quality: 20% weight (non-degenerate = 20, degenerate = 0)
    """
    # Perplexity component: map [1, 50] -> [40, 0], clamped
    ppl = result.avg_perplexity
    if math.isnan(ppl):
        ppl_score = 20.0  # neutral when unknown
    else:
        ppl_clamped = max(1.0, min(50.0, ppl))
        ppl_score = 40.0 * (1.0 - (ppl_clamped - 1.0) / 49.0)

    # Keyword component
    kw_score = 40.0 * result.avg_keyword_hits

    # Length component
    if result.prompts_evaluated > 0:
        degen_frac = result.degenerate_count / result.prompts_evaluated
        len_score = 20.0 * (1.0 - degen_frac)
    else:
        len_score = 0.0

    return round(ppl_score + kw_score + len_score, 2)


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------


def build_comparison(
    v2_results: dict[str, DomainResult],
    v3_results: dict[str, DomainResult],
    v2_cross: dict[str, dict[str, float]] | None = None,
    v3_cross: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Build the full comparison report."""

    all_domains = sorted(set(v2_results.keys()) | set(v3_results.keys()))

    # Per-domain comparison table
    table: list[dict[str, Any]] = []
    v2_scores = []
    v3_scores = []
    deltas = []
    regressions = []

    for domain in all_domains:
        v2r = v2_results.get(domain)
        v3r = v3_results.get(domain)

        v2s = composite_score(v2r) if v2r else 0.0
        v3s = composite_score(v3r) if v3r else 0.0
        delta = round(v3s - v2s, 2)

        if v2r:
            v2_scores.append(v2s)
        if v3r:
            v3_scores.append(v3s)
        if v2r and v3r:
            deltas.append(delta)

        winner = "v3" if v3s > v2s else ("v2" if v2s > v3s else "tie")
        if delta < -5.0:
            regressions.append({"domain": domain, "delta": delta})

        row = {
            "domain": domain,
            "v2_score": v2s,
            "v3_score": v3s,
            "delta": delta,
            "winner": winner,
            "v2_only": domain in v2_results and domain not in v3_results,
            "v3_only": domain not in v2_results and domain in v3_results,
        }

        # Add sub-metrics for detail
        if v2r:
            row["v2_perplexity"] = v2r.avg_perplexity
            row["v2_keyword_hits"] = v2r.avg_keyword_hits
            row["v2_degenerate"] = v2r.degenerate_count
        if v3r:
            row["v3_perplexity"] = v3r.avg_perplexity
            row["v3_keyword_hits"] = v3r.avg_keyword_hits
            row["v3_degenerate"] = v3r.degenerate_count

        table.append(row)

    # Aggregate stats
    avg_v2 = sum(v2_scores) / len(v2_scores) if v2_scores else 0.0
    avg_v3 = sum(v3_scores) / len(v3_scores) if v3_scores else 0.0
    avg_delta = sum(deltas) / len(deltas) if deltas else 0.0
    v3_wins = sum(1 for r in table if r["winner"] == "v3")
    v2_wins = sum(1 for r in table if r["winner"] == "v2")
    ties = sum(1 for r in table if r["winner"] == "tie")

    # Forgetting analysis
    forgetting = {}
    if v2_cross:
        forgetting["v2"] = compute_forgetting_rate(v2_cross)
    if v3_cross:
        forgetting["v3"] = compute_forgetting_rate(v3_cross)

    report = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "v2_domains": len(v2_results),
            "v3_domains": len(v3_results),
            "total_domains_compared": len(all_domains),
            "prompts_per_domain": 5,
        },
        "aggregate": {
            "avg_v2_score": round(avg_v2, 2),
            "avg_v3_score": round(avg_v3, 2),
            "avg_improvement": round(avg_delta, 2),
            "v3_wins": v3_wins,
            "v2_wins": v2_wins,
            "ties": ties,
            "worst_regressions": sorted(regressions, key=lambda r: r["delta"]),
        },
        "comparison_table": table,
    }

    if forgetting:
        report["forgetting"] = forgetting

    return report


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------


def print_table(report: dict[str, Any]) -> None:
    """Print a human-readable comparison table to stdout."""
    table = report["comparison_table"]
    agg = report["aggregate"]

    print()
    print("=" * 80)
    print("  micro-kiki V2 vs V3 Stack Evaluation")
    print("=" * 80)
    print()
    print(
        f"{'Domain':<20} {'V2 Score':>10} {'V3 Score':>10} "
        f"{'Delta':>8} {'Winner':>8}"
    )
    print("-" * 60)

    for row in table:
        marker = ""
        if row.get("v2_only"):
            marker = " (v2 only)"
        elif row.get("v3_only"):
            marker = " (v3 only)"

        print(
            f"{row['domain']:<20} "
            f"{row['v2_score']:>10.2f} "
            f"{row['v3_score']:>10.2f} "
            f"{row['delta']:>+8.2f} "
            f"{row['winner']:>8}"
            f"{marker}"
        )

    print("-" * 60)
    print(
        f"{'AVERAGE':<20} "
        f"{agg['avg_v2_score']:>10.2f} "
        f"{agg['avg_v3_score']:>10.2f} "
        f"{agg['avg_improvement']:>+8.2f}"
    )
    print()
    print(
        f"  V3 wins: {agg['v3_wins']}  |  "
        f"V2 wins: {agg['v2_wins']}  |  "
        f"Ties: {agg['ties']}"
    )

    if agg["worst_regressions"]:
        print()
        print("  Worst regressions (delta < -5.0):")
        for reg in agg["worst_regressions"]:
            print(f"    {reg['domain']}: {reg['delta']:+.2f}")

    # Forgetting summary
    forgetting = report.get("forgetting", {})
    if forgetting:
        print()
        print("  Forgetting analysis (cross-eval):")
        for ver, stats in forgetting.items():
            spec = stats.get("specialization_gap", 0)
            self_s = stats.get("avg_self_domain_score", 0)
            cross_s = stats.get("avg_cross_domain_score", 0)
            print(
                f"    {ver}: self={self_s:.3f}  cross={cross_s:.3f}  "
                f"specialization_gap={spec:+.3f}"
            )

    print()
    print("=" * 80)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark V2 vs V3 MoE-LoRA stacks on Qwen3.5-4B. "
            "Compares per-domain adapters using perplexity, keyword hits, "
            "response length, and forgetting metrics."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Built-in domains with test prompts:\n  "
            + "\n  ".join(sorted(DOMAIN_PROMPTS.keys()))
            + "\n\nAlso loads prompts from data/prompts/<domain>.jsonl if available."
        ),
    )

    parser.add_argument(
        "--v2-dir",
        required=True,
        help="Path to V2 stacks directory (e.g. output/micro-kiki/stacks-v2/)",
    )
    parser.add_argument(
        "--v3-dir",
        required=True,
        help="Path to V3 stacks directory (e.g. output/micro-kiki/stacks/)",
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="Path to base model (Qwen3.5-4B) or HuggingFace repo ID",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help=(
            "Specific domains to evaluate (default: all discovered). "
            "Example: --domains chat-fr python embedded"
        ),
    )
    parser.add_argument(
        "--prompts-per-domain",
        type=int,
        default=5,
        help="Number of test prompts per domain (default: 5)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max tokens to generate per response (default: 256)",
    )
    parser.add_argument(
        "--cross-eval",
        action="store_true",
        help="Enable cross-domain forgetting evaluation (slower)",
    )
    parser.add_argument(
        "--cross-prompts",
        type=int,
        default=2,
        help="Prompts per domain in cross-eval (default: 2)",
    )
    parser.add_argument(
        "--output",
        default="results/v2-vs-v3.json",
        help="Output JSON path (default: results/v2-vs-v3.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use stub backend (no model loading, for testing the pipeline)",
    )
    parser.add_argument(
        "--backend",
        choices=["stub", "safetensors", "mlx"],
        default=None,
        help="Backend to use (default: auto-detect from --dry-run flag)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Select backend
    if args.dry_run or args.backend == "stub":
        backend: AdapterBackend = StubBackend()
        logger.info("Using stub backend (dry-run mode)")
    elif args.backend == "mlx":
        backend = MLXBackend()
        logger.info("Using MLX backend")
    elif args.backend == "safetensors":
        backend = SafetensorsBackend()
    else:
        # Default: try MLX first, then safetensors, fall back to stub
        try:
            backend = MLXBackend()
            logger.info("Using MLX backend (auto-detected)")
        except Exception:
            try:
                backend = SafetensorsBackend()
                logger.info("Using safetensors backend")
            except ImportError:
                logger.warning(
                    "Neither mlx-lm nor safetensors installed, "
                    "falling back to stub backend."
                )
                backend = StubBackend()

    # Load base model
    backend.load_base_model(args.base_model)

    # Discover stacks
    v2_dir = Path(args.v2_dir)
    v3_dir = Path(args.v3_dir)
    v2_stacks = discover_stacks(v2_dir)
    v3_stacks = discover_stacks(v3_dir)

    logger.info("V2 stacks: %d domains", len(v2_stacks))
    logger.info("V3 stacks: %d domains", len(v3_stacks))

    # Determine which domains to evaluate
    if args.domains:
        eval_domains = args.domains
    else:
        eval_domains = sorted(set(v2_stacks.keys()) | set(v3_stacks.keys()))

    if not eval_domains:
        # Fall back to built-in prompt domains
        eval_domains = sorted(DOMAIN_PROMPTS.keys())
        logger.warning(
            "No stacks discovered. Using built-in prompt domains for dry-run."
        )

    logger.info("Evaluating %d domains: %s", len(eval_domains), eval_domains)

    # Load prompts for each domain
    prompts_per_domain: dict[str, list[str]] = {}
    for domain in eval_domains:
        prompts_per_domain[domain] = load_file_prompts(
            domain, max_prompts=args.prompts_per_domain
        )

    # Evaluate V2
    v2_results: dict[str, DomainResult] = {}
    for domain in eval_domains:
        prompts = prompts_per_domain[domain]
        adapter_dir = v2_stacks.get(domain)
        result = evaluate_domain(
            domain, "v2", adapter_dir, backend, prompts, args.max_tokens
        )
        v2_results[domain] = result
        logger.info(
            "[v2/%s] ppl=%.2f  kw=%.3f  degen=%d/%d",
            domain,
            result.avg_perplexity,
            result.avg_keyword_hits,
            result.degenerate_count,
            result.prompts_evaluated,
        )

    # Evaluate V3
    v3_results: dict[str, DomainResult] = {}
    for domain in eval_domains:
        prompts = prompts_per_domain[domain]
        adapter_dir = v3_stacks.get(domain)
        result = evaluate_domain(
            domain, "v3", adapter_dir, backend, prompts, args.max_tokens
        )
        v3_results[domain] = result
        logger.info(
            "[v3/%s] ppl=%.2f  kw=%.3f  degen=%d/%d",
            domain,
            result.avg_perplexity,
            result.avg_keyword_hits,
            result.degenerate_count,
            result.prompts_evaluated,
        )

    # Cross-eval (forgetting)
    v2_cross = None
    v3_cross = None
    if args.cross_eval:
        logger.info("Running cross-domain forgetting evaluation...")
        v2_cross = cross_evaluate(
            eval_domains, "v2", v2_stacks, backend,
            prompts_per_domain, args.cross_prompts,
        )
        v3_cross = cross_evaluate(
            eval_domains, "v3", v3_stacks, backend,
            prompts_per_domain, args.cross_prompts,
        )

    # Build comparison report
    report = build_comparison(v2_results, v3_results, v2_cross, v3_cross)

    # Add metadata
    report["config"]["v2_dir"] = str(v2_dir)
    report["config"]["v3_dir"] = str(v3_dir)
    report["config"]["base_model"] = args.base_model
    report["config"]["max_tokens"] = args.max_tokens
    report["config"]["dry_run"] = args.dry_run

    # Print to stdout
    print_table(report)

    # Save JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    logger.info("Report saved -> %s", out_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate synthetic training data for sparse Micro_KIKI domains.

Uses local teacher models (Qwen3.5-122B, Qwen3.5-35B via mlx-vlm)
to fill domains that have fewer examples than their target after classification.

Usage:
    python scripts/micro_kiki/generate_missing.py \
        --config configs/micro_kiki/domains.yaml \
        --classified-dir data/micro-kiki/classified \
        --output-dir data/micro-kiki/generated \
        --teacher-model models/Qwen3.5-35B-A3B-Opus-vlm \
        [--max-generate 500]
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import yaml


# Seed prompts per domain category to bootstrap generation
DOMAIN_SEED_PROMPTS: dict[str, list[str]] = {
    "chat-fr": [
        "Explique le concept de la récursivité en termes simples, en français.",
        "Rédige un email professionnel pour demander un délai supplémentaire sur un projet.",
        "Compare les avantages et inconvénients du travail à distance, en français.",
        "Résume les principaux enjeux du changement climatique pour un public non-spécialisé.",
        "Donne des conseils pratiques pour organiser une réunion d'équipe efficace.",
    ],
    "reasoning": [
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Think step by step.",
        "If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly? Analyze the logical structure.",
        "Three switches control three light bulbs in another room. You can flip switches as many times as you want but can only enter the room once. How do you determine which switch controls which bulb?",
        "Explain why correlation does not imply causation, with a concrete example.",
        "A farmer needs to cross a river with a wolf, a goat, and cabbage. He can only take one at a time. The wolf eats the goat if left alone, the goat eats the cabbage. How does he do it?",
    ],
    "python": [
        "Implement a thread-safe LRU cache in Python with O(1) get and put operations.",
        "Write a Python decorator that retries a function up to N times with exponential backoff.",
        "Create a Python context manager that temporarily changes the working directory.",
        "Implement a Python generator that yields Fibonacci numbers lazily with memoization.",
        "Write a Python function that validates a nested JSON structure against a schema dict.",
    ],
    "typescript": [
        "Implement a type-safe event emitter in TypeScript with generics.",
        "Write a TypeScript utility type that makes all nested properties optional (DeepPartial).",
        "Create a React custom hook useDebounce<T>(value: T, delay: number) in TypeScript.",
        "Implement a TypeScript discriminated union for API response handling with exhaustive checks.",
        "Write a Zod schema for validating a complex form with nested address and payment fields.",
    ],
    "cpp": [
        "Implement a RAII wrapper for a file descriptor in C++ that handles move semantics correctly.",
        "Write a C++ template metaprogram that computes factorial at compile time using constexpr.",
        "Implement a lock-free single-producer single-consumer queue in C++ using atomics.",
        "Create a C++ smart pointer implementation similar to unique_ptr with custom deleter support.",
        "Write a C++ coroutine-based async TCP client using C++20 coroutines.",
    ],
    "rust": [
        "Implement a concurrent hash map in Rust using RwLock with proper lifetime annotations.",
        "Write a Rust macro that generates getter/setter methods for struct fields.",
        "Implement a Rust async stream that reads chunks from a file using tokio.",
        "Create a Rust trait object with dynamic dispatch and explain the trade-offs vs generics.",
        "Write a Rust error type hierarchy using thiserror with conversion from multiple sources.",
    ],
    "html-css": [
        "Build a responsive card grid using CSS Grid that collapses to a single column on mobile.",
        "Create a pure CSS animated hamburger menu toggle without JavaScript.",
        "Implement a dark/light theme toggle using CSS custom properties and prefers-color-scheme.",
        "Build an accessible modal dialog with proper focus trapping using only HTML and CSS.",
        "Create a CSS-only tooltip that appears on hover with an arrow pointer.",
    ],
    "shell": [
        "Write a bash script that monitors a directory for new files and processes them automatically.",
        "Create a shell function that safely rotates log files with compression and retention policy.",
        "Write a bash script that creates a full system backup with incremental snapshots using rsync.",
        "Implement a shell script that parses command-line arguments with both short and long options.",
        "Write a bash script that checks the health of multiple services and sends alerts on failure.",
    ],
    "sql": [
        "Write a SQL query to find the second highest salary in each department using window functions.",
        "Design a database schema for a multi-tenant SaaS application with row-level security.",
        "Write a recursive CTE that traverses a tree structure stored as adjacency list.",
        "Create a SQL migration that adds a new column with a default value without locking the table.",
        "Write a query that implements a leaderboard with dense ranking and pagination.",
    ],
    "yaml-json": [
        "Write an OpenAPI 3.1 spec for a REST API with authentication, pagination, and error responses.",
        "Create a JSON Schema that validates a complex configuration file with conditional requirements.",
        "Write a GitHub Actions workflow YAML that runs tests, builds, and deploys on tag push.",
        "Design a Kubernetes deployment YAML with health checks, resource limits, and auto-scaling.",
        "Create a docker-compose.yaml for a full stack app with database, cache, and reverse proxy.",
    ],
    "docker": [
        "Write a multi-stage Dockerfile for a Node.js app that minimizes the final image size.",
        "Create a docker-compose setup for a development environment with hot reload and debugging.",
        "Write a Dockerfile for a Python ML application with CUDA support and proper layer caching.",
        "Implement a Docker health check that verifies both the HTTP endpoint and database connectivity.",
        "Design a Docker networking setup for microservices with service discovery and load balancing.",
    ],
    "kicad-dsl": [
        "Write a KiCad symbol library entry for a custom voltage regulator with all pin definitions.",
        "Create a KiCad footprint for a QFN-32 package with thermal pad and proper clearances.",
        "Write a KiCad schematic s-expression for a basic ESP32-S3 minimum circuit.",
        "Generate a KiCad netlist from a hierarchical schematic with power flags.",
        "Create a KiCad symbol for a connector with variant pins and electrical types.",
    ],
    "spice": [
        "Write a SPICE netlist for a common-emitter amplifier with biasing network.",
        "Create an ngspice simulation for a second-order active low-pass Butterworth filter.",
        "Write a SPICE subcircuit model for a simple op-amp with gain and bandwidth parameters.",
        "Design a SPICE testbench for a buck converter with switching waveforms.",
        "Create an LTspice simulation for a crystal oscillator startup behavior.",
    ],
    "lua-upy": [
        "Write a MicroPython driver for an I2C temperature sensor (e.g., TMP102).",
        "Implement a Lua coroutine-based cooperative multitasking scheduler.",
        "Create a MicroPython web server that serves sensor data as JSON over WiFi.",
        "Write a Lua module for parsing and generating MQTT packets.",
        "Implement a MicroPython class for controlling a stepper motor with acceleration profiles.",
    ],
    "embedded": [
        "Write ESP-IDF code to configure and use the ADC with DMA for continuous sampling.",
        "Implement a FreeRTOS task that manages a circular buffer with ISR-safe producer/consumer.",
        "Create an ESP-IDF component for OTA firmware updates with rollback support.",
        "Write bare-metal startup code for an ARM Cortex-M4 with vector table and clock init.",
        "Implement a bootloader that verifies firmware integrity using CRC32 before jumping to app.",
    ],
    "stm32": [
        "Configure STM32 HAL for UART communication with DMA and idle line detection.",
        "Write STM32 code for reading multiple ADC channels with DMA and double buffering.",
        "Implement a STM32 USB CDC device class for virtual COM port communication.",
        "Configure STM32 timer for PWM generation with complementary outputs and dead time.",
        "Write STM32 CAN bus driver using HAL with message filtering and error handling.",
    ],
    "iot": [
        "Implement an MQTT client on ESP32 with TLS, auto-reconnect, and QoS 1 message handling.",
        "Write BLE GATT server code for ESP32 that exposes sensor readings as characteristics.",
        "Create an ESP-NOW mesh network protocol with peer discovery and message routing.",
        "Implement a LoRa point-to-point communication system with acknowledgment and retry.",
        "Design a Home Assistant custom component that integrates with a custom MQTT device.",
    ],
    "freecad": [
        "Write a FreeCAD Python macro to create a parametric enclosure with snap-fit joints.",
        "Create a FreeCAD script that imports a STEP file and extracts dimensional measurements.",
        "Write a FreeCAD PartDesign script for a gear with involute tooth profile.",
        "Implement a FreeCAD macro for generating a PCB enclosure from board outline dimensions.",
        "Create a FreeCAD assembly script that positions components with mate constraints.",
    ],
    "platformio": [
        "Write a platformio.ini configuration for multi-environment ESP32 and STM32 builds.",
        "Create a PlatformIO custom build script that generates version headers from git tags.",
        "Configure PlatformIO for unit testing with Unity framework on native and embedded targets.",
        "Write a PlatformIO library.json for a reusable sensor library with dependency management.",
        "Set up PlatformIO CI with GitHub Actions for building, testing, and OTA deployment.",
    ],
    "power": [
        "Design a synchronous buck converter circuit for 12V to 3.3V at 3A with component selection.",
        "Calculate and design an LDO power supply for sensitive analog circuits with noise analysis.",
        "Design a battery charging circuit for 3S Li-Ion with balancing and protection.",
        "Implement MPPT algorithm for a solar panel charger with perturb-and-observe method.",
        "Design a power sequencing circuit for an FPGA board with multiple voltage rails.",
    ],
    "emc": [
        "Design a common-mode choke filter for a USB 2.0 interface with impedance calculations.",
        "Calculate the required decoupling capacitor values for a 100MHz digital IC.",
        "Design a shielded enclosure for a sensitive RF receiver with gasket specifications.",
        "Analyze and fix a conducted EMI problem on a switching power supply.",
        "Design a ferrite bead filter network for a noisy power rail feeding an ADC.",
    ],
    "dsp": [
        "Implement a real-time FIR filter in C using circular buffer and CMSIS-DSP optimizations.",
        "Design a digital PLL (phase-locked loop) for clock recovery from a noisy signal.",
        "Implement an FFT-based spectrum analyzer with windowing and overlap-add method.",
        "Write a Goertzel algorithm implementation for detecting specific DTMF frequencies.",
        "Implement a Kalman filter for sensor fusion of accelerometer and gyroscope data.",
    ],
    "spice-sim": [
        "Set up a Monte Carlo simulation in ngspice to analyze component tolerance effects.",
        "Create a parametric DC sweep simulation to characterize a transistor's IV curves.",
        "Design a transient simulation for a boost converter with startup and load step response.",
        "Implement an AC analysis simulation for a multi-stage amplifier with Bode plot extraction.",
        "Set up a worst-case analysis for an analog filter using corner models.",
    ],
    "electronics": [
        "Design a precision instrumentation amplifier circuit with common-mode rejection analysis.",
        "Calculate the component values for a 4th-order Chebyshev bandpass filter.",
        "Design a current sensing circuit using a shunt resistor and differential amplifier.",
        "Analyze the stability of a feedback amplifier using Nyquist criterion.",
        "Design a voltage reference circuit with temperature compensation for 12-bit ADC accuracy.",
    ],
    "kicad-pcb": [
        "Design a 4-layer PCB stackup for a mixed-signal board with impedance control.",
        "Write KiCad DRC rules for a high-speed differential pair with length matching.",
        "Design a ground plane strategy for a board with both analog and digital sections.",
        "Create routing guidelines for a USB 3.0 interface with impedance matching.",
        "Design a thermal relief pattern for a power MOSFET with via stitching.",
    ],
    "web-frontend": [
        "Implement a React compound component pattern for a customizable Tabs component.",
        "Create a performant virtual scroll list in React that handles 100K items.",
        "Implement optimistic UI updates with rollback for a React + TanStack Query app.",
        "Build a drag-and-drop kanban board using React with accessible keyboard support.",
        "Create a React form with dynamic field arrays, validation, and error display using RHF.",
    ],
    "web-backend": [
        "Implement a FastAPI middleware for rate limiting with Redis backend and sliding window.",
        "Create a Hono API with OpenAPI spec generation, Zod validation, and error handling.",
        "Implement a webhook handler with signature verification, retry logic, and idempotency.",
        "Design a FastAPI dependency injection system for database sessions with proper cleanup.",
        "Build an Express.js API with JWT auth, refresh tokens, and role-based access control.",
    ],
    "music-audio": [
        "Implement a Web Audio API synthesizer with oscillators, filters, and envelope generators.",
        "Write a Python script for real-time audio pitch detection using autocorrelation.",
        "Create a MIDI parser and player in JavaScript that handles note on/off and control changes.",
        "Implement a basic audio compressor/limiter in C for an embedded DSP processor.",
        "Design a TTS pipeline integration using Piper with streaming output and queue management.",
    ],
    "devops": [
        "Write a GitHub Actions workflow for a monorepo with conditional job execution per changed path.",
        "Create a Terraform module for deploying a containerized app to AWS with auto-scaling.",
        "Set up a Prometheus + Grafana monitoring stack with custom alerts and dashboards.",
        "Write an Ansible playbook for provisioning a production server with hardened SSH and firewall.",
        "Design a blue-green deployment strategy using nginx with health checks and automatic rollback.",
    ],
    "llm-orch": [
        "Implement a RAG pipeline with hybrid search (BM25 + vector) and reranking.",
        "Design a multi-agent orchestration system with message passing and task delegation.",
        "Create a prompt routing system that selects the optimal model based on task classification.",
        "Implement a streaming LLM response handler with token-by-token processing and cancellation.",
        "Build a vector store with HNSW indexing, metadata filtering, and incremental updates.",
    ],
    "math": [
        "Prove that the square root of 2 is irrational using proof by contradiction.",
        "Derive the formula for the area of a circle using integration in polar coordinates.",
        "Solve the system of differential equations dx/dt = 2x - y, dy/dt = x + y.",
        "Explain eigenvalue decomposition and its application to principal component analysis.",
        "Prove the central limit theorem intuitively and explain when it fails.",
    ],
    "security": [
        "Implement a constant-time string comparison function to prevent timing attacks.",
        "Design a secure session management system with CSRF protection and cookie security.",
        "Perform a security audit of this login endpoint and list all OWASP Top 10 vulnerabilities.",
        "Implement certificate pinning for a mobile app communicating with a REST API.",
        "Design a secrets management system using HashiCorp Vault with auto-rotation.",
    ],
}


def build_generation_prompt(domain_name: str, domain_cfg: dict) -> str:
    """Build a meta-prompt that asks the teacher to generate a training example.

    The teacher produces a USER/ASSISTANT pair in the target domain with
    <thinking> reasoning blocks.
    """
    keywords = ", ".join(domain_cfg.get("keywords", [])[:8])
    seed_prompts = DOMAIN_SEED_PROMPTS.get(domain_name, [])
    seed = random.choice(seed_prompts) if seed_prompts else f"a question about {domain_name}"

    return f"""You are an expert data generator for fine-tuning LLMs. Generate a high-quality training example for the "{domain_name}" domain.

Domain keywords: {keywords}

Generate a realistic USER question and a detailed ASSISTANT response. The assistant response MUST include <thinking>...</thinking> tags showing the reasoning process before the final answer.

Format your output EXACTLY as:
USER: [the user's question]
ASSISTANT: <thinking>[detailed reasoning]</thinking>

[clear, comprehensive answer]

Example topic direction (vary significantly): {seed}

Requirements:
- The question should be practical and specific, not generic
- The thinking block should show genuine reasoning (2-5 sentences)
- The answer should be technically accurate and detailed
- Include code blocks with correct syntax when relevant
- Vary difficulty: some beginner, some intermediate, some advanced"""


def parse_teacher_response(raw: str) -> dict | None:
    """Parse teacher output into chat format.

    Expected format:
    USER: ...
    ASSISTANT: ...
    """
    # Find USER: and ASSISTANT: markers
    lines = raw.split("\n")
    user_start = -1
    assistant_start = -1

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("USER:") and user_start == -1:
            user_start = i
        elif stripped.startswith("ASSISTANT:") and assistant_start == -1:
            assistant_start = i

    if user_start == -1 or assistant_start == -1:
        return None

    # Extract user content (from USER: line to ASSISTANT: line)
    user_lines = []
    first_user_line = lines[user_start].strip()
    user_lines.append(first_user_line[len("USER:"):].strip())
    for i in range(user_start + 1, assistant_start):
        user_lines.append(lines[i])
    user_content = "\n".join(user_lines).strip()

    # Extract assistant content (from ASSISTANT: line to end)
    assistant_lines = []
    first_assistant_line = lines[assistant_start].strip()
    assistant_lines.append(first_assistant_line[len("ASSISTANT:"):].strip())
    for i in range(assistant_start + 1, len(lines)):
        assistant_lines.append(lines[i])
    assistant_content = "\n".join(assistant_lines).strip()

    if not user_content or not assistant_content:
        return None

    # Normalize thinking tags
    assistant_content = (assistant_content
                         .replace("<think>", "<thinking>")
                         .replace("</think>", "</thinking>"))

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


_loaded_model = None
_loaded_tokenizer = None
_loaded_model_path = None


def _ensure_model(model_path: str):
    """Load model once, reuse across all generations."""
    global _loaded_model, _loaded_tokenizer, _loaded_model_path
    if _loaded_model_path == model_path:
        return _loaded_model, _loaded_tokenizer

    print(f"  Loading teacher model: {model_path} ...")
    t0 = time.time()
    from mlx_lm import load
    _loaded_model, _loaded_tokenizer = load(model_path)
    _loaded_model_path = model_path
    print(f"  Model loaded in {time.time()-t0:.1f}s")
    return _loaded_model, _loaded_tokenizer


def generate_with_mlx(prompt: str, model_path: str, max_tokens: int = 2048) -> str:
    """Generate text using mlx-lm with model loaded once in memory."""
    model, tokenizer = _ensure_model(model_path)

    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=0.8, top_p=0.95)
    try:
        result = generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )
        return result.strip() if isinstance(result, str) else str(result).strip()
    except Exception as e:
        print(f"  Generation error: {e}", file=sys.stderr)
        return ""


def count_classified(classified_dir: Path) -> dict[str, int]:
    """Count examples per domain in classified output."""
    counts = {}
    if not classified_dir.exists():
        return counts
    for f in classified_dir.glob("*.jsonl"):
        domain = f.stem
        with open(f) as fh:
            counts[domain] = sum(1 for line in fh if line.strip())
    return counts


def run_generation(config_path: str, classified_dir: str, output_dir: str,
                   teacher_model: str, max_generate: int = 500,
                   dry_run: bool = False,
                   only_domains: list[str] | None = None) -> dict[str, int]:
    """Generate missing data for sparse domains.

    Returns dict of domain -> number of generated examples.
    """
    config = yaml.safe_load(Path(config_path).read_text())
    domains = config["domains"]
    existing = count_classified(Path(classified_dir))
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    generated_counts = {}
    total_t0 = time.time()

    for name, cfg in domains.items():
        if only_domains and name not in only_domains:
            continue
        target = cfg["target"]
        have = existing.get(name, 0)
        need = max(0, target - have)

        if need == 0:
            print(f"  {name}: {have}/{target} — sufficient")
            generated_counts[name] = 0
            continue

        to_generate = min(need, max_generate)
        print(f"  {name}: {have}/{target} — need {need}, generating {to_generate}")

        if dry_run:
            generated_counts[name] = to_generate
            continue

        generated = []
        attempts = 0
        max_attempts = to_generate * 3  # Allow 3x attempts for parsing failures

        while len(generated) < to_generate and attempts < max_attempts:
            prompt = build_generation_prompt(name, cfg)
            raw = generate_with_mlx(prompt, teacher_model)
            if not raw:
                attempts += 1
                continue

            parsed = parse_teacher_response(raw)
            if parsed is not None:
                generated.append(parsed)
            attempts += 1

            if len(generated) % 10 == 0 and len(generated) > 0:
                print(f"    {name}: {len(generated)}/{to_generate} generated")

        # Write generated examples
        outfile = out_path / f"{name}.jsonl"
        with open(outfile, "w") as f:
            for ex in generated:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        generated_counts[name] = len(generated)
        print(f"    {name}: wrote {len(generated)} examples to {outfile}")

    elapsed = time.time() - total_t0
    total_gen = sum(generated_counts.values())
    print(f"\n=== Generation Summary ===")
    print(f"Total generated: {total_gen} examples in {elapsed:.0f}s")
    if total_gen > 0:
        print(f"Speed: {elapsed/total_gen:.1f}s/example")
    for name, count in sorted(generated_counts.items()):
        if count > 0:
            print(f"  {name}: {count}")

    return generated_counts


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic data for sparse Micro_KIKI domains")
    parser.add_argument("--config", default="configs/micro_kiki/domains.yaml")
    parser.add_argument("--classified-dir", default="data/micro-kiki/classified")
    parser.add_argument("--output-dir", default="data/micro-kiki/generated")
    parser.add_argument("--teacher-model", default="models/Qwen3.5-35B-A3B-Opus-vlm",
                        help="Path to local teacher model for mlx-lm generation")
    parser.add_argument("--max-generate", type=int, default=500,
                        help="Max examples to generate per domain per run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be generated without running the teacher")
    parser.add_argument("--only-domains", type=str, default=None,
                        help="Comma-separated list of domain names to generate (skip others)")
    args = parser.parse_args()

    run_generation(args.config, args.classified_dir, args.output_dir,
                   args.teacher_model, args.max_generate, args.dry_run,
                   only_domains=args.only_domains.split(",") if args.only_domains else None)


if __name__ == "__main__":
    main()

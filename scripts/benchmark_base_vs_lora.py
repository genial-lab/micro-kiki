#!/usr/bin/env python3
"""Benchmark base 35B model across all 32 domains.

Evaluates Qwen3.5-35B-A3B on representative prompts to classify each domain
as "niche" (LoRA needed) or "known" (base model sufficient).

SCAFFOLD: inference calls are stubbed — prompts + scoring framework are real.

Usage:
    uv run python scripts/benchmark_base_vs_lora.py --help
    uv run python scripts/benchmark_base_vs_lora.py \\
        --model Qwen/Qwen3.5-35B-A3B \\
        --output results/base-benchmark.json
    uv run python scripts/benchmark_base_vs_lora.py \\
        --model Qwen/Qwen3.5-35B-A3B \\
        --domains kicad-dsl spice emc \\
        --output results/base-benchmark-niche.json
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain registry
# ---------------------------------------------------------------------------

# "niche" = domains where the base model is expected to be weak,
#            requiring a LoRA adapter.
# "known" = domains where the base model is expected to be sufficient.

DOMAIN_CLASS: dict[str, str] = {
    # --- Niche (hardware/EDA/embedded — highly specialised vocabulary) ---
    "kicad-dsl":    "niche",
    "spice":        "niche",
    "spice-sim":    "niche",
    "emc":          "niche",
    "stm32":        "niche",
    "embedded":     "niche",
    "freecad":      "niche",
    "platformio":   "niche",
    "power":        "niche",
    "dsp":          "niche",
    "kicad-pcb":    "niche",
    "electronics":  "niche",
    "iot":          "niche",
    "lua-upy":      "niche",
    "music-audio":  "niche",
    # --- Known (well-covered by general pre-training) ---
    "chat-fr":      "known",
    "reasoning":    "known",
    "python":       "known",
    "cpp":          "known",
    "math":         "known",
    "typescript":   "known",
    "rust":         "known",
    "html-css":     "known",
    "shell":        "known",
    "sql":          "known",
    "yaml-json":    "known",
    "docker":       "known",
    "web-frontend": "known",
    "web-backend":  "known",
    "devops":       "known",
    "llm-orch":     "known",
    "security":     "known",
}

# ---------------------------------------------------------------------------
# Eval prompts — 5 per domain, technically specific
# ---------------------------------------------------------------------------

DOMAIN_PROMPTS: dict[str, list[str]] = {
    # ── NICHE ────────────────────────────────────────────────────────────────
    "kicad-dsl": [
        "Write a KiCad schematic symbol DSL definition for a 16-pin STM32F103C8T6. "
        "Include pin types (power, I/O, bidirectional) and multi-unit split.",
        "In KiCad 8 footprint scripting (Python API), write a function that creates a "
        "THT resistor footprint from pitch, pad_size, and drill parameters.",
        "Explain the difference between `B.Cu`, `In1.Cu`, and `F.Mask` layer meanings "
        "in a KiCad PCB file (.kicad_pcb) and when each is referenced in an s-expression.",
        "Given a KiCad netlist in IPC-D-356A format, write a parser that extracts "
        "net names, component references, and pin numbers into a Python dict.",
        "Describe the `(net_tie_pad_groups ...)` directive in a KiCad footprint: "
        "what does it do, when is it needed, and provide an example for a 0-Ohm jumper.",
    ],
    "spice": [
        "Write a SPICE netlist for a non-inverting op-amp with gain 10 using TL071, "
        "with AC analysis from 1 Hz to 10 MHz and .MEASURE to extract -3 dB bandwidth.",
        "Explain the difference between SPICE `.TRAN`, `.AC`, `.DC`, and `.NOISE` "
        "analyses, including what each sweeps and what it outputs.",
        "In ngspice, write a behavioral voltage source using B-elements that models "
        "a PWM signal with variable duty cycle controlled by a voltage V(ctrl).",
        "A MOSFET switch driving an inductive load has Vgs=10V, Id=5A, L=100uH. "
        "Write a SPICE subcircuit that models the switching transient and snubber.",
        "Convert a SPICE .MODEL card for a Schottky diode (IS, N, RS, CJO, VJ, M, TT) "
        "into an equivalent KiCad simulation model attribute block.",
    ],
    "spice-sim": [
        "In a SPICE simulation of a buck converter at 400 kHz, the output ripple is "
        "50 mV peak-to-peak. Write the .TRAN command and .MEASURE commands needed "
        "to extract Vout_avg, Vripple_pp, and efficiency.",
        "How do you model parasitic inductance of a PCB trace in SPICE? Write a "
        "subcircuit for a 10 nH trace inductance with 50 mΩ series resistance.",
        "Explain Monte Carlo analysis in ngspice: write a `.MC` command block that "
        "runs 100 trials varying R1 ±5% and C1 ±10% with normal distribution.",
        "A simulation shows ringing at 50 MHz on a gate drive signal. What SPICE "
        "parameters (gate resistance, parasitic L, driver impedance) would you vary "
        "to damp the oscillation, and how would you verify with .MEAS?",
        "Write a Verilog-A model (compatible with ngspice) for a temperature-dependent "
        "resistor with TCR1=3000 ppm/K and TCR2=0.5 ppm/K².",
    ],
    "emc": [
        "Calculate the minimum trace spacing on a 4-layer PCB to maintain 5 kV/mm "
        "creepage distance per IEC 60664-1, for a 230 VAC application at pollution "
        "degree 2, material group IIIa.",
        "A DC-DC converter at 400 kHz fails EN 55032 Class B conducted emissions at "
        "the 5th harmonic. Describe three mitigation techniques and which component "
        "changes they require.",
        "Explain the difference between common-mode and differential-mode EMI in a "
        "SMPS, and describe the role of X-capacitors versus Y-capacitors in each case.",
        "A PCB has a split ground plane. Trace the return current path at 10 MHz and "
        "explain why slots and gaps in the ground plane increase radiated emissions.",
        "What is the 20H rule in PCB design for EMC? Derive the 20H distance for a "
        "power plane at 3.3V, assuming the power plane edge acts as a monopole "
        "antenna with the dielectric constant of FR4 (εr=4.5).",
    ],
    "stm32": [
        "Write STM32 HAL C code to configure TIM2 as a 32-bit up-counter at 1 MHz "
        "with an output compare on channel 1 to toggle PA0 every 500 ms.",
        "On an STM32H7 with dual-bank flash, write the sequence to perform an "
        "in-application programming (IAP) update of bank 2 while executing from "
        "bank 1, using HAL_FLASH_Program().",
        "Explain the STM32 clock tree: HSI, HSE, PLL, SYSCLK, HCLK, APB1, APB2. "
        "Write the CubeMX-equivalent C initialisation for SYSCLK=480 MHz on H743.",
        "Write a DMA-based UART receive on STM32F4 using circular buffer mode. "
        "Show the NVIC priority configuration and the idle-line detection interrupt "
        "used to delimit packets.",
        "In STM32 FreeRTOS, explain the priority inversion problem and how "
        "priority inheritance is configured via `configUSE_MUTEXES` and "
        "`configUSE_RECURSIVE_MUTEXES`.",
    ],
    "embedded": [
        "Write an interrupt-safe ring buffer in C for bare-metal ARM Cortex-M, "
        "ensuring head/tail pointer updates are atomic using `__disable_irq()` "
        "and `__enable_irq()` critical sections.",
        "Explain the startup sequence of a bare-metal ARM Cortex-M4: from reset "
        "vector fetch through stack pointer load, `.data` copy, `.bss` zero-fill, "
        "to `main()` call.",
        "A Cortex-M hard fault handler must preserve crash context. Write the "
        "assembly stub that saves EXC_RETURN, r0-r3, r12, lr, pc, xpsr into a "
        "struct and calls a C fault logger.",
        "Compare I²C, SPI, and UART protocols for a sensor hub application: "
        "arbitration, clock stretching, full-duplex capability, and typical "
        "throughput on an embedded MCU at 72 MHz.",
        "Write a linker script fragment for an STM32F4 that places `.fastcode` "
        "section in CCMRAM at 0x10000000 and `.data` in SRAM at 0x20000000.",
    ],
    "freecad": [
        "Write a FreeCAD Python macro that creates a parametric L-bracket: "
        "width W, height H, thickness T, hole_diameter D, with all dimensions "
        "as named parameters in the `App.ActiveDocument`.",
        "Explain FreeCAD's topological naming problem: why does renaming a sketch "
        "edge break downstream features, and what is the TNP-fix approach in "
        "FreeCAD 1.0 (ElementMap)?",
        "Using FreeCAD's Part workbench Python API, write code to Boolean-subtract "
        "a cylinder from a box, then export the result to STEP format.",
        "In FreeCAD Sketcher, what are the constraint degrees of freedom, and how "
        "does the solver count DoF to determine if a sketch is fully constrained?",
        "Write a FreeCAD Path (CAM) operation in Python that generates a 2.5D "
        "pocket toolpath for a rectangular pocket 40x20x5 mm using a 6 mm end mill.",
    ],
    "platformio": [
        "Write a `platformio.ini` for an ESP32-S3 project that builds both a "
        "`release` and `debug` environment, with `-DDEBUG_LEVEL=3` only in debug, "
        "and uses Unity for unit tests in a `test/` directory.",
        "In PlatformIO, how does the `lib_deps` directive resolve version conflicts "
        "when two libraries depend on different versions of the same dependency? "
        "What is the `lib_compat_mode` option?",
        "Write a PlatformIO custom upload script (`upload.py`) that flashes an "
        "ESP32 over the air using `esptool.py` HTTP OTA instead of serial.",
        "Explain the PlatformIO build system's `SConscript` integration: how do "
        "you add a custom build step that runs `protoc` to generate `.pb.c/.pb.h` "
        "from `.proto` files before compilation?",
        "A PlatformIO project uses FreeRTOS on ESP-IDF. How do you configure "
        "`CONFIG_FREERTOS_HZ`, stack sizes, and heap allocator via "
        "`sdkconfig.defaults` and `board_build.cmake_extra_args`?",
    ],
    "power": [
        "Design a synchronous buck converter for 12V→3.3V at 5A, 400 kHz. "
        "Calculate the inductor value, output capacitor ESR requirement, and "
        "peak-to-peak ripple current.",
        "A flyback converter has Np:Ns=8:1, Vin=48V, Vout=5V, Iout=2A, f=100kHz. "
        "Calculate the transformer AL value needed, peak flux density, and duty cycle.",
        "Explain the differences between CCM, DCM, and BCM in a boost converter: "
        "which mode has lower peak current, which has simpler control loop?",
        "Write the small-signal transfer function (control-to-output) for a "
        "voltage-mode buck converter, including ESR zero. Derive the phase margin "
        "condition for a Type-II compensator.",
        "A LDO regulator has Vdropout=200mV, Vin=5V, Vout=3.3V, Iout=500mA. "
        "Calculate the power dissipation and junction temperature if θJA=40°C/W "
        "and ambient is 25°C. Is thermal shutdown a concern?",
    ],
    "dsp": [
        "Implement a 4th-order Butterworth low-pass filter in direct form II "
        "transposed (biquad cascade) in C, with fixed-point Q15 arithmetic and "
        "saturation on overflow.",
        "Explain the Goertzel algorithm: write a C function that detects the "
        "presence of a 1209 Hz DTMF tone in a 8000 Hz PCM buffer without FFT.",
        "A signal at 440 Hz is sampled at 44100 Hz. Write a C function that "
        "computes the real FFT using the Cooley-Tukey radix-2 DIT algorithm "
        "and returns the magnitude of the bin closest to 440 Hz.",
        "What is the difference between FIR and IIR filters in terms of phase "
        "linearity, stability, and computational cost? Give an example of when "
        "to prefer each in an audio embedded application.",
        "Design a 61-tap symmetric FIR bandpass filter for 300–3400 Hz at "
        "8 kHz sample rate using the window method (Hamming). Write the Python "
        "scipy.signal code and verify the frequency response.",
    ],
    "kicad-pcb": [
        "In KiCad PCB layout, explain the Design Rules Check (DRC) categories: "
        "clearance, short circuit, unconnected, silk-to-pad. How do you configure "
        "net-class specific clearances for high-voltage nets?",
        "Write a KiCad action plugin (Python) that iterates all copper pads on "
        "F.Cu, groups them by net name, and reports the total copper area per net.",
        "A 4-layer stackup is: F.Cu (signal), In1.Cu (GND), In2.Cu (PWR), B.Cu "
        "(signal). For a 100 Ω differential pair on F.Cu, calculate the trace "
        "width and gap for FR4 (εr=4.5, substrate height 0.2mm).",
        "Explain via stitching for EMC in KiCad: when and where to place stitching "
        "vias, recommended spacing formula relative to wavelength, and how to "
        "automate placement with a KiCad Python script.",
        "What is a footprint `courtyard` layer in KiCad and how does the DRC use "
        "it? Write a Python scripting snippet that finds all courtyard overlaps "
        "programmatically without running the full DRC.",
    ],
    "electronics": [
        "A common-emitter BJT amplifier has Vcc=12V, RC=4.7kΩ, RE=1kΩ, "
        "β=100. Calculate the Q-point (ICQ, VCEQ) and small-signal voltage gain.",
        "Explain the Miller effect in a MOSFET amplifier: how does Cgd create an "
        "effective input capacitance Miller_C = Cgd*(1+Av), and how does it limit "
        "bandwidth in a cascode topology?",
        "An RC charge pump doubles 3.3V to 6.6V at 1 mA. Calculate the flying "
        "capacitor value, output ripple, and efficiency loss due to switch "
        "resistance Rsw=0.5Ω at 1 MHz switching frequency.",
        "Compare NMOS vs PMOS for high-side switching: gate drive requirements, "
        "body diode direction, on-resistance at same die size, and why NMOS is "
        "preferred with a bootstrap gate drive.",
        "A Wheatstone bridge uses four 10kΩ resistors. One arm changes to "
        "10.1kΩ (ΔR/R=1%). Calculate the differential output voltage for "
        "Vexcitation=5V and the sensitivity in mV/V/%.",
    ],
    "iot": [
        "Write an ESP-IDF C component that implements MQTT over TLS to AWS IoT Core, "
        "with X.509 certificate authentication stored in NVS flash, and automatic "
        "reconnect with exponential backoff.",
        "Design a LoRaWAN end-node for a soil moisture sensor: SF7, BW125, CR4/5, "
        "OTAA join. Calculate the time-on-air for a 20-byte payload and the "
        "maximum duty cycle constraint for EU868.",
        "Explain Thread vs Zigbee vs Z-Wave vs Bluetooth Mesh for a smart home "
        "mesh network: topology, frequency band, security model, and typical "
        "latency.",
        "Write a MicroPython script for an ESP32-C3 that reads a DHT22 sensor "
        "every 30 seconds and publishes JSON to an MQTT broker with QoS 1, "
        "using `umqtt.robust` with reconnect logic.",
        "A battery-powered IoT node uses deep sleep. Calculate the average current "
        "consumption for a 10-second wake period (active current 80 mA) and "
        "350-second sleep period (5 µA). Estimate battery life for 2000 mAh.",
    ],
    "lua-upy": [
        "Write a MicroPython class `I2CDevice` that wraps `machine.I2C` with "
        "read_register(reg_addr, n_bytes) and write_register(reg_addr, data) "
        "methods, including 7-bit address handling.",
        "In Lua (for an ESP32 NodeMCU), write a coroutine-based HTTP GET client "
        "using the `net` module that reads a JSON response and calls a callback "
        "with the parsed table.",
        "Explain MicroPython's `asyncio` (uasyncio) event loop on ESP32: how "
        "does it differ from CPython asyncio, and what are the constraints on "
        "task stack size and timer resolution?",
        "Write a Lua script for NodeMCU that reads an ADC (adc.read) every 100 ms, "
        "applies a simple 8-sample moving average, and publishes the result via MQTT.",
        "In MicroPython, explain `micropython.const()`, `@micropython.native`, "
        "and `@micropython.viper` decorators: what optimisations do they apply "
        "and what Python features do they restrict?",
    ],
    "music-audio": [
        "Implement a Karplus-Strong string synthesis algorithm in Python (NumPy): "
        "ring buffer size, attenuation filter, initial noise burst, and output "
        "sample rate conversion to 44100 Hz.",
        "Write a JSFX (REAPER) effect plugin that implements a stereo ping-pong "
        "delay with tempo sync, feedback, and wet/dry mix using the JSFX `slider` "
        "and `@sample` block.",
        "Explain the difference between sample-accurate and block-based audio "
        "processing in a VST plugin. How does JUCE's `processBlock()` callback "
        "relate to DAW buffer size and latency?",
        "A 44100 Hz WAV file has a spectral peak at 880 Hz. Write Python code "
        "(librosa) to pitch-shift it up by 7 semitones without changing duration, "
        "using phase vocoder.",
        "Design a simple 3-band equaliser using biquad IIR filters: low shelf at "
        "200 Hz, peaking EQ at 1 kHz (Q=1.4, gain=6 dB), high shelf at 8 kHz. "
        "Write the coefficient calculation in Python.",
    ],
    # ── KNOWN ────────────────────────────────────────────────────────────────
    "chat-fr": [
        "Explique de manière accessible la différence entre l'intelligence "
        "artificielle générative et discriminative, avec un exemple concret pour "
        "chaque type.",
        "Rédige un email professionnel pour demander un report de réunion en "
        "préservant un ton courtois et en proposant deux créneaux alternatifs.",
        "Comment expliquer le concept de 'prompt engineering' à un chef de projet "
        "non-technique ? Donne une analogie claire.",
        "Traduis et adapte culturellement cette phrase anglaise en français "
        "professionnel : 'Let's circle back on this offline and touch base EOD.'",
        "Résume en 3 points clés le principe du consensus de Nakamoto dans Bitcoin, "
        "en termes simples pour un public non-informaticien.",
    ],
    "reasoning": [
        "A snail climbs a 10-meter pole. Each day it climbs 3 meters, each night "
        "it slips back 2 meters. On which day does it reach the top? Show reasoning.",
        "Given: All A are B. Some B are C. No C are D. Can we conclude that some "
        "A are not D? Construct a validity proof or counterexample.",
        "You have 12 coins, one is counterfeit (lighter or heavier). With exactly "
        "3 weighings on a balance scale, identify the counterfeit coin and whether "
        "it is lighter or heavier. Describe the strategy.",
        "A factory produces widgets. Machine A produces 60% of output with 2% "
        "defect rate. Machine B produces 40% with 5% defect rate. A widget is "
        "defective — what is the probability it came from Machine B? (Bayes)",
        "Prove by induction that the sum of the first n odd numbers equals n². "
        "State the base case, inductive hypothesis, and inductive step clearly.",
    ],
    "python": [
        "Write a Python context manager that wraps a `sqlite3` connection: commits "
        "on success, rolls back on exception, and ensures the connection is closed.",
        "Implement a generic `LRUCache` class using `collections.OrderedDict` with "
        "`get(key)` and `put(key, value)` in O(1) time.",
        "Explain Python's GIL: why does it exist, when does it NOT prevent "
        "parallelism (I/O bound vs CPU bound), and how does `multiprocessing` "
        "work around it?",
        "Write a decorator `@retry(max_attempts=3, delay=1.0, exceptions=(IOError,))`"
        " that retries a function with exponential backoff on specified exceptions.",
        "Using `asyncio`, write a coroutine that fetches 10 URLs concurrently with "
        "`aiohttp`, collects results, and returns them sorted by response time.",
    ],
    "cpp": [
        "Implement a thread-safe `ObjectPool<T>` in C++17 using `std::mutex`, "
        "`std::condition_variable`, and move semantics, with `acquire()` blocking "
        "when the pool is empty.",
        "Explain the Rule of Five in C++11: copy constructor, copy assignment, "
        "move constructor, move assignment, destructor. When is it necessary vs "
        "the Rule of Zero?",
        "Write a variadic template `type_list` that supports `type_list::get<N>`, "
        "`type_list::size`, and `type_list::contains<T>` at compile time (C++17).",
        "In C++, what is undefined behaviour with signed integer overflow? How does "
        "`-fwrapv` change the semantics, and when should you use `__builtin_add_overflow`?",
        "Implement `std::optional<T>` from scratch in C++17 using aligned storage, "
        "with `value()`, `has_value()`, `emplace()`, and move/copy operations.",
    ],
    "math": [
        "Compute the eigenvalues and eigenvectors of the matrix [[3,-2],[1,0]] "
        "by hand. Verify orthogonality for a symmetric matrix case.",
        "Prove that √2 is irrational using proof by contradiction. State every "
        "step explicitly.",
        "Solve the differential equation dy/dx = y·sin(x) with initial condition "
        "y(0)=1. Express the general solution and the particular solution.",
        "Using the residue theorem, evaluate the contour integral of "
        "f(z)=1/(z²+1) over the positively oriented unit circle |z|=2.",
        "State and prove the Cauchy-Schwarz inequality for inner product spaces. "
        "Give a concrete application in probability theory (correlation bound).",
    ],
    "typescript": [
        "Write a TypeScript generic `Result<T, E>` type (discriminated union) with "
        "`ok()`, `err()`, `map()`, `flatMap()`, and `unwrapOr()` utility functions.",
        "Explain TypeScript's structural typing vs nominal typing: why does "
        "`{name: string}` satisfy `interface Named {name: string}` without "
        "explicit `implements`?",
        "Write a TypeScript decorator `@Memoize()` that caches method results "
        "based on serialised arguments, compatible with `experimentalDecorators`.",
        "In TypeScript 5, what are `const` type parameters and how do they differ "
        "from `readonly`? Give an example where `const T` prevents widening.",
        "Write a Zod schema for a nested API response with optional fields, arrays, "
        "and union types, then infer the TypeScript type from it.",
    ],
    "rust": [
        "Implement a lock-free MPSC queue in Rust using `std::sync::atomic` and "
        "`Arc<AtomicPtr<Node<T>>>`. Explain the memory ordering choices.",
        "Explain Rust's borrow checker rules: why can't you have a mutable "
        "reference while an immutable reference is alive? Give an example that "
        "compiles and one that doesn't.",
        "Write a Rust procedural macro `#[derive(Builder)]` that generates a "
        "builder pattern struct for any struct with named fields.",
        "What is the `Pin<P>` type and why is it needed for async/await? Write "
        "a custom `Future` implementation that uses `Pin<Box<dyn Future>>`.",
        "In Rust, explain the difference between `Box<dyn Trait>`, `impl Trait`, "
        "and `T: Trait` generics in terms of vtable dispatch and monomorphisation.",
    ],
    "html-css": [
        "Write a CSS custom property system for a design token hierarchy: "
        "brand colours, semantic colours, component tokens. Show how specificity "
        "cascading is used to theme a button component.",
        "Explain CSS `contain: layout paint style` property: what does each value "
        "isolate, when does it improve rendering performance?",
        "Implement a CSS-only accordion component using `<details>` and `<summary>` "
        "with smooth height animation via `@starting-style` and `transition`.",
        "Write accessible HTML for a data table with row/column headers, merged "
        "cells, and a caption. Include ARIA attributes needed for screen readers.",
        "In CSS Grid, explain `subgrid`: how does it allow nested grids to align "
        "to the parent grid tracks, and what is the browser support status?",
    ],
    "shell": [
        "Write a bash script that monitors a directory for new `.log` files, "
        "tails each new file, and sends an alert email if the word 'ERROR' appears, "
        "using `inotifywait` and `mail`.",
        "Explain the difference between `set -e`, `set -u`, `set -o pipefail`, "
        "and `set -x` in bash. Write a script header that enables safe defaults.",
        "Write a POSIX-compatible shell function `retry(n, delay, cmd)` that "
        "re-runs `cmd` up to `n` times with `delay` seconds between attempts.",
        "In bash, how do you process a CSV file with quoted fields containing "
        "commas? Write a solution using `awk` or pure bash that handles edge cases.",
        "Write a shell pipeline that counts the 10 most frequent words in a text "
        "file, excluding stop words from a list file, sorted by frequency.",
    ],
    "sql": [
        "Write a SQL window function query that, for each customer, returns their "
        "last 3 orders and a running total of order value, using `ROWS BETWEEN`.",
        "Explain the difference between `INNER JOIN`, `LEFT JOIN`, `FULL OUTER JOIN`, "
        "and `CROSS JOIN` with examples showing when each produces different rows.",
        "Write a PostgreSQL CTE that recursively traverses an employee hierarchy "
        "(manager_id self-reference) to compute depth and full path for each node.",
        "Design a normalised schema (3NF) for an e-commerce system with products, "
        "orders, customers, and line items. Identify primary keys, foreign keys, "
        "and explain why it satisfies 3NF.",
        "Explain the `EXPLAIN ANALYZE` output in PostgreSQL: what do `Seq Scan`, "
        "`Index Scan`, `Hash Join`, and `actual rows` vs `estimated rows` tell you "
        "about query performance?",
    ],
    "yaml-json": [
        "Write a JSON Schema that validates a CI/CD pipeline config with required "
        "fields `name`, `on` (object), `jobs` (object with at least one key), "
        "and optional `env` (string→string map).",
        "Explain YAML anchors (`&`), aliases (`*`), and merge keys (`<<:`): "
        "write an example with a base job definition reused across 3 environments.",
        "Write a `jq` filter that, given a JSON array of objects with `name` and "
        "`scores` (array of ints), outputs each name with average score, sorted "
        "descending.",
        "What are the YAML 1.1 vs 1.2 specification differences around boolean "
        "literals (`yes`, `on`, `true`) and how do parsers like PyYAML handle them?",
        "Write a JSON Pointer (RFC 6901) path and a JSON Patch (RFC 6902) document "
        "that adds a `timeout: 30` field to the first element of a `jobs` array.",
    ],
    "docker": [
        "Write a multi-stage Dockerfile for a Python FastAPI app: build stage "
        "installs deps with `uv`, final stage is `python:3.12-slim`, non-root user, "
        "and health check via `/health` endpoint.",
        "Explain Docker's overlay2 storage driver: how do layers stack, what is a "
        "whiteout file, and why does `docker history` show intermediate layer sizes?",
        "Write a Docker Compose file for a 3-service app (FastAPI + PostgreSQL + "
        "Redis) with named volumes, health checks, and a custom network with "
        "subnet `172.20.0.0/24`.",
        "What is the difference between `CMD` and `ENTRYPOINT` in a Dockerfile? "
        "Give an example where `ENTRYPOINT` + `CMD` enables both default behaviour "
        "and override.",
        "Explain Docker BuildKit cache mounts (`--mount=type=cache`): how do they "
        "differ from layer caching, and write a Dockerfile that uses them for "
        "pip/uv install caching.",
    ],
    "web-frontend": [
        "Write a React 19 component that uses the new `use()` hook to consume a "
        "Promise inside Suspense, with an error boundary and loading fallback.",
        "Explain React's reconciliation algorithm (Fiber): how does key-based "
        "reconciliation prevent unnecessary unmounting in lists?",
        "Implement a custom React hook `useDebounce(value, delay)` and explain "
        "why the cleanup function in `useEffect` is critical for correctness.",
        "In a Vite + React TypeScript project, how do you configure absolute "
        "imports (`@/components/...`) in both `tsconfig.json` and `vite.config.ts`?",
        "Write a CSS-in-JS (styled-components) theme provider that supports "
        "dark/light mode via `prefers-color-scheme` with manual override stored "
        "in `localStorage`.",
    ],
    "web-backend": [
        "Write a FastAPI endpoint with dependency injection for database sessions, "
        "JWT authentication middleware, and a background task that sends an email "
        "after request completion.",
        "Explain the N+1 query problem in SQLAlchemy ORM and how `selectinload()` "
        "vs `joinedload()` solve it differently.",
        "Implement a rate limiter middleware for a Hono (TypeScript) API using "
        "Redis `INCR` + `EXPIRE`, sliding window algorithm, with per-IP limiting.",
        "In a Node.js Express app, write global error handling middleware that "
        "distinguishes operational errors (4xx) from programmer errors (5xx) and "
        "never leaks stack traces to clients.",
        "Explain HTTP/2 server push vs HTTP/2 multiplexing: when is push still "
        "useful in 2025 and how does it interact with browser caching?",
    ],
    "devops": [
        "Write a GitHub Actions workflow that builds and pushes a Docker image "
        "to GHCR on `main`, tags with the commit SHA, and deploys to a VPS via "
        "SSH only if tests pass.",
        "In Prometheus + Grafana, write a PromQL query that alerts when the "
        "95th-percentile request latency exceeds 500ms for 5 minutes, with "
        "appropriate `for` and `labels` in the alert rule.",
        "Explain the difference between blue-green deployment and canary deployment: "
        "rollback strategy, traffic routing mechanism, and database migration risks.",
        "Write a Terraform resource block for an AWS ECS Fargate service with "
        "auto-scaling based on CPU utilisation, VPC networking, and a load balancer.",
        "What is `etcd` in Kubernetes and why is it critical? Explain the Raft "
        "consensus algorithm at a high level and what happens during a leader "
        "election when a node fails.",
    ],
    "llm-orch": [
        "Write a Python class `LLMRouter` that takes a list of provider configs, "
        "routes requests to the cheapest model that can handle the context length, "
        "and falls back on timeout using `asyncio.wait_for`.",
        "Explain the ReAct (Reason + Act) prompting pattern: how does it differ "
        "from chain-of-thought, and what is the tool-call loop mechanism?",
        "Write a LangChain-style agent that uses a vector store for retrieval, "
        "a calculator tool, and a web search tool, with structured output parsing.",
        "Describe semantic caching for LLM responses: how do you compute "
        "similarity thresholds, what are the risks of cache poisoning, and "
        "how does it differ from exact-match caching?",
        "In a multi-agent system, explain the difference between orchestrator-worker "
        "and peer-to-peer topologies, with examples of when each is preferable.",
    ],
    "security": [
        "Explain SQL injection (UNION-based) step by step: how does the attacker "
        "discover column count, what payload extracts `information_schema.tables`, "
        "and how does parameterised query prevent it?",
        "Write a Python function that implements constant-time string comparison "
        "to prevent timing attacks, and explain why `==` is vulnerable.",
        "In JWT authentication, what is the `alg:none` attack and how does "
        "signature verification prevent it? Write the secure verification code.",
        "Explain CSRF: how does a cross-origin form submission exploit session "
        "cookies, and what are the differences between `SameSite=Lax` and "
        "`SameSite=Strict` as mitigations?",
        "Write a Content Security Policy header that allows scripts from the same "
        "origin and a CDN, blocks inline scripts, and prevents clickjacking. "
        "Explain each directive.",
    ],
}

# ---------------------------------------------------------------------------
# Judge prompt template
# ---------------------------------------------------------------------------

JUDGE_PROMPT_TEMPLATE = """\
You are an expert technical evaluator. Score the following model response.

## Domain
{domain}

## Question
{prompt}

## Response to evaluate
{response}

## Scoring rubric
Score each dimension from 0 to 10:

1. **Accuracy** (0-10): Is the technical content correct? Are formulas, API names,
   protocol details, and code syntactically and semantically right?
   0 = completely wrong  |  5 = partially correct with errors  |  10 = fully correct

2. **Completeness** (0-10): Does the response address all parts of the question?
   0 = ignores question  |  5 = covers half  |  10 = fully addresses all sub-parts

3. **Usefulness** (0-10): Would a working engineer find this actionable?
   0 = useless  |  5 = useful but vague  |  10 = immediately actionable

## Output format (JSON only, no prose):
{{
  "accuracy": <int 0-10>,
  "completeness": <int 0-10>,
  "usefulness": <int 0-10>,
  "overall": <float, weighted mean: accuracy*0.5 + completeness*0.3 + usefulness*0.2>,
  "notes": "<one sentence summarising main quality issue or strength>"
}}
"""

# Score threshold below which a domain is classified as "niche" (LoRA needed)
NICHE_THRESHOLD = 7.0  # overall score out of 10

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PromptResult:
    domain: str
    prompt_idx: int
    prompt: str
    response: str
    scores: dict[str, float]
    status: str  # "mock" | "real"


@dataclass
class DomainResult:
    domain: str
    hypothesis: str          # "niche" | "known" from DOMAIN_CLASS
    verdict: str             # "niche" | "known" — derived from actual scores
    avg_score: float
    prompt_results: list[PromptResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Inference stub — replace with real call to enable live benchmarking
# ---------------------------------------------------------------------------


def _infer_stub(model: str, prompt: str, domain: str) -> str:
    """Stubbed inference: returns a placeholder response without loading the model.

    Replace this function with a real call to mlx_lm, vLLM, or the OpenAI
    API to run the benchmark against an actual model.
    """
    seed = (sum(ord(c) for c in domain + prompt[:20]) % 100) / 100.0
    logger.debug(
        "STUB inference for domain=%s model=%s prompt_len=%d",
        domain, model, len(prompt),
    )
    return (
        f"[STUB response for domain '{domain}'] "
        f"This is a placeholder. Replace _infer_stub() with a real inference "
        f"call to enable benchmarking. Deterministic seed={seed:.4f}."
    )


def _judge_stub(domain: str, prompt: str, response: str) -> dict[str, float]:
    """Stubbed judge scoring: returns deterministic fake scores.

    Replace with a real call to an LLM judge (e.g. Qwen3.5-35B-A3B via
    JUDGE_PROMPT_TEMPLATE) to enable real evaluation.
    """
    seed = sum(ord(c) for c in domain + prompt[:10]) % 10
    # Niche domains get lower simulated scores to validate the hypothesis
    base = 5.0 if DOMAIN_CLASS.get(domain) == "niche" else 8.0
    jitter = (seed % 3) * 0.3
    accuracy = min(10.0, base + jitter)
    completeness = min(10.0, base - jitter * 0.5)
    usefulness = min(10.0, base + jitter * 0.2)
    overall = accuracy * 0.5 + completeness * 0.3 + usefulness * 0.2
    logger.debug("STUB judge for domain=%s overall=%.2f", domain, overall)
    return {
        "accuracy": round(accuracy, 2),
        "completeness": round(completeness, 2),
        "usefulness": round(usefulness, 2),
        "overall": round(overall, 2),
        "notes": "[stub] deterministic fake score — replace judge stub for real eval",
    }


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------


def eval_domain(
    domain: str,
    model: str,
    prompts: list[str],
) -> DomainResult:
    """Evaluate a single domain: infer + judge for each prompt."""
    prompt_results: list[PromptResult] = []

    for idx, prompt in enumerate(prompts):
        logger.info(
            "  [%s] prompt %d/%d",
            domain, idx + 1, len(prompts),
        )
        response = _infer_stub(model, prompt, domain)
        scores = _judge_stub(domain, prompt, response)
        prompt_results.append(
            PromptResult(
                domain=domain,
                prompt_idx=idx,
                prompt=prompt,
                response=response,
                scores=scores,
                status="mock",
            )
        )

    avg_score = (
        sum(r.scores["overall"] for r in prompt_results) / len(prompt_results)
        if prompt_results
        else 0.0
    )
    verdict = "known" if avg_score >= NICHE_THRESHOLD else "niche"

    return DomainResult(
        domain=domain,
        hypothesis=DOMAIN_CLASS.get(domain, "unknown"),
        verdict=verdict,
        avg_score=round(avg_score, 3),
        prompt_results=prompt_results,
    )


def run_benchmark(
    model: str,
    domains: list[str] | None,
) -> list[DomainResult]:
    """Run full benchmark across requested domains."""
    target_domains = domains if domains else sorted(DOMAIN_PROMPTS.keys())
    unknown = [d for d in target_domains if d not in DOMAIN_PROMPTS]
    if unknown:
        raise ValueError(f"Unknown domains: {unknown}. Available: {sorted(DOMAIN_PROMPTS)}")

    results: list[DomainResult] = []
    for domain in target_domains:
        logger.info("Evaluating domain: %s (%s)", domain, DOMAIN_CLASS.get(domain))
        result = eval_domain(domain, model, DOMAIN_PROMPTS[domain])
        results.append(result)
        logger.info(
            "  → avg_score=%.2f  hypothesis=%s  verdict=%s  match=%s",
            result.avg_score,
            result.hypothesis,
            result.verdict,
            "✓" if result.hypothesis == result.verdict else "✗",
        )

    return results


# ---------------------------------------------------------------------------
# Report serialisation
# ---------------------------------------------------------------------------


def build_report(model: str, results: list[DomainResult]) -> dict:
    """Aggregate results into a JSON-serialisable report dict."""
    niche_results = [r for r in results if r.verdict == "niche"]
    known_results = [r for r in results if r.verdict == "known"]

    # Hypothesis accuracy: how well do real scores match our a-priori labels?
    correct = sum(1 for r in results if r.hypothesis == r.verdict)
    total = len(results)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "mode": "scaffold_stub",
        "niche_threshold": NICHE_THRESHOLD,
        "domains_evaluated": total,
        "niche_count": len(niche_results),
        "known_count": len(known_results),
        "hypothesis_accuracy": round(correct / total, 3) if total else 0.0,
        "niche_domains": sorted(r.domain for r in niche_results),
        "known_domains": sorted(r.domain for r in known_results),
        "hypothesis_mismatches": [
            {"domain": r.domain, "hypothesis": r.hypothesis, "verdict": r.verdict,
             "avg_score": r.avg_score}
            for r in results if r.hypothesis != r.verdict
        ],
    }

    domain_details = {
        r.domain: {
            "hypothesis": r.hypothesis,
            "verdict": r.verdict,
            "avg_score": r.avg_score,
            "prompts": [
                {
                    "idx": pr.prompt_idx,
                    "prompt": pr.prompt[:120] + "..." if len(pr.prompt) > 120 else pr.prompt,
                    "scores": pr.scores,
                    "status": pr.status,
                }
                for pr in r.prompt_results
            ],
        }
        for r in results
    }

    return {"summary": summary, "domains": domain_details}


def save_report(report: dict, output: str) -> Path:
    """Write report as pretty-printed JSON."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info("Report saved → %s", path)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Qwen3.5-35B-A3B base model across 32 domains to validate "
            "which require a LoRA adapter (niche) vs which the base handles well "
            "(known). Inference is stubbed — see _infer_stub()."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Available domains:\n  "
            + "\n  ".join(sorted(DOMAIN_PROMPTS.keys()))
        ),
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-35B-A3B",
        help="Model path or HF repo ID (default: Qwen/Qwen3.5-35B-A3B)",
    )
    parser.add_argument(
        "--output",
        default="results/base-benchmark.json",
        help="Output JSON path (default: results/base-benchmark.json)",
    )
    parser.add_argument(
        "--domains",
        nargs="*",
        metavar="DOMAIN",
        help=(
            "Optional subset of domains to evaluate (space-separated). "
            "Defaults to all 32 domains if omitted."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stub determinism (default: 42)",
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
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    random.seed(args.seed)

    logger.info(
        "Starting benchmark: model=%s domains=%s output=%s",
        args.model,
        args.domains or "all",
        args.output,
    )
    logger.warning(
        "Running in SCAFFOLD mode — inference is stubbed. "
        "Replace _infer_stub() and _judge_stub() for real evaluation."
    )

    results = run_benchmark(model=args.model, domains=args.domains)
    report = build_report(model=args.model, results=results)
    save_report(report, args.output)

    summary = report["summary"]
    logger.info(
        "Done: %d domains | niche=%d known=%d | hypothesis_accuracy=%.1f%%",
        summary["domains_evaluated"],
        summary["niche_count"],
        summary["known_count"],
        summary["hypothesis_accuracy"] * 100,
    )


if __name__ == "__main__":
    main()

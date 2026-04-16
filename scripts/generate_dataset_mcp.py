#!/usr/bin/env python3
"""Generate high-quality training datasets using Claude CLI + MCP servers.

Uses Claude Code with MCP tools (ESP-IDF, KiCad, SPICE) to generate
contextual, tool-augmented training data for micro-kiki niche domains.

Each domain has a set of task prompts that trigger MCP tool usage.
Claude's responses (including tool results) are captured as training data.

Usage:
    python3 scripts/generate_dataset_mcp.py --domain kicad-dsl --max 50
    python3 scripts/generate_dataset_mcp.py --all --max 20
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

os.chdir("/Users/clems/micro-kiki")
OUTPUT_ROOT = Path("data/mcp-generated")
CLAUDE = "/Users/clems/.local/bin/claude"

# Domain-specific task prompts that trigger MCP tool usage
DOMAIN_PROMPTS = {
    "kicad-dsl": [
        "Create a KiCad schematic for a USB-C power delivery circuit with a FUSB302 controller. Show the S-expression for the schematic.",
        "Design a KiCad footprint for a QFN-48 package with thermal pad. Output the S-expression.",
        "Write a KiCad DRC rule file that enforces 0.15mm minimum trace width and 0.2mm clearance for a 4-layer PCB.",
        "Create a KiCad symbol for a buck converter (TPS54331) with all pins correctly labeled. Show the lib_symbol S-expression.",
        "Generate a KiCad netlist for a differential pair USB 2.0 connection with ESD protection (TPD2E2U06).",
        "Design a KiCad schematic for an I2C level shifter circuit using BSS138 MOSFETs. Include pull-up resistors.",
        "Create a KiCad BOM export script that groups components by value and outputs CSV format.",
        "Write a KiCad zone fill rule for a ground plane with thermal relief on through-hole pads.",
        "Design a KiCad schematic for a 3.3V LDO power supply (AMS1117-3.3) with input/output capacitors.",
        "Create a KiCad hierarchical schematic structure for a sensor node with MCU, power, and communication blocks.",
        "Generate a KiCad footprint for a 2512 package power resistor with wide traces for high current.",
        "Write a KiCad electrical rules check configuration for automotive (AEC-Q100) design requirements.",
        "Design a KiCad schematic for an SPI flash memory (W25Q128) connected to an ESP32-S3.",
        "Create a KiCad PCB stackup definition for a 4-layer impedance-controlled board (50 ohm single-ended).",
        "Generate a KiCad schematic for a CAN bus transceiver (MCP2551) with termination resistor.",
        "Design a complete KiCad schematic for a battery management system with BQ25895 charger IC.",
        "Write KiCad design rules for a high-speed DDR3 memory layout including length matching constraints.",
        "Create a KiCad schematic for an analog front-end with instrumentation amplifier (INA128) and ADC.",
        "Generate a KiCad power distribution network showing decoupling capacitor placement strategy.",
        "Design a KiCad schematic for a LoRa module (SX1276) with antenna matching network.",
    ],
    "spice": [
        "Write a SPICE netlist for a Class-D audio amplifier with LC output filter and feedback loop.",
        "Create an ngspice Monte Carlo analysis for a bandgap voltage reference with 1% component tolerance.",
        "Design a SPICE model for a flyback transformer with leakage inductance and coupling coefficient.",
        "Write a SPICE transient simulation for a synchronous buck converter at 500kHz switching frequency.",
        "Create a SPICE netlist for a 4th-order Butterworth active low-pass filter using Sallen-Key topology.",
        "Design a SPICE subcircuit for a MOSFET H-bridge motor driver with dead-time control.",
        "Write ngspice commands for AC analysis of a common-emitter amplifier from 1Hz to 100MHz.",
        "Create a SPICE model for a solar cell with series and shunt resistance parameters.",
        "Design a SPICE simulation for a charge pump voltage doubler with flying capacitor.",
        "Write a SPICE netlist for an LC tank oscillator (Colpitts) at 433MHz for RF applications.",
        "Create a SPICE parametric sweep analyzing the effect of load resistance on LDO dropout voltage.",
        "Design a SPICE transient simulation for a PLL frequency synthesizer phase acquisition.",
        "Write ngspice commands for noise analysis of a low-noise preamplifier with JFET input stage.",
        "Create a SPICE model for a piezoelectric sensor with mechanical-electrical coupling.",
        "Design a SPICE simulation comparing Schottky vs silicon diode rectifier efficiency.",
        "Write a SPICE netlist for a Wien bridge oscillator with AGC using JFET limiter.",
        "Create a comprehensive SPICE model for a lithium-ion battery cell with impedance matching.",
        "Design a SPICE simulation for EMI filter characterization with common-mode choke.",
        "Write ngspice commands for worst-case analysis of an op-amp offset voltage circuit.",
        "Create a SPICE netlist for a current-mode PWM controller with slope compensation.",
    ],
    "emc": [
        "Design a common-mode filter for USB 3.0 that meets CISPR 32 Class B conducted emissions.",
        "Calculate the shielding effectiveness needed for a 1GHz clock signal in an aluminum enclosure.",
        "Write the grounding strategy for a mixed-signal PCB with separate analog and digital grounds.",
        "Design a Pi-filter network for DC power input compliance with MIL-STD-461G CE102.",
        "Explain the PCB layout rules for minimizing radiated emissions from a 100MHz differential clock.",
        "Design a ferrite bead selection guide for a 3.3V power rail with 2A current requirement.",
        "Write pre-compliance test procedure for conducted emissions using a LISN and spectrum analyzer.",
        "Design a PCB ground plane stitching strategy for a 6-layer board with split power planes.",
        "Calculate the return current path for a high-speed signal crossing a plane split.",
        "Design an ESD protection scheme for exposed I/O connectors meeting IEC 61000-4-2 Level 4.",
        "Write the EMC design review checklist for an automotive ECU (CISPR 25 Class 5).",
        "Design a cable shield grounding strategy for a shielded Ethernet connection.",
        "Calculate the resonant frequency of a decoupling capacitor network on a PCB power plane.",
        "Design a spread-spectrum clock modulation profile to reduce peak conducted emissions.",
        "Write the test setup for radiated immunity testing per IEC 61000-4-3 at 10V/m.",
    ],
    "stm32": [
        "Write STM32 HAL code for configuring a timer in PWM mode on TIM1 channel 1 at 20kHz.",
        "Create STM32 CubeMX-style initialization code for SPI1 in full-duplex master mode at 10MHz.",
        "Write STM32 HAL code for DMA-based ADC conversion on 4 channels with circular buffer.",
        "Design an STM32 interrupt priority scheme for a system with UART, SPI, Timer, and external IRQ.",
        "Write STM32 HAL code for I2C communication with a BME280 temperature sensor.",
        "Create STM32 low-power mode configuration using STOP2 mode with RTC wakeup.",
        "Write STM32 HAL code for USB CDC device implementation for serial communication.",
        "Design an STM32 bootloader that updates firmware via UART with CRC32 verification.",
        "Write STM32 HAL code for CAN bus communication with filtering for specific message IDs.",
        "Create STM32 clock configuration for STM32H743 at 480MHz with PLL cascade.",
        "Write STM32 HAL code for hardware watchdog (IWDG) with 2-second timeout.",
        "Design an STM32 RTOS task structure using FreeRTOS for a sensor data acquisition system.",
        "Write STM32 HAL code for external flash memory (QSPI) memory-mapped mode.",
        "Create STM32 MPU configuration for protecting critical memory regions in a safety application.",
        "Write STM32 HAL code for generating a sine wave using DAC with DMA and timer trigger.",
    ],
    "embedded": [
        "Write a bare-metal SPI driver for a MAX31855 thermocouple interface on ARM Cortex-M4.",
        "Design a circular buffer implementation in C for a UART receive interrupt handler.",
        "Write a FreeRTOS task synchronization pattern using semaphores for producer-consumer.",
        "Create a memory-mapped I/O register abstraction layer for a custom peripheral in C.",
        "Write an interrupt-safe ring buffer for passing data between ISR and main loop.",
        "Design a power management state machine for a battery-powered IoT sensor node.",
        "Write a DMA transfer completion handler with double-buffering for ADC data acquisition.",
        "Create a software timer wheel implementation for managing multiple timeouts in embedded C.",
        "Write a HAL abstraction for GPIO that works across STM32, ESP32, and nRF52 platforms.",
        "Design a CRC-32 implementation optimized for ARM Cortex-M using hardware acceleration.",
        "Write a bootloader protocol for firmware update over BLE with integrity verification.",
        "Create a real-time data logging system using SD card with wear leveling strategy.",
        "Write a PID controller implementation in fixed-point arithmetic for motor control.",
        "Design a fault-tolerant communication protocol for industrial sensor network (RS-485).",
        "Write an RTOS-aware memory allocator that prevents heap fragmentation in embedded systems.",
    ],
    "power": [
        "Design a synchronous buck converter for 12V to 3.3V at 5A using TPS54560.",
        "Calculate the inductor and capacitor values for a boost converter from 3.7V to 5V at 2A.",
        "Design a MOSFET selection guide for a half-bridge inverter at 400V/10A switching at 100kHz.",
        "Write the thermal analysis for a linear regulator (LM7805) dissipating 5W in TO-220 package.",
        "Design a soft-start circuit for a high-side switch to limit inrush current.",
        "Calculate the loop compensation for a voltage-mode buck converter with Type III compensator.",
        "Design an isolated DC-DC converter using a flyback topology for medical equipment (4kV isolation).",
        "Write the MOSFET gate driver circuit design for a full-bridge inverter with bootstrap.",
        "Design a power supply sequencing circuit for an FPGA board with multiple voltage rails.",
        "Calculate efficiency and losses for a synchronous buck at various load conditions.",
    ],
    "dsp": [
        "Implement a 256-point FFT in fixed-point arithmetic optimized for ARM Cortex-M4F DSP.",
        "Design a digital PLL for frequency tracking with loop filter coefficients for 1kHz bandwidth.",
        "Write a Goertzel algorithm implementation for DTMF tone detection on embedded system.",
        "Design an adaptive noise cancellation filter using LMS algorithm for microphone array.",
        "Implement a cascaded biquad IIR filter with coefficient quantization analysis.",
        "Write a real-time audio processing pipeline with sample rate conversion (48kHz to 16kHz).",
        "Design a digital AGC (automatic gain control) for a software-defined radio receiver.",
        "Implement an FIR filter using ARM CMSIS-DSP library functions for efficient processing.",
        "Design a phase-locked loop demodulator for FSK communication on embedded platform.",
        "Write a spectral analysis tool using Welch's method for vibration monitoring.",
    ],
    "electronics": [
        "Design an instrumentation amplifier circuit using 3 op-amps with gain of 100 and CMRR > 80dB.",
        "Calculate the bias point and small-signal parameters for a common-emitter amplifier stage.",
        "Design a precision voltage reference circuit using LM399 with < 1ppm/°C temperature coefficient.",
        "Write the design equations for a Chebyshev Type I bandpass filter at 10.7MHz for IF stage.",
        "Design a current sense amplifier circuit for measuring 0-50A with INA240 and shunt resistor.",
        "Calculate the noise figure and sensitivity for a receiver front-end with LNA and mixer.",
        "Design a sample-and-hold circuit for 16-bit ADC with aperture jitter < 1ps.",
        "Write the analysis of a cascode amplifier stage for RF application at 2.4GHz.",
        "Design a programmable gain amplifier using MUX and resistor network for data acquisition.",
        "Calculate the thermal noise contribution of each component in a transimpedance amplifier.",
    ],
    "freecad": [
        "Write a FreeCAD Python macro to create a parametric enclosure box with rounded corners and mounting holes.",
        "Create a FreeCAD script for generating a heatsink with fins from thermal parameters.",
        "Write a FreeCAD macro to import a STEP file and add custom mounting brackets.",
        "Design a FreeCAD parametric model for a DIN rail mounting bracket.",
        "Create a FreeCAD script to generate a cable gland plate with multiple cutouts.",
    ],
    "platformio": [
        "Write a platformio.ini configuration for multi-environment build targeting ESP32, STM32, and nRF52.",
        "Create a PlatformIO custom build script that generates version headers from git tags.",
        "Write a PlatformIO test configuration for running Unity unit tests on both native and hardware.",
        "Design a PlatformIO library structure with proper library.json for a reusable sensor driver.",
        "Create a PlatformIO CI/CD pipeline configuration for GitHub Actions with hardware-in-the-loop.",
    ],
}


def generate_with_claude(prompt: str, domain: str) -> dict | None:
    """Run Claude CLI with the prompt and capture response."""
    system_prompt = (
        f"You are an expert in {domain}. Provide detailed, technically accurate "
        f"responses with complete code examples. Be thorough and precise."
    )

    try:
        result = subprocess.run(
            [CLAUDE, "--print", "-p", prompt],
            capture_output=True, text=True, timeout=120,
            cwd="/Users/clems/micro-kiki",
        )
        if result.returncode == 0 and result.stdout.strip():
            return {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": result.stdout.strip()},
                ],
                "domain": domain,
                "source": "claude-mcp-generated",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
    except subprocess.TimeoutExpired:
        logger.warning("Timeout for prompt: %s...", prompt[:50])
    except Exception as e:
        logger.warning("Error: %s", e)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate training data using Claude CLI + MCP.",
    )
    parser.add_argument("--domain", help="Single domain")
    parser.add_argument("--all", action="store_true", help="All domains")
    parser.add_argument("--max", type=int, default=20, help="Max prompts per domain")
    parser.add_argument("--dry-run", action="store_true", help="List prompts only")
    args = parser.parse_args()

    domains = [args.domain] if args.domain else (
        list(DOMAIN_PROMPTS.keys()) if args.all else []
    )
    if not domains:
        parser.print_help()
        return

    for domain in domains:
        prompts = DOMAIN_PROMPTS.get(domain, [])[:args.max]
        if not prompts:
            logger.warning("No prompts for %s", domain)
            continue

        if args.dry_run:
            logger.info("%s: %d prompts", domain, len(prompts))
            for p in prompts[:3]:
                logger.info("  → %s", p[:80])
            continue

        out_dir = OUTPUT_ROOT / domain
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "train.jsonl"

        existing = sum(1 for _ in open(out_file)) if out_file.exists() else 0
        logger.info("GENERATE %s: %d prompts (skip %d existing)", domain, len(prompts), existing)

        with open(out_file, "a") as f:
            for i, prompt in enumerate(prompts):
                if i < existing:
                    continue
                logger.info("%s %d/%d: %s...", domain, i + 1, len(prompts), prompt[:50])
                example = generate_with_claude(prompt, domain)
                if example:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
                    f.flush()
                else:
                    logger.warning("  FAILED")

        total = sum(1 for _ in open(out_file)) if out_file.exists() else 0
        logger.info("DONE %s: %d examples", domain, total)

    logger.info("ALL DONE")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Expand domain prompts x25 using parametric templates.

Takes existing prompts and generates variations by:
1. Swapping component values, part numbers, frequencies
2. Varying complexity (beginner → expert)
3. Combining sub-tasks into compound prompts
4. Adding constraints (cost, size, power, temperature)

Output: data/prompts-expanded/<domain>.jsonl
"""
from __future__ import annotations

import json
import random
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

random.seed(42)

# Component libraries for parametric variation
RESISTORS = ["100R", "1K", "4.7K", "10K", "47K", "100K", "1M"]
CAPACITORS = ["100pF", "1nF", "10nF", "100nF", "1uF", "10uF", "100uF", "470uF"]
VOLTAGES = ["1.8V", "2.5V", "3.3V", "5V", "12V", "24V", "48V"]
CURRENTS = ["100mA", "500mA", "1A", "2A", "5A", "10A", "20A"]
FREQUENCIES = ["1kHz", "10kHz", "100kHz", "500kHz", "1MHz", "10MHz", "100MHz", "1GHz"]
MCU_PARTS = ["STM32F103", "STM32F407", "STM32H743", "STM32L476", "STM32G431", "STM32U575"]
ESP_PARTS = ["ESP32", "ESP32-S2", "ESP32-S3", "ESP32-C3", "ESP32-C6", "ESP32-H2"]
INTERFACES = ["SPI", "I2C", "UART", "CAN", "USB", "Ethernet", "BLE", "WiFi", "LoRa"]
SENSORS = ["BME280", "MPU6050", "BMP390", "ADXL345", "MAX31855", "ADS1115", "INA226", "SHT40"]
REGULATORS = ["AMS1117", "TPS54560", "LM2596", "TPS63020", "BQ25895", "LTC3588", "MAX17048"]
CONNECTORS = ["USB-C", "USB-A", "RJ45", "SMA", "U.FL", "JST-PH", "Molex"]
PACKAGES = ["SOT-23", "QFN-32", "QFN-48", "QFP-64", "BGA-256", "TSSOP-20", "TO-220"]
PROTOCOLS = ["Modbus RTU", "MQTT", "CoAP", "HTTP REST", "gRPC", "Protobuf", "JSON-RPC"]
FILTER_TYPES = ["Butterworth", "Chebyshev", "Bessel", "Elliptic", "Sallen-Key", "MFB"]
TOPOLOGIES = ["buck", "boost", "buck-boost", "flyback", "forward", "SEPIC", "Cuk"]
EMC_STANDARDS = ["CISPR 32", "CISPR 25", "MIL-STD-461G", "IEC 61000-4-2", "EN 55032", "FCC Part 15"]
PCB_LAYERS = ["2-layer", "4-layer", "6-layer", "8-layer"]

KICAD_TEMPLATES = [
    "Create a KiCad schematic for a {voltage} power supply using {regulator} with {connector} input connector.",
    "Design a KiCad footprint for a {package} package with {pin_count} pins and thermal pad.",
    "Write a KiCad DRC rule file for a {layers} PCB with {clearance}mm minimum clearance.",
    "Create a KiCad symbol for {sensor} sensor with all pins labeled for {interface} communication.",
    "Generate a KiCad netlist for {interface} connection between {mcu} and {sensor}.",
    "Design a KiCad schematic for ESD protection on {connector} connector using TVS diodes.",
    "Create a KiCad hierarchical schematic for a {mcu}-based IoT node with power, MCU, and {interface} blocks.",
    "Write a KiCad PCB stackup definition for a {layers} impedance-controlled board.",
    "Design a KiCad schematic for a {interface} level shifter between {voltage_a} and {voltage_b} domains.",
    "Generate a KiCad BOM for a sensor board with {mcu}, {sensor}, {regulator}, and {connector}.",
    "Create a KiCad schematic for an {interface} bus with {device_count} devices and proper pull-ups.",
    "Design a KiCad power distribution network with {regulator} and decoupling for {mcu}.",
    "Write KiCad design rules for {frequency} high-speed {interface} routing on {layers} board.",
    "Create a KiCad schematic for battery management with {regulator} charger and fuel gauge.",
    "Generate a KiCad footprint for a custom transformer with {pin_count} pins and safety clearance.",
]

SPICE_TEMPLATES = [
    "Write a SPICE netlist for a {filter_type} {filter_order}-order low-pass filter at {frequency}.",
    "Create an ngspice {analysis} analysis for a {topology} converter at {frequency} switching frequency.",
    "Design a SPICE model for a {component} with parasitic elements and temperature dependence.",
    "Write a SPICE subcircuit for a {topology} power stage with {voltage_in} input and {voltage_out} output at {current}.",
    "Create a SPICE parametric sweep varying {component} from {value_min} to {value_max}.",
    "Design a SPICE simulation for {interface} signal integrity at {frequency} data rate.",
    "Write ngspice commands for noise analysis of a {gain}dB gain amplifier with {bandwidth} bandwidth.",
    "Create a SPICE Monte Carlo analysis with {tolerance}% component tolerance for a {circuit}.",
    "Design a SPICE transient simulation for a {topology} converter startup with soft-start.",
    "Write a SPICE netlist for a {filter_type} active bandpass filter centered at {frequency}.",
    "Simulate a {component} thermal runaway scenario in SPICE with temperature feedback loop.",
    "Create a SPICE model for a {connector} connector with impedance mismatch at {frequency}.",
    "Design a SPICE simulation comparing efficiency of {topology_a} vs {topology_b} at {power}W load.",
    "Write ngspice worst-case analysis for a {circuit} operating from {temp_min}C to {temp_max}C.",
    "Create a SPICE netlist for a PLL frequency synthesizer locking to {frequency} reference.",
]

EMC_TEMPLATES = [
    "Design a common-mode filter for {interface} that meets {standard} conducted emissions.",
    "Calculate shielding effectiveness needed for a {frequency} clock in a {material} enclosure.",
    "Write the grounding strategy for a {layers} mixed-signal PCB with {interface} high-speed signals.",
    "Design an EMI filter for {voltage} {current} DC power input targeting {standard}.",
    "Explain PCB layout rules for minimizing radiated emissions from a {frequency} {interface} signal.",
    "Design a ferrite bead selection for a {voltage} power rail with {current} requirement.",
    "Write a pre-compliance test procedure for {standard} using {test_equipment}.",
    "Calculate the return current path for a {interface} signal crossing a plane split on {layers} board.",
    "Design ESD protection for {connector} connector meeting {esd_standard} Level {level}.",
    "Write an EMC design review checklist for {application} targeting {standard}.",
    "Design a cable shield grounding strategy for {interface} connection at {frequency}.",
    "Calculate decoupling capacitor network resonance for a {layers} board power plane.",
    "Design spread-spectrum clock modulation to reduce peak emissions at {frequency}.",
    "Write the test setup for radiated immunity testing per {standard} at {field_strength}V/m.",
    "Analyze the effect of via stitching spacing on {layers} board EMC at {frequency}.",
]

STM32_TEMPLATES = [
    "Write {mcu} HAL code for configuring {peripheral} in {mode} mode at {frequency}.",
    "Create {mcu} initialization code for {interface} in {duplex} mode at {speed}.",
    "Write {mcu} HAL code for DMA-based {peripheral} with {channel_count} channels.",
    "Design an {mcu} interrupt priority scheme for {irq_list}.",
    "Write {mcu} HAL code for {interface} communication with {sensor} sensor.",
    "Create {mcu} low-power mode configuration using {lp_mode} with {wakeup_source} wakeup.",
    "Write {mcu} HAL code for {peripheral} with {timer} trigger at {frequency}.",
    "Design an {mcu} bootloader updating firmware via {interface} with CRC32 verification.",
    "Write {mcu} clock configuration for {clock_speed}MHz with PLL from {clock_source}.",
    "Create {mcu} {rtos} task structure for a {application} with {task_count} tasks.",
    "Write {mcu} HAL code for {memory_type} memory in {memory_mode} mode.",
    "Design {mcu} MPU configuration for protecting {region_count} memory regions.",
    "Write {mcu} HAL code for generating {waveform} wave using DAC with DMA.",
    "Create {mcu} {interface} driver with error handling and retry logic.",
    "Write {mcu} HAL code for hardware {watchdog} with {timeout}s timeout.",
]

EMBEDDED_TEMPLATES = [
    "Write a bare-metal {interface} driver for {sensor} on {architecture}.",
    "Design a {buffer_type} buffer implementation in C for {peripheral} {direction} handler.",
    "Write a {rtos} task synchronization pattern using {sync_primitive} for {pattern}.",
    "Create a memory-mapped I/O register abstraction for {peripheral} in C.",
    "Write an interrupt-safe {data_structure} for passing data between ISR and {context}.",
    "Design a power management state machine for a {powered_by} {device_type}.",
    "Write a DMA transfer handler with {buffering} for {peripheral} data acquisition.",
    "Create a {timer_type} timer implementation for managing {timer_count} timeouts.",
    "Write a HAL abstraction for {peripheral} that works across {platform_list}.",
    "Design a {algorithm} implementation optimized for {architecture} using {optimization}.",
    "Write a bootloader protocol for firmware update over {interface} with {verification}.",
    "Create a data logging system using {storage} with {wear_strategy}.",
    "Write a {controller_type} controller in {arithmetic} for {application}.",
    "Design a fault-tolerant communication protocol for {bus_type} {application}.",
    "Write an RTOS-aware {allocator_type} that prevents {problem} in embedded systems.",
]

POWER_TEMPLATES = [
    "Design a {topology} converter for {voltage_in} to {voltage_out} at {current} using {regulator}.",
    "Calculate inductor and capacitor values for a {topology} converter from {voltage_in} to {voltage_out} at {current}.",
    "Design MOSFET selection for a {bridge_type} at {voltage}/{current} switching at {frequency}.",
    "Write thermal analysis for a {regulator} dissipating {power}W in {package}.",
    "Design a soft-start circuit for a high-side switch limiting inrush to {current}.",
    "Calculate loop compensation for a {mode}-mode {topology} with {compensator} compensator.",
    "Design an isolated DC-DC converter using {topology} for {application} ({isolation}kV isolation).",
    "Design power supply sequencing for a board with {rail_count} voltage rails.",
    "Calculate efficiency and losses for a {topology} at {load_min}% to {load_max}% load.",
    "Design a {protection_type} protection circuit for {voltage} {current} power rail.",
]

DSP_TEMPLATES = [
    "Implement a {fft_size}-point FFT in {arithmetic} optimized for {architecture}.",
    "Design a digital {pll_type} for {application} with {bandwidth} bandwidth.",
    "Write a {algorithm} implementation for {application} on {platform}.",
    "Design an adaptive {filter_type} filter using {adaptation} algorithm for {application}.",
    "Implement a cascaded biquad IIR filter with {quantization} quantization analysis.",
    "Write a real-time audio processing pipeline with sample rate conversion ({rate_in}kHz to {rate_out}kHz).",
    "Design a digital {agc_type} for a {receiver_type} receiver.",
    "Implement a {filter_type} filter using {library} functions for {application}.",
    "Design a {modulation} demodulator for {protocol} on {platform}.",
    "Write a spectral analysis tool using {method} for {application}.",
]


def expand_kicad(n: int) -> list[str]:
    prompts = []
    for _ in range(n):
        t = random.choice(KICAD_TEMPLATES)
        prompts.append(t.format(
            voltage=random.choice(VOLTAGES),
            voltage_a=random.choice(["1.8V", "2.5V", "3.3V"]),
            voltage_b=random.choice(["3.3V", "5V", "12V"]),
            regulator=random.choice(REGULATORS),
            connector=random.choice(CONNECTORS),
            package=random.choice(PACKAGES),
            pin_count=random.choice([8, 16, 20, 32, 48, 64, 100]),
            layers=random.choice(PCB_LAYERS),
            clearance=random.choice([0.1, 0.15, 0.2, 0.25]),
            sensor=random.choice(SENSORS),
            interface=random.choice(INTERFACES),
            mcu=random.choice(MCU_PARTS + ESP_PARTS),
            frequency=random.choice(FREQUENCIES),
            device_count=random.choice([2, 4, 8, 16]),
        ))
    return prompts


def expand_spice(n: int) -> list[str]:
    prompts = []
    for _ in range(n):
        t = random.choice(SPICE_TEMPLATES)
        prompts.append(t.format(
            filter_type=random.choice(FILTER_TYPES),
            filter_order=random.choice([2, 3, 4, 6, 8]),
            frequency=random.choice(FREQUENCIES),
            topology=random.choice(TOPOLOGIES),
            topology_a=random.choice(TOPOLOGIES[:3]),
            topology_b=random.choice(TOPOLOGIES[3:]),
            analysis=random.choice(["transient", "AC", "DC sweep", "noise", "Monte Carlo"]),
            component=random.choice(["MOSFET", "BJT", "diode", "transformer", "inductor"]),
            voltage_in=random.choice(["5V", "12V", "24V", "48V"]),
            voltage_out=random.choice(["1.8V", "3.3V", "5V", "12V"]),
            current=random.choice(CURRENTS),
            value_min=random.choice(["1K", "100pF", "10nF"]),
            value_max=random.choice(["100K", "10nF", "1uF"]),
            tolerance=random.choice([1, 2, 5, 10]),
            circuit=random.choice(["voltage reference", "op-amp inverter", "LDO", "oscillator"]),
            gain=random.choice([10, 20, 40, 60]),
            bandwidth=random.choice(["1kHz", "10kHz", "100kHz", "1MHz"]),
            temp_min=random.choice([-40, -20, 0]),
            temp_max=random.choice([85, 105, 125]),
            power=random.choice([1, 5, 10, 50, 100]),
            connector=random.choice(CONNECTORS),
            interface=random.choice(INTERFACES),
        ))
    return prompts


def expand_emc(n: int) -> list[str]:
    prompts = []
    for _ in range(n):
        t = random.choice(EMC_TEMPLATES)
        prompts.append(t.format(
            interface=random.choice(INTERFACES),
            standard=random.choice(EMC_STANDARDS),
            esd_standard=random.choice(["IEC 61000-4-2", "ISO 10605"]),
            frequency=random.choice(FREQUENCIES),
            material=random.choice(["aluminum", "steel", "mu-metal", "copper"]),
            layers=random.choice(PCB_LAYERS),
            voltage=random.choice(VOLTAGES),
            current=random.choice(CURRENTS),
            test_equipment=random.choice(["LISN", "spectrum analyzer", "EMI receiver", "near-field probe"]),
            connector=random.choice(CONNECTORS),
            level=random.choice([1, 2, 3, 4]),
            application=random.choice(["automotive ECU", "industrial PLC", "medical device", "consumer IoT"]),
            field_strength=random.choice([3, 10, 20, 30]),
        ))
    return prompts


def expand_stm32(n: int) -> list[str]:
    prompts = []
    for _ in range(n):
        t = random.choice(STM32_TEMPLATES)
        prompts.append(t.format(
            mcu=random.choice(MCU_PARTS),
            peripheral=random.choice(["ADC", "DAC", "Timer", "UART", "SPI", "I2C", "CAN"]),
            mode=random.choice(["interrupt", "DMA", "polling", "continuous", "single-shot"]),
            frequency=random.choice(FREQUENCIES[:5]),
            interface=random.choice(INTERFACES[:6]),
            duplex=random.choice(["full-duplex", "half-duplex"]),
            speed=random.choice(["100kHz", "400kHz", "1MHz", "10MHz"]),
            channel_count=random.choice([1, 2, 4, 8, 16]),
            irq_list=", ".join(random.sample(["UART", "SPI", "Timer", "EXTI", "DMA", "ADC"], 4)),
            sensor=random.choice(SENSORS),
            lp_mode=random.choice(["STOP0", "STOP1", "STOP2", "STANDBY"]),
            wakeup_source=random.choice(["RTC", "EXTI", "UART", "LPTIM"]),
            timer=random.choice(["TIM1", "TIM2", "TIM3", "TIM6"]),
            clock_speed=random.choice([48, 72, 100, 168, 216, 480]),
            clock_source=random.choice(["HSE", "HSI", "MSI"]),
            rtos=random.choice(["FreeRTOS", "Zephyr", "ThreadX"]),
            application=random.choice(["sensor acquisition", "motor control", "data logger", "IoT gateway"]),
            task_count=random.choice([3, 4, 5, 8]),
            memory_type=random.choice(["QSPI flash", "external SRAM", "SDRAM"]),
            memory_mode=random.choice(["memory-mapped", "indirect", "status-polling"]),
            region_count=random.choice([2, 4, 8]),
            waveform=random.choice(["sine", "triangle", "sawtooth", "PWM"]),
            watchdog=random.choice(["IWDG", "WWDG"]),
            timeout=random.choice([1, 2, 5, 10]),
        ))
    return prompts


def expand_embedded(n: int) -> list[str]:
    prompts = []
    for _ in range(n):
        t = random.choice(EMBEDDED_TEMPLATES)
        prompts.append(t.format(
            interface=random.choice(INTERFACES[:6]),
            sensor=random.choice(SENSORS),
            architecture=random.choice(["ARM Cortex-M4", "ARM Cortex-M7", "RISC-V", "Xtensa (ESP32)"]),
            buffer_type=random.choice(["circular", "double", "ping-pong", "FIFO"]),
            peripheral=random.choice(["UART", "SPI", "ADC", "I2C"]),
            direction=random.choice(["receive", "transmit"]),
            rtos=random.choice(["FreeRTOS", "Zephyr", "ThreadX", "bare-metal"]),
            sync_primitive=random.choice(["semaphore", "mutex", "event group", "message queue"]),
            pattern=random.choice(["producer-consumer", "reader-writer", "pipeline"]),
            data_structure=random.choice(["ring buffer", "queue", "stack", "linked list"]),
            context=random.choice(["main loop", "RTOS task", "DPC handler"]),
            powered_by=random.choice(["battery", "solar", "USB", "PoE"]),
            device_type=random.choice(["IoT sensor node", "wearable", "remote meter", "asset tracker"]),
            buffering=random.choice(["double-buffering", "triple-buffering", "scatter-gather"]),
            timer_type=random.choice(["software timer wheel", "delta queue", "hierarchical"]),
            timer_count=random.choice([8, 16, 32, 64]),
            platform_list=random.choice(["STM32, ESP32, nRF52", "STM32, SAMD, RP2040", "ESP32, nRF52, RISC-V"]),
            algorithm=random.choice(["CRC-32", "SHA-256", "AES-128", "LZ4 compression"]),
            optimization=random.choice(["hardware acceleration", "SIMD intrinsics", "lookup table"]),
            verification=random.choice(["CRC32", "SHA-256", "ECDSA signature"]),
            storage=random.choice(["SD card", "SPI flash", "EEPROM", "FRAM"]),
            wear_strategy=random.choice(["wear leveling", "log-structured", "journaling"]),
            controller_type=random.choice(["PID", "PI", "cascaded PID", "state-space"]),
            arithmetic=random.choice(["fixed-point Q15", "fixed-point Q31", "floating-point"]),
            application=random.choice(["motor control", "temperature control", "position control"]),
            bus_type=random.choice(["RS-485", "CAN", "LIN", "Modbus"]),
            allocator_type=random.choice(["memory pool", "slab allocator", "buddy allocator"]),
            problem=random.choice(["heap fragmentation", "memory leaks", "stack overflow"]),
        ))
    return prompts


def expand_generic(n: int, domain: str) -> list[str]:
    """For domains without templates, create variations of existing prompts."""
    base_file = Path(f"data/mcp_extra_prompts.json")
    existing = []
    if base_file.exists():
        data = json.load(open(base_file))
        existing = data.get(domain, [])

    # Generate variations by adding constraints
    constraints = [
        "Optimize for minimum cost.",
        "Target automotive grade (-40C to 125C).",
        "Design for minimum PCB area.",
        "Ensure compliance with safety standards.",
        "Optimize for lowest power consumption.",
        "Design for high-volume manufacturing.",
        "Include test points for production testing.",
        "Design for EMC compliance.",
        "Optimize for signal integrity.",
        "Include thermal management considerations.",
        "Design for reliability (MTBF > 100,000 hours).",
        "Target consumer price point.",
        "Include protection against reverse polarity.",
        "Design for outdoor environmental conditions (IP65).",
        "Optimize for fastest response time.",
    ]

    prompts = list(existing)
    while len(prompts) < n:
        if existing:
            base = random.choice(existing)
            constraint = random.choice(constraints)
            prompts.append(f"{base} {constraint}")
        else:
            prompts.append(f"Explain a key concept in {domain} with a practical example.")
    return prompts[:n]


EXPANDERS = {
    "kicad-dsl": expand_kicad,
    "spice": expand_spice,
    "emc": expand_emc,
    "stm32": expand_stm32,
    "embedded": expand_embedded,
}

TARGETS = {
    "kicad-dsl": 500, "spice": 500, "emc": 500, "stm32": 500,
    "embedded": 500, "power": 250, "dsp": 250, "electronics": 250,
    "freecad": 125, "platformio": 125,
}

OUTPUT_ROOT = Path("data/prompts-expanded")


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    total = 0

    for domain, target in TARGETS.items():
        expander = EXPANDERS.get(domain)
        if expander:
            prompts = expander(target)
        else:
            prompts = expand_generic(target, domain)

        # Deduplicate
        seen = set()
        unique = []
        for p in prompts:
            if p not in seen:
                seen.add(p)
                unique.append(p)

        out_file = OUTPUT_ROOT / f"{domain}.jsonl"
        with open(out_file, "w") as f:
            for p in unique:
                f.write(json.dumps({"prompt": p, "domain": domain, "source": "expanded-template"}, ensure_ascii=False) + "\n")

        logger.info("%s: %d unique prompts (target %d)", domain, len(unique), target)
        total += len(unique)

    logger.info("TOTAL: %d expanded prompts across %d domains", total, len(TARGETS))


if __name__ == "__main__":
    main()

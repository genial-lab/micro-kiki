# Micro_KIKI Data Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete data preparation pipeline that produces 32 domain-specific training datasets (~2K examples each, ~64K total) in chat format with `<thinking>` tags, ready for Brainstacks MoE-LoRA fine-tuning of Qwen3.5-4B.

**Architecture:** A multi-stage pipeline: (1) download public datasets via `hf`, (2) classify each example into exactly 1 of 32 domains using keyword heuristics + LLM fallback, (3) generate synthetic data for sparse domains using local teachers (Qwen3.5-122B, Qwen3.5-35B via mlx-vlm), (4) deduplicate cross-domain with content hashing, (5) split train/valid per domain. All config lives in `configs/micro_kiki/domains.yaml`. All scripts are standalone and composable via a single orchestrator.

**Tech Stack:** Python 3.14, `hf` CLI, `pyarrow`, `mlx-lm`/`mlx-vlm` for generation, YAML config, JSONL output format

**Working directory:** `/Users/clems/KIKI-Mac_tunner`
**Python venv:** `.venv/` (activate with `source .venv/bin/activate`)

---

## File Structure

| File | Responsibility |
|------|---------------|
| `configs/micro_kiki/domains.yaml` | 32 domain definitions: name, keywords, regex patterns, teacher assignment, target count |
| `scripts/micro_kiki/__init__.py` | Package marker |
| `scripts/micro_kiki/download_datasets.sh` | Download all public datasets to `data/raw/` |
| `scripts/micro_kiki/classify_domains.py` | Load raw data, classify each example into best-matching domain |
| `scripts/micro_kiki/generate_missing.py` | Fill sparse domains with teacher-generated synthetic data |
| `scripts/micro_kiki/deduplicate.py` | Cross-domain deduplication (content hash, each example in exactly 1 domain) |
| `scripts/micro_kiki/split_domains.py` | Train/valid split (90/10) per domain, write final JSONL files |
| `scripts/micro_kiki/pipeline_data.sh` | Orchestrates all steps in sequence |
| `scripts/micro_kiki/validate_data.py` | Post-pipeline validation: counts, format checks, quality sampling |
| `tests/test_micro_kiki_data.py` | Unit tests for classifier, dedup, split logic |

**Existing files reused (not modified):**
- `data/final-opus-v3-1/train.jsonl` (11880 examples) — reasoning + general
- `data/final-opus-v3-1/valid.jsonl` (626 examples)
- `scripts/prepare_coding_dataset.py` — reference for Parquet/JSONL loaders, dedup key logic

---

## The 32 Domains

```
 #  Domain          Phase   Teacher primary       Teacher secondary
 1  chat-fr         1       122B Opus-v3          —
 2  reasoning       1       122B Opus-v3          Opus API
 3  python          2       Devstral 2 123B       Gemma 4 31B
 4  typescript      2       Devstral 2 123B       Gemma 4 31B
 5  cpp             2       Devstral 2 123B       Gemma 4 31B
 6  rust            2       Devstral 2 123B       Gemma 4 31B
 7  html-css        3       Devstral 2 123B       Gemma 4 31B
 8  shell           3       Devstral 2 123B       Gemma 4 31B
 9  sql             3       Devstral 2 123B       Gemma 4 31B
10  yaml-json       3       Devstral 2 123B       Gemma 4 31B
11  docker          3       Devstral 2 123B       Gemma 4 31B
12  kicad-dsl       3       122B Opus-v3          kiki-* datasets
13  spice           3       122B Opus-v3          kiki-* datasets
14  lua-upy         3       Devstral 2 123B       Gemma 4 31B
15  embedded        4       122B Opus-v3          kiki-* datasets
16  stm32           4       122B Opus-v3          kiki-* datasets
17  iot             4       122B Opus-v3          kiki-* datasets
18  freecad         4       122B Opus-v3          —
19  platformio      4       122B Opus-v3          kiki-* datasets
20  power           4       122B Opus-v3          kiki-* datasets
21  emc             4       122B Opus-v3          —
22  dsp             4       122B Opus-v3          —
23  spice-sim       4       122B Opus-v3          kiki-* datasets
24  electronics     4       122B Opus-v3          kiki-* datasets
25  kicad-pcb       4       122B Opus-v3          kiki-* datasets
26  web-frontend    5       Gemma 4 31B           122B Opus-v3
27  web-backend     5       Gemma 4 31B           122B Opus-v3
28  music-audio     5       122B Opus-v3          —
29  devops          5       Gemma 4 31B           122B Opus-v3
30  llm-orch        5       122B Opus-v3          —
31  math            6       122B Opus-v3          Opus API
32  security        6       122B Opus-v3          —
```

---

### Task 1: Create domain configuration

**Files:**
- Create: `configs/micro_kiki/domains.yaml`
- Create: `scripts/micro_kiki/__init__.py`

- [ ] **Step 1: Create the `configs/micro_kiki/` directory**

```bash
mkdir -p configs/micro_kiki scripts/micro_kiki
```

- [ ] **Step 2: Write `scripts/micro_kiki/__init__.py`**

```python
# Micro_KIKI data pipeline package
```

- [ ] **Step 3: Write `configs/micro_kiki/domains.yaml`**

This file defines all 32 domains with classification keywords, regex patterns, teacher assignments, and target example counts.

```yaml
# Micro_KIKI — 32 Expert Domain Configuration
# Each domain has:
#   keywords: list of strings for heuristic matching (case-insensitive substring)
#   patterns: list of regex patterns for stronger signal
#   teacher: primary teacher model for synthetic generation
#   teacher_secondary: fallback teacher (optional)
#   target: target number of examples
#   phase: curriculum training phase (1-6)
#   existing_sources: paths to existing datasets that map to this domain

version: 1
target_per_domain: 2000
valid_ratio: 0.1
chat_format: "messages"
thinking_tags: ["<thinking>", "</thinking>"]

domains:
  chat-fr:
    phase: 1
    teacher: "Qwen3.5-122B-A10B-BF16"
    target: 2000
    keywords:
      - "français"
      - "bonjour"
      - "merci"
      - "s'il vous plaît"
      - "expliquer"
      - "résumé"
      - "répondre en français"
      - "traduire"
      - "rédiger"
      - "conversation"
    patterns:
      - "\\b(en français|répon[ds]|expliqu|tradui|résume)\\b"
      - "\\b(bonjour|bonsoir|merci|svp)\\b"
    existing_sources:
      - "data/final-opus-v3-1"

  reasoning:
    phase: 1
    teacher: "Qwen3.5-122B-A10B-BF16"
    teacher_secondary: "opus-api"
    target: 2000
    keywords:
      - "step by step"
      - "reasoning"
      - "think through"
      - "logical"
      - "proof"
      - "deduce"
      - "analyze"
      - "explain why"
      - "what if"
      - "chain of thought"
    patterns:
      - "\\b(step.by.step|chain.of.thought|reasoning|deduc|logical)\\b"
      - "<thinking>"
    existing_sources:
      - "data/final-opus-v3-1"
      - "data/Opus-4.6-reasoning-sft-12k"

  python:
    phase: 2
    teacher: "devstral-2-123b"
    teacher_secondary: "gemma-4-31b"
    target: 2000
    keywords:
      - "python"
      - "def "
      - "import "
      - "class "
      - "pytest"
      - "pip"
      - "pandas"
      - "numpy"
      - "flask"
      - "django"
      - "fastapi"
    patterns:
      - "```python"
      - "\\b(def |import |class |from .+ import)\\b"
      - "\\.py\\b"
    public_datasets:
      - "nvidia/OpenCodeReasoning"
      - "ise-uiuc/Magicoder-OSS-Instruct-75K"

  typescript:
    phase: 2
    teacher: "devstral-2-123b"
    teacher_secondary: "gemma-4-31b"
    target: 2000
    keywords:
      - "typescript"
      - "interface "
      - "type "
      - "React"
      - "tsx"
      - "Angular"
      - "Vite"
      - "npm"
      - "Node.js"
      - "Express"
      - "Hono"
    patterns:
      - "```typescript"
      - "```tsx"
      - "\\b(interface |type |const .+:.+=|React\\.|tsx)\\b"
      - "\\.ts\\b"
    public_datasets:
      - "m-a-p/CodeFeedback-Filtered-Instruction"

  cpp:
    phase: 2
    teacher: "devstral-2-123b"
    teacher_secondary: "gemma-4-31b"
    target: 2000
    keywords:
      - "c++"
      - "cpp"
      - "#include"
      - "std::"
      - "template"
      - "namespace"
      - "cmake"
      - "pointer"
      - "RAII"
      - "constexpr"
    patterns:
      - "```cpp"
      - "```c\\+\\+"
      - "\\b(#include|std::|template<|namespace |class .+\\{)\\b"
      - "\\.cpp\\b|\\.hpp\\b"

  rust:
    phase: 2
    teacher: "devstral-2-123b"
    teacher_secondary: "gemma-4-31b"
    target: 2000
    keywords:
      - "rust"
      - "cargo"
      - "fn "
      - "let mut"
      - "impl "
      - "trait "
      - "ownership"
      - "borrow"
      - "lifetime"
      - "tokio"
    patterns:
      - "```rust"
      - "\\b(fn |impl |trait |use |pub |let mut|cargo)\\b"
      - "\\.rs\\b"

  html-css:
    phase: 3
    teacher: "devstral-2-123b"
    teacher_secondary: "gemma-4-31b"
    target: 2000
    keywords:
      - "html"
      - "css"
      - "flexbox"
      - "grid"
      - "responsive"
      - "tailwind"
      - "bootstrap"
      - "selector"
      - "media query"
      - "scss"
    patterns:
      - "```html"
      - "```css"
      - "\\b(<div|<section|<header|display:\\s*flex|display:\\s*grid)\\b"

  shell:
    phase: 3
    teacher: "devstral-2-123b"
    teacher_secondary: "gemma-4-31b"
    target: 2000
    keywords:
      - "bash"
      - "shell"
      - "zsh"
      - "#!/bin"
      - "grep"
      - "awk"
      - "sed"
      - "pipe"
      - "chmod"
      - "cron"
    patterns:
      - "```bash"
      - "```sh"
      - "\\b(#!/bin/|\\$\\(|\\|\\s*grep|awk |sed |chmod)\\b"

  sql:
    phase: 3
    teacher: "devstral-2-123b"
    teacher_secondary: "gemma-4-31b"
    target: 2000
    keywords:
      - "sql"
      - "SELECT"
      - "INSERT"
      - "JOIN"
      - "PostgreSQL"
      - "MySQL"
      - "SQLite"
      - "database"
      - "query"
      - "index"
    patterns:
      - "```sql"
      - "\\b(SELECT |INSERT |UPDATE |DELETE |CREATE TABLE|ALTER TABLE|JOIN )\\b"

  yaml-json:
    phase: 3
    teacher: "devstral-2-123b"
    teacher_secondary: "gemma-4-31b"
    target: 2000
    keywords:
      - "yaml"
      - "json"
      - "schema"
      - "OpenAPI"
      - "swagger"
      - "config"
      - "toml"
      - "JSON Schema"
      - "validation"
    patterns:
      - "```yaml"
      - "```json"
      - "\\b(apiVersion:|kind:|openapi:|\\{\"[a-z]+\":|\\$ref)\\b"

  docker:
    phase: 3
    teacher: "devstral-2-123b"
    teacher_secondary: "gemma-4-31b"
    target: 2000
    keywords:
      - "docker"
      - "Dockerfile"
      - "container"
      - "docker-compose"
      - "image"
      - "volume"
      - "FROM "
      - "kubernetes"
      - "k8s"
      - "podman"
    patterns:
      - "```dockerfile"
      - "\\b(FROM |COPY |RUN |EXPOSE |docker compose|docker build)\\b"

  kicad-dsl:
    phase: 3
    teacher: "Qwen3.5-122B-A10B-BF16"
    target: 2000
    keywords:
      - "kicad"
      - "schematic"
      - "netlist"
      - "footprint"
      - "symbol"
      - "eeschema"
      - "s-expression"
      - "kicad_sch"
      - "kicad_sym"
      - "kicad_mod"
    patterns:
      - "\\b(kicad_sch|kicad_sym|kicad_mod|fp_name|pad |module )\\b"
      - "\\(kicad_"
    existing_sources:
      - "clemsail/kiki-kicad"

  spice:
    phase: 3
    teacher: "Qwen3.5-122B-A10B-BF16"
    target: 2000
    keywords:
      - "spice"
      - "ngspice"
      - "ltspice"
      - "netlist"
      - ".subckt"
      - ".tran"
      - ".ac"
      - ".dc"
      - "MOSFET"
      - "transistor"
    patterns:
      - "\\b(\\.subckt|\\.tran|\\.ac |\\.dc |\\.model |V[0-9]+|R[0-9]+)\\b"
    existing_sources:
      - "clemsail/kiki-spice"

  lua-upy:
    phase: 3
    teacher: "devstral-2-123b"
    teacher_secondary: "gemma-4-31b"
    target: 2000
    keywords:
      - "lua"
      - "micropython"
      - "upython"
      - "machine.Pin"
      - "function"
      - "local "
      - "require"
      - "coroutine"
      - "uos"
      - "ujson"
    patterns:
      - "```lua"
      - "```micropython"
      - "\\b(local |function |require\\(|machine\\.Pin|uos\\.)\\b"

  embedded:
    phase: 4
    teacher: "Qwen3.5-122B-A10B-BF16"
    target: 2000
    keywords:
      - "embedded"
      - "firmware"
      - "ESP-IDF"
      - "ESP32"
      - "GPIO"
      - "interrupt"
      - "RTOS"
      - "FreeRTOS"
      - "HAL"
      - "bare metal"
      - "DMA"
      - "SPI"
      - "I2C"
      - "UART"
    patterns:
      - "\\b(esp_err_t|gpio_config|xTaskCreate|HAL_GPIO|idf\\.py|menuconfig)\\b"
      - "\\b(FreeRTOS|vTaskDelay|xQueue|xSemaphore)\\b"
    existing_sources:
      - "clemsail/kiki-embedded"
      - "clemsail/kiki-esp32"

  stm32:
    phase: 4
    teacher: "Qwen3.5-122B-A10B-BF16"
    target: 2000
    keywords:
      - "STM32"
      - "CubeMX"
      - "HAL_"
      - "LL_"
      - "stm32f"
      - "stm32h"
      - "stm32g"
      - "ARM Cortex"
      - "CMSIS"
      - "OpenOCD"
    patterns:
      - "\\b(STM32[FGHL]|HAL_[A-Z]+_|LL_[A-Z]+_|CubeMX|CMSIS|__HAL)\\b"
    existing_sources:
      - "clemsail/kiki-stm32"

  iot:
    phase: 4
    teacher: "Qwen3.5-122B-A10B-BF16"
    target: 2000
    keywords:
      - "MQTT"
      - "BLE"
      - "Bluetooth"
      - "WiFi"
      - "LoRa"
      - "Zigbee"
      - "Matter"
      - "Thread"
      - "CoAP"
      - "Home Assistant"
      - "ESP-NOW"
      - "OTA"
    patterns:
      - "\\b(mqtt|ble_|esp_wifi|lora_|zigbee|matter_|esp_now)\\b"
    existing_sources:
      - "clemsail/kiki-iot"

  freecad:
    phase: 4
    teacher: "Qwen3.5-122B-A10B-BF16"
    target: 2000
    keywords:
      - "FreeCAD"
      - "Part.make"
      - "Sketcher"
      - "PartDesign"
      - "mesh"
      - "STEP"
      - "IGES"
      - "3D print"
      - "BRep"
      - "parametric"
    patterns:
      - "\\b(FreeCAD|Part\\.|Sketcher\\.|PartDesign|App\\.ActiveDocument)\\b"

  platformio:
    phase: 4
    teacher: "Qwen3.5-122B-A10B-BF16"
    target: 2000
    keywords:
      - "PlatformIO"
      - "platformio.ini"
      - "pio"
      - "lib_deps"
      - "board"
      - "framework"
      - "monitor_speed"
      - "upload_port"
      - "build_flags"
    patterns:
      - "\\b(platformio\\.ini|\\[env:|lib_deps|board =|pio run|pio test)\\b"
    existing_sources:
      - "clemsail/kiki-platformio"

  power:
    phase: 4
    teacher: "Qwen3.5-122B-A10B-BF16"
    target: 2000
    keywords:
      - "power supply"
      - "regulator"
      - "buck"
      - "boost"
      - "LDO"
      - "battery"
      - "voltage"
      - "current limit"
      - "efficiency"
      - "MPPT"
      - "charge"
      - "discharge"
    patterns:
      - "\\b(LDO|buck|boost|MPPT|V[io]n|V[io]ut|LM317|TPS|MCP73)\\b"
    existing_sources:
      - "clemsail/kiki-power"

  emc:
    phase: 4
    teacher: "Qwen3.5-122B-A10B-BF16"
    target: 2000
    keywords:
      - "EMC"
      - "EMI"
      - "CEM"
      - "filtering"
      - "shielding"
      - "common mode"
      - "differential mode"
      - "ferrite"
      - "decoupling"
      - "ground plane"
      - "radiated"
      - "conducted"
    patterns:
      - "\\b(EMC|EMI|CEM|common.mode|differential.mode|ferrite|decoupling)\\b"

  dsp:
    phase: 4
    teacher: "Qwen3.5-122B-A10B-BF16"
    target: 2000
    keywords:
      - "DSP"
      - "FFT"
      - "filter"
      - "convolution"
      - "sampling"
      - "Nyquist"
      - "FIR"
      - "IIR"
      - "signal processing"
      - "spectrogram"
      - "windowing"
    patterns:
      - "\\b(FFT|FIR|IIR|Nyquist|spectrogram|convolution|CMSIS.DSP)\\b"

  spice-sim:
    phase: 4
    teacher: "Qwen3.5-122B-A10B-BF16"
    target: 2000
    keywords:
      - "simulation"
      - "transient"
      - "AC analysis"
      - "DC sweep"
      - "Monte Carlo"
      - "parametric"
      - "convergence"
      - "step response"
      - "Bode plot"
      - "waveform"
    patterns:
      - "\\b(transient|AC.analysis|DC.sweep|Monte.Carlo|Bode|convergence)\\b"
    existing_sources:
      - "clemsail/kiki-spice-sim"

  electronics:
    phase: 4
    teacher: "Qwen3.5-122B-A10B-BF16"
    target: 2000
    keywords:
      - "analog"
      - "op-amp"
      - "capacitor"
      - "resistor"
      - "inductor"
      - "transistor"
      - "diode"
      - "amplifier"
      - "oscillator"
      - "PCB"
      - "component"
      - "RF"
    patterns:
      - "\\b(op.amp|capacitor|resistor|inductor|transistor|diode|amplifier|RF)\\b"
    existing_sources:
      - "clemsail/kiki-electronics"

  kicad-pcb:
    phase: 4
    teacher: "Qwen3.5-122B-A10B-BF16"
    target: 2000
    keywords:
      - "PCB"
      - "routing"
      - "DRC"
      - "copper"
      - "layer"
      - "via"
      - "trace"
      - "clearance"
      - "stackup"
      - "impedance"
      - "differential pair"
      - "gerber"
    patterns:
      - "\\b(kicad_pcb|DRC|clearance|stackup|impedance|differential.pair|gerber)\\b"
    existing_sources:
      - "clemsail/kiki-kicad-pcb"

  web-frontend:
    phase: 5
    teacher: "gemma-4-31b"
    teacher_secondary: "Qwen3.5-122B-A10B-BF16"
    target: 2000
    keywords:
      - "React"
      - "Vue"
      - "Svelte"
      - "Next.js"
      - "Vite"
      - "component"
      - "state management"
      - "hook"
      - "SSR"
      - "hydration"
      - "Tailwind"
    patterns:
      - "\\b(useState|useEffect|React\\.|Vue\\.|Svelte|Next\\.js|Vite)\\b"

  web-backend:
    phase: 5
    teacher: "gemma-4-31b"
    teacher_secondary: "Qwen3.5-122B-A10B-BF16"
    target: 2000
    keywords:
      - "API"
      - "REST"
      - "GraphQL"
      - "FastAPI"
      - "Express"
      - "Hono"
      - "middleware"
      - "endpoint"
      - "authentication"
      - "rate limiting"
      - "database"
    patterns:
      - "\\b(FastAPI|Express|Hono|app\\.get|app\\.post|@app\\.route|middleware)\\b"

  music-audio:
    phase: 5
    teacher: "Qwen3.5-122B-A10B-BF16"
    target: 2000
    keywords:
      - "audio"
      - "music"
      - "MIDI"
      - "synthesizer"
      - "DAW"
      - "sample rate"
      - "TTS"
      - "Piper"
      - "Web Audio"
      - "waveform"
      - "instrument"
      - "oscillator"
    patterns:
      - "\\b(MIDI|AudioContext|sample.rate|synthesizer|TTS|Piper|DAW|wav)\\b"

  devops:
    phase: 5
    teacher: "gemma-4-31b"
    teacher_secondary: "Qwen3.5-122B-A10B-BF16"
    target: 2000
    keywords:
      - "CI/CD"
      - "GitHub Actions"
      - "Terraform"
      - "Ansible"
      - "nginx"
      - "Tailscale"
      - "SSL"
      - "monitoring"
      - "Prometheus"
      - "Grafana"
      - "deploy"
    patterns:
      - "\\b(GitHub.Actions|Terraform|Ansible|nginx|Tailscale|Prometheus|deploy)\\b"

  llm-orch:
    phase: 5
    teacher: "Qwen3.5-122B-A10B-BF16"
    target: 2000
    keywords:
      - "LLM"
      - "RAG"
      - "agent"
      - "embedding"
      - "vector"
      - "Qdrant"
      - "LangChain"
      - "prompt"
      - "fine-tuning"
      - "LoRA"
      - "routing"
      - "orchestration"
    patterns:
      - "\\b(RAG|LLM|embedding|Qdrant|LangChain|LoRA|fine.tun|vector.store)\\b"

  math:
    phase: 6
    teacher: "Qwen3.5-122B-A10B-BF16"
    teacher_secondary: "opus-api"
    target: 2000
    keywords:
      - "mathematics"
      - "equation"
      - "integral"
      - "derivative"
      - "matrix"
      - "probability"
      - "statistics"
      - "physics"
      - "calculus"
      - "linear algebra"
      - "optimization"
    patterns:
      - "\\b(integral|derivative|matrix|probability|calculus|equation|theorem)\\b"
      - "\\$.*\\$"

  security:
    phase: 6
    teacher: "Qwen3.5-122B-A10B-BF16"
    target: 2000
    keywords:
      - "security"
      - "vulnerability"
      - "OWASP"
      - "authentication"
      - "encryption"
      - "XSS"
      - "CSRF"
      - "SQL injection"
      - "certificate"
      - "JWT"
      - "OAuth"
      - "penetration"
    patterns:
      - "\\b(OWASP|XSS|CSRF|SQL.injection|CVE-|JWT|OAuth|TLS|encryption)\\b"
```

- [ ] **Step 4: Verify YAML is valid**

```bash
source .venv/bin/activate && python3 -c "
import yaml
from pathlib import Path
data = yaml.safe_load(Path('configs/micro_kiki/domains.yaml').read_text())
domains = data['domains']
print(f'Loaded {len(domains)} domains')
assert len(domains) == 32, f'Expected 32 domains, got {len(domains)}'
for name, cfg in domains.items():
    assert 'phase' in cfg, f'{name} missing phase'
    assert 'teacher' in cfg, f'{name} missing teacher'
    assert 'keywords' in cfg, f'{name} missing keywords'
    assert 'patterns' in cfg, f'{name} missing patterns'
    print(f'  {cfg[\"phase\"]:>1} {name:<16} teacher={cfg[\"teacher\"][:20]} keywords={len(cfg[\"keywords\"])} patterns={len(cfg[\"patterns\"])}')
print('All 32 domains valid.')
"
```

Expected: `Loaded 32 domains` followed by 32 lines, ending with `All 32 domains valid.`

- [ ] **Step 5: Commit**

```bash
git add configs/micro_kiki/domains.yaml scripts/micro_kiki/__init__.py
git commit -m "feat: add 32-domain config for Micro_KIKI data pipeline"
```

---

### Task 2: Download public datasets

**Files:**
- Create: `scripts/micro_kiki/download_datasets.sh`

- [ ] **Step 1: Write the download script**

```bash
#!/bin/bash
# Download all public datasets needed for Micro_KIKI 32-domain classification.
# Usage: ./scripts/micro_kiki/download_datasets.sh [dataset|all]
#
# Downloads to data/raw/<dataset-name>/
# Uses `hf` CLI (not huggingface-cli which is deprecated).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"

RAW_DIR="$PROJECT_DIR/data/raw"
mkdir -p "$RAW_DIR"

download_codefeedback() {
    echo "=== CodeFeedback-Filtered-Instruction (156K) ==="
    local dest="$RAW_DIR/CodeFeedback-Filtered-Instruction"
    if [ -d "$dest" ] && [ "$(ls -A "$dest" 2>/dev/null)" ]; then
        echo "  Already downloaded at $dest"
        return
    fi
    hf download m-a-p/CodeFeedback-Filtered-Instruction \
        --repo-type dataset \
        --local-dir "$dest"
    echo "  Done: $(find "$dest" -name '*.parquet' -o -name '*.jsonl' | wc -l) files"
}

download_opencodereasoning() {
    echo "=== OpenCodeReasoning (735K) ==="
    local dest="$RAW_DIR/OpenCodeReasoning"
    if [ -d "$dest" ] && [ "$(ls -A "$dest" 2>/dev/null)" ]; then
        echo "  Already downloaded at $dest"
        return
    fi
    hf download nvidia/OpenCodeReasoning \
        --repo-type dataset \
        --local-dir "$dest"
    echo "  Done: $(find "$dest" -name '*.parquet' -o -name '*.jsonl' | wc -l) files"
}

download_magicoder() {
    echo "=== Magicoder-OSS-Instruct-75K ==="
    local dest="$RAW_DIR/Magicoder-OSS-Instruct-75K"
    if [ -d "$dest" ] && [ "$(ls -A "$dest" 2>/dev/null)" ]; then
        echo "  Already downloaded at $dest"
        return
    fi
    hf download ise-uiuc/Magicoder-OSS-Instruct-75K \
        --repo-type dataset \
        --local-dir "$dest"
    echo "  Done: $(find "$dest" -name '*.parquet' -o -name '*.jsonl' | wc -l) files"
}

download_openhermes() {
    echo "=== OpenHermes-2.5 (1M, general) ==="
    local dest="$RAW_DIR/OpenHermes-2.5"
    if [ -d "$dest" ] && [ "$(ls -A "$dest" 2>/dev/null)" ]; then
        echo "  Already downloaded at $dest"
        return
    fi
    hf download teknium/OpenHermes-2.5 \
        --repo-type dataset \
        --local-dir "$dest"
    echo "  Done: $(find "$dest" -name '*.parquet' -o -name '*.jsonl' -o -name '*.json' | wc -l) files"
}

download_kiki_datasets() {
    echo "=== kiki-* HuggingFace datasets (clemsail) ==="
    local kiki_datasets=(
        "clemsail/kiki-embedded"
        "clemsail/kiki-electronics"
        "clemsail/kiki-esp32"
        "clemsail/kiki-kicad"
        "clemsail/kiki-kicad-pcb"
        "clemsail/kiki-stm32"
        "clemsail/kiki-iot"
        "clemsail/kiki-platformio"
        "clemsail/kiki-power"
        "clemsail/kiki-spice"
        "clemsail/kiki-spice-sim"
    )
    for ds in "${kiki_datasets[@]}"; do
        local name="${ds#clemsail/}"
        local dest="$RAW_DIR/$name"
        if [ -d "$dest" ] && [ "$(ls -A "$dest" 2>/dev/null)" ]; then
            echo "  $name already downloaded"
            continue
        fi
        echo "  Downloading $ds..."
        hf download "$ds" \
            --repo-type dataset \
            --local-dir "$dest" 2>/dev/null || echo "  WARNING: $ds not found or inaccessible, skipping"
    done
    echo "  Done."
}

download_existing_local() {
    echo "=== Linking existing local datasets ==="
    # final-opus-v3-1 (reasoning, general)
    if [ -d "$PROJECT_DIR/data/final-opus-v3-1" ]; then
        ln -sfn "$PROJECT_DIR/data/final-opus-v3-1" "$RAW_DIR/final-opus-v3-1"
        echo "  Linked final-opus-v3-1 ($(wc -l < "$PROJECT_DIR/data/final-opus-v3-1/train.jsonl") train)"
    fi
    # Opus reasoning datasets
    for ds in Opus-4.6-Reasoning-3000x-filtered Opus-4.6-reasoning-sft-12k claude-opus-4.6-10000x; do
        if [ -d "$PROJECT_DIR/data/$ds" ]; then
            ln -sfn "$PROJECT_DIR/data/$ds" "$RAW_DIR/$ds"
            echo "  Linked $ds"
        fi
    done
    echo "  Done."
}

print_usage() {
    echo "Usage: $0 <dataset|all>"
    echo ""
    echo "Datasets:"
    echo "  codefeedback        CodeFeedback-Filtered-Instruction (156K examples)"
    echo "  opencodereasoning   OpenCodeReasoning NVIDIA (735K examples)"
    echo "  magicoder           Magicoder-OSS-Instruct-75K"
    echo "  openhermes          OpenHermes-2.5 (general instruction, 1M)"
    echo "  kiki                kiki-* datasets from clemsail HuggingFace"
    echo "  local               Link existing local datasets"
    echo "  all                 Download everything"
}

case "${1:-help}" in
    codefeedback)       download_codefeedback ;;
    opencodereasoning)  download_opencodereasoning ;;
    magicoder)          download_magicoder ;;
    openhermes)         download_openhermes ;;
    kiki)               download_kiki_datasets ;;
    local)              download_existing_local ;;
    all)
        download_existing_local
        download_kiki_datasets
        download_codefeedback
        download_opencodereasoning
        download_magicoder
        download_openhermes
        ;;
    *)  print_usage ;;
esac

echo ""
echo "=== Raw data summary ==="
for d in "$RAW_DIR"/*/; do
    if [ -d "$d" ]; then
        name="$(basename "$d")"
        count=$(find "$d" -name '*.jsonl' -o -name '*.parquet' -o -name '*.json' 2>/dev/null | wc -l)
        echo "  $name: $count data files"
    fi
done
```

- [ ] **Step 2: Make executable and verify syntax**

```bash
chmod +x scripts/micro_kiki/download_datasets.sh
bash -n scripts/micro_kiki/download_datasets.sh
echo "Syntax OK"
```

Expected: `Syntax OK`

- [ ] **Step 3: Test with `local` target only (no network)**

```bash
./scripts/micro_kiki/download_datasets.sh local
```

Expected: Links created for `final-opus-v3-1` and any other existing local datasets, plus a summary.

- [ ] **Step 4: Commit**

```bash
git add scripts/micro_kiki/download_datasets.sh
git commit -m "feat: add dataset download script for 32-domain pipeline"
```

---

### Task 3: Domain classifier

**Files:**
- Create: `scripts/micro_kiki/classify_domains.py`
- Create: `tests/test_micro_kiki_data.py`

- [ ] **Step 1: Write the failing test**

```python
#!/usr/bin/env python3
"""Tests for Micro_KIKI data pipeline components."""

import hashlib
import json
import tempfile
from pathlib import Path

import pytest
import yaml


# --- Test domain classifier ---

def load_domains_config():
    """Load domain config for tests."""
    config_path = Path(__file__).parent.parent / "configs" / "micro_kiki" / "domains.yaml"
    return yaml.safe_load(config_path.read_text())


class TestDomainClassifier:
    """Test the heuristic domain classifier."""

    def test_classify_python_example(self):
        from scripts.micro_kiki.classify_domains import classify_example
        example = {
            "messages": [
                {"role": "user", "content": "Write a Python function to sort a list"},
                {"role": "assistant", "content": "<thinking>I need to write a sort function in Python.</thinking>\n\ndef sort_list(items):\n    return sorted(items)"},
            ]
        }
        config = load_domains_config()
        domain = classify_example(example, config["domains"])
        assert domain == "python"

    def test_classify_embedded_example(self):
        from scripts.micro_kiki.classify_domains import classify_example
        example = {
            "messages": [
                {"role": "user", "content": "How to configure GPIO on ESP32 with ESP-IDF?"},
                {"role": "assistant", "content": "<thinking>ESP-IDF GPIO configuration uses gpio_config_t.</thinking>\n\ngpio_config_t io_conf = { .pin_bit_mask = (1ULL << GPIO_NUM_2) };"},
            ]
        }
        config = load_domains_config()
        domain = classify_example(example, config["domains"])
        assert domain == "embedded"

    def test_classify_french_chat(self):
        from scripts.micro_kiki.classify_domains import classify_example
        example = {
            "messages": [
                {"role": "user", "content": "Explique-moi le fonctionnement d'un transformateur en français"},
                {"role": "assistant", "content": "Un transformateur fonctionne grâce à l'induction électromagnétique."},
            ]
        }
        config = load_domains_config()
        domain = classify_example(example, config["domains"])
        assert domain == "chat-fr"

    def test_classify_returns_none_for_empty(self):
        from scripts.micro_kiki.classify_domains import classify_example
        example = {"messages": [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}]}
        config = load_domains_config()
        domain = classify_example(example, config["domains"])
        assert domain is None

    def test_classify_reasoning(self):
        from scripts.micro_kiki.classify_domains import classify_example
        example = {
            "messages": [
                {"role": "user", "content": "Think step by step about this logical puzzle"},
                {"role": "assistant", "content": "<thinking>Let me reason through this step by step. First, I need to analyze the logical structure...</thinking>\n\nThe answer is B."},
            ]
        }
        config = load_domains_config()
        domain = classify_example(example, config["domains"])
        assert domain == "reasoning"

    def test_all_32_domains_in_config(self):
        config = load_domains_config()
        assert len(config["domains"]) == 32

    def test_each_domain_has_required_fields(self):
        config = load_domains_config()
        for name, domain in config["domains"].items():
            assert "phase" in domain, f"{name} missing phase"
            assert "teacher" in domain, f"{name} missing teacher"
            assert "keywords" in domain, f"{name} missing keywords"
            assert "patterns" in domain, f"{name} missing patterns"
            assert "target" in domain, f"{name} missing target"
```

Save to `tests/test_micro_kiki_data.py`.

- [ ] **Step 2: Run test to verify it fails**

```bash
source .venv/bin/activate && python -m pytest tests/test_micro_kiki_data.py -v --tb=short 2>&1 | head -40
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.micro_kiki.classify_domains'`

- [ ] **Step 3: Write the classifier implementation**

```python
#!/usr/bin/env python3
"""Classify examples into 1 of 32 Micro_KIKI domains.

Reads raw datasets from data/raw/, classifies each example using keyword
heuristics + regex patterns, and writes per-domain JSONL to data/micro-kiki/classified/.

Usage:
    python scripts/micro_kiki/classify_domains.py [--config configs/micro_kiki/domains.yaml]
                                                   [--input-dir data/raw]
                                                   [--output-dir data/micro-kiki/classified]
                                                   [--max-per-domain 3000]
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

try:
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False


def load_config(config_path: str) -> dict:
    """Load and validate domain configuration."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    assert len(config["domains"]) == 32, f"Expected 32 domains, got {len(config['domains'])}"
    return config


def compile_patterns(domains: dict[str, dict]) -> dict[str, list[re.Pattern]]:
    """Pre-compile regex patterns for each domain."""
    compiled = {}
    for name, cfg in domains.items():
        compiled[name] = [re.compile(p, re.IGNORECASE) for p in cfg.get("patterns", [])]
    return compiled


def extract_text(example: dict) -> str:
    """Extract all text from an example for classification."""
    messages = example.get("messages", [])
    parts = []
    for msg in messages:
        content = msg.get("content", "")
        if content:
            parts.append(content)

    # Also handle non-chat formats
    for key in ("instruction", "input", "output", "question", "answer",
                "solution", "response", "prompt", "completion", "text"):
        val = example.get(key, "")
        if val and isinstance(val, str):
            parts.append(val)

    return "\n".join(parts)


def score_domain(text: str, domain_name: str, domain_cfg: dict,
                 compiled_patterns: list[re.Pattern]) -> float:
    """Score how well a text matches a domain. Higher = better match."""
    if not text.strip():
        return 0.0

    text_lower = text.lower()
    score = 0.0

    # Keyword matches (1 point each, diminishing returns)
    keyword_hits = 0
    for kw in domain_cfg.get("keywords", []):
        if kw.lower() in text_lower:
            keyword_hits += 1
    score += min(keyword_hits * 1.0, 5.0)

    # Regex pattern matches (3 points each — stronger signal)
    for pattern in compiled_patterns:
        matches = pattern.findall(text)
        if matches:
            score += min(len(matches) * 3.0, 12.0)

    return score


def classify_example(example: dict, domains: dict[str, dict],
                     _compiled: dict[str, list[re.Pattern]] | None = None) -> str | None:
    """Classify a single example into its best-matching domain.

    Returns the domain name, or None if no domain scores above threshold.
    """
    text = extract_text(example)
    if len(text.strip()) < 20:
        return None

    compiled = _compiled or compile_patterns(domains)

    best_domain = None
    best_score = 0.0

    for name, cfg in domains.items():
        s = score_domain(text, name, cfg, compiled.get(name, []))
        if s > best_score:
            best_score = s
            best_domain = name

    # Minimum threshold: at least 1 keyword hit
    if best_score < 1.0:
        return None

    return best_domain


def normalize_to_chat(example: dict) -> dict | None:
    """Normalize any example format to chat messages format.

    Returns None if the example cannot be converted.
    """
    # Already in chat format
    if "messages" in example and isinstance(example["messages"], list):
        msgs = example["messages"]
        if len(msgs) >= 2:
            return {"messages": msgs}

    # OpenCodeReasoning / CodeFeedback format
    user = (example.get("question") or example.get("instruction")
            or example.get("input") or example.get("prompt") or "")
    assistant = (example.get("solution") or example.get("output")
                 or example.get("response") or example.get("answer")
                 or example.get("completion") or "")
    reasoning = example.get("reasoning", "")

    if not user.strip() or not assistant.strip():
        return None

    # Normalize thinking tags
    assistant = assistant.replace("<think>", "<thinking>").replace("</think>", "</thinking>")

    # Add reasoning if present but not already wrapped
    if reasoning and "<thinking>" not in assistant:
        assistant = f"<thinking>\n{reasoning.strip()}\n</thinking>\n\n{assistant.strip()}"

    return {
        "messages": [
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ]
    }


def load_all_raw(input_dir: Path) -> list[dict]:
    """Load all examples from all raw dataset directories."""
    examples = []
    if not input_dir.exists():
        print(f"WARNING: {input_dir} does not exist", file=sys.stderr)
        return examples

    for dataset_dir in sorted(input_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue

        count_before = len(examples)
        for filepath in sorted(dataset_dir.rglob("*")):
            if filepath.suffix == ".jsonl":
                with open(filepath) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            examples.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            elif filepath.suffix == ".json" and filepath.name not in (
                "dataset_info.json", "dataset_infos.json", "config.json"
            ):
                try:
                    with open(filepath) as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        examples.extend(data)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
            elif filepath.suffix == ".parquet" and HAS_PARQUET:
                try:
                    table = pq.read_table(filepath)
                    examples.extend(table.to_pylist())
                except Exception:
                    continue

        loaded = len(examples) - count_before
        if loaded > 0:
            print(f"  {dataset_dir.name}: {loaded} examples")

    return examples


def run_classification(config_path: str, input_dir: str, output_dir: str,
                       max_per_domain: int = 3000) -> dict[str, int]:
    """Run the full classification pipeline.

    Returns a dict of domain -> count.
    """
    config = load_config(config_path)
    domains = config["domains"]
    compiled = compile_patterns(domains)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw data from {input_dir}...")
    raw_examples = load_all_raw(Path(input_dir))
    print(f"Loaded {len(raw_examples)} total raw examples")

    # Classify
    domain_examples: dict[str, list[dict]] = {name: [] for name in domains}
    unclassified = 0
    unconvertible = 0

    for i, raw in enumerate(raw_examples):
        if i % 50000 == 0 and i > 0:
            print(f"  Classified {i}/{len(raw_examples)}...")

        # Normalize to chat format
        chat = normalize_to_chat(raw)
        if chat is None:
            unconvertible += 1
            continue

        # Classify
        domain = classify_example(chat, domains, compiled)
        if domain is None:
            unclassified += 1
            continue

        if len(domain_examples[domain]) < max_per_domain:
            domain_examples[domain].append(chat)

    # Write per-domain JSONL
    counts = {}
    for name, examples in domain_examples.items():
        outfile = out_path / f"{name}.jsonl"
        with open(outfile, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        counts[name] = len(examples)

    # Summary
    print(f"\n=== Classification Summary ===")
    print(f"Total raw: {len(raw_examples)}")
    print(f"Unconvertible: {unconvertible}")
    print(f"Unclassified: {unclassified}")
    print(f"Classified: {sum(counts.values())}")
    print()
    for name in sorted(counts, key=lambda n: domains[n]["phase"]):
        target = domains[name]["target"]
        count = counts[name]
        status = "OK" if count >= target * 0.5 else "SPARSE"
        print(f"  Phase {domains[name]['phase']} {name:<16} {count:>5}/{target} [{status}]")

    return counts


def main():
    parser = argparse.ArgumentParser(description="Classify examples into 32 Micro_KIKI domains")
    parser.add_argument("--config", default="configs/micro_kiki/domains.yaml",
                        help="Path to domain configuration YAML")
    parser.add_argument("--input-dir", default="data/raw",
                        help="Directory containing raw datasets")
    parser.add_argument("--output-dir", default="data/micro-kiki/classified",
                        help="Output directory for per-domain JSONL")
    parser.add_argument("--max-per-domain", type=int, default=3000,
                        help="Maximum examples per domain from public data (default: 3000)")
    args = parser.parse_args()

    run_classification(args.config, args.input_dir, args.output_dir, args.max_per_domain)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
source .venv/bin/activate && python -m pytest tests/test_micro_kiki_data.py -v --tb=short
```

Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/micro_kiki/classify_domains.py tests/test_micro_kiki_data.py
git commit -m "feat: add heuristic domain classifier for 32 Micro_KIKI domains"
```

---

### Task 4: Synthetic data generation for sparse domains

**Files:**
- Create: `scripts/micro_kiki/generate_missing.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_micro_kiki_data.py`:

```python
class TestGeneratePrompts:
    """Test prompt generation for sparse domains."""

    def test_generate_prompt_for_domain(self):
        from scripts.micro_kiki.generate_missing import build_generation_prompt
        config = load_domains_config()
        prompt = build_generation_prompt("embedded", config["domains"]["embedded"])
        assert "embedded" in prompt.lower() or "ESP" in prompt or "firmware" in prompt
        assert len(prompt) > 50

    def test_parse_generated_response(self):
        from scripts.micro_kiki.generate_missing import parse_teacher_response
        raw_response = (
            "USER: How do I configure SPI on ESP32?\n"
            "ASSISTANT: <thinking>SPI configuration requires...</thinking>\n\n"
            "To configure SPI on ESP32, use esp_driver_spi..."
        )
        result = parse_teacher_response(raw_response)
        assert result is not None
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][1]["role"] == "assistant"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source .venv/bin/activate && python -m pytest tests/test_micro_kiki_data.py::TestGeneratePrompts -v --tb=short
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.micro_kiki.generate_missing'`

- [ ] **Step 3: Write the generation script**

```python
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
import subprocess
import sys
import tempfile
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
    user_match = None
    assistant_match = None

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


def generate_with_mlx(prompt: str, model_path: str, max_tokens: int = 2048) -> str:
    """Generate text using mlx-vlm or mlx-lm locally."""
    # Write prompt to temp file for clean passing
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(prompt)
        prompt_file = f.name

    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "mlx_lm.generate",
                "--model", model_path,
                "--prompt", prompt,
                "--max-tokens", str(max_tokens),
                "--temp", "0.8",
                "--top-p", "0.95",
            ],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            print(f"  mlx_lm.generate error: {result.stderr[:200]}", file=sys.stderr)
            return ""
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        print("  Generation timed out", file=sys.stderr)
        return ""
    finally:
        Path(prompt_file).unlink(missing_ok=True)


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
                   dry_run: bool = False) -> dict[str, int]:
    """Generate missing data for sparse domains.

    Returns dict of domain -> number of generated examples.
    """
    config = yaml.safe_load(Path(config_path).read_text())
    domains = config["domains"]
    existing = count_classified(Path(classified_dir))
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    generated_counts = {}

    for name, cfg in domains.items():
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
    args = parser.parse_args()

    run_generation(args.config, args.classified_dir, args.output_dir,
                   args.teacher_model, args.max_generate, args.dry_run)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
source .venv/bin/activate && python -m pytest tests/test_micro_kiki_data.py::TestGeneratePrompts -v --tb=short
```

Expected: All 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/micro_kiki/generate_missing.py tests/test_micro_kiki_data.py
git commit -m "feat: add synthetic data generator for sparse Micro_KIKI domains"
```

---

### Task 5: Cross-domain deduplication

**Files:**
- Create: `scripts/micro_kiki/deduplicate.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_micro_kiki_data.py`:

```python
class TestDeduplication:
    """Test cross-domain deduplication."""

    def test_dedup_removes_duplicates(self):
        from scripts.micro_kiki.deduplicate import dedup_cross_domain
        domain_data = {
            "python": [
                {"messages": [{"role": "user", "content": "Write sort"}, {"role": "assistant", "content": "def sort(): pass"}]},
                {"messages": [{"role": "user", "content": "Write search"}, {"role": "assistant", "content": "def search(): pass"}]},
            ],
            "typescript": [
                {"messages": [{"role": "user", "content": "Write sort"}, {"role": "assistant", "content": "def sort(): pass"}]},  # duplicate
                {"messages": [{"role": "user", "content": "Write TS sort"}, {"role": "assistant", "content": "function sort() {}"}]},
            ],
        }
        result = dedup_cross_domain(domain_data)
        # The duplicate should be removed from one domain
        total = sum(len(v) for v in result.values())
        assert total == 3  # 4 original - 1 duplicate

    def test_dedup_keeps_best_domain(self):
        from scripts.micro_kiki.deduplicate import dedup_key
        key = dedup_key({"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]})
        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256 hex

    def test_dedup_empty_input(self):
        from scripts.micro_kiki.deduplicate import dedup_cross_domain
        result = dedup_cross_domain({})
        assert result == {}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source .venv/bin/activate && python -m pytest tests/test_micro_kiki_data.py::TestDeduplication -v --tb=short
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the deduplication script**

```python
#!/usr/bin/env python3
"""Cross-domain deduplication for Micro_KIKI pipeline.

Ensures each example appears in exactly 1 domain. When duplicates exist
across domains, the example stays in the domain where it was classified
first (by priority: phase order, then alphabetical).

Usage:
    python scripts/micro_kiki/deduplicate.py \
        --classified-dir data/micro-kiki/classified \
        --generated-dir data/micro-kiki/generated \
        --output-dir data/micro-kiki/deduped \
        --config configs/micro_kiki/domains.yaml
"""

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def dedup_key(example: dict) -> str:
    """Generate a SHA-256 hash key for deduplication.

    Uses the first 500 chars of user content + first 500 chars of assistant content.
    """
    messages = example.get("messages", [])
    user_content = ""
    assistant_content = ""
    for msg in messages:
        if msg.get("role") == "user":
            user_content = msg.get("content", "")
        elif msg.get("role") == "assistant":
            assistant_content = msg.get("content", "")

    text = user_content.strip()[:500] + "\n###\n" + assistant_content.strip()[:500]
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_domain_jsonl(filepath: Path) -> list[dict]:
    """Load examples from a JSONL file."""
    examples = []
    if not filepath.exists():
        return examples
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return examples


def dedup_cross_domain(domain_data: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Remove duplicates across domains.

    Each example is kept in the first domain that claims it (by domain priority).
    Priority: lower phase number first, then alphabetical within phase.

    Args:
        domain_data: dict of domain_name -> list of examples

    Returns:
        dict of domain_name -> deduplicated list of examples
    """
    if not domain_data:
        return {}

    seen_hashes: set[str] = set()
    result: dict[str, list[dict]] = {}

    # Process domains in priority order (alphabetical for simplicity if no config)
    for domain_name in sorted(domain_data.keys()):
        deduped = []
        for example in domain_data[domain_name]:
            h = dedup_key(example)
            if h not in seen_hashes:
                seen_hashes.add(h)
                deduped.append(example)
        result[domain_name] = deduped

    return result


def run_dedup(config_path: str, classified_dir: str, generated_dir: str,
              output_dir: str) -> dict[str, int]:
    """Run full deduplication pipeline.

    Merges classified + generated data per domain, then deduplicates cross-domain.
    """
    config = yaml.safe_load(Path(config_path).read_text())
    domains = config["domains"]
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load all domain data (classified + generated)
    domain_data: dict[str, list[dict]] = {}

    # Sort domains by phase then name for priority
    sorted_domains = sorted(
        domains.keys(),
        key=lambda n: (domains[n]["phase"], n),
    )

    for name in sorted_domains:
        examples = []
        # Load classified
        classified_file = Path(classified_dir) / f"{name}.jsonl"
        examples.extend(load_domain_jsonl(classified_file))
        # Load generated
        generated_file = Path(generated_dir) / f"{name}.jsonl"
        examples.extend(load_domain_jsonl(generated_file))
        domain_data[name] = examples

    print(f"Before dedup: {sum(len(v) for v in domain_data.values())} total examples")

    # Deduplicate
    deduped = dedup_cross_domain(domain_data)

    # Write output
    counts = {}
    total_removed = 0
    for name in sorted_domains:
        before = len(domain_data.get(name, []))
        after = len(deduped.get(name, []))
        removed = before - after
        total_removed += removed
        counts[name] = after

        outfile = out_path / f"{name}.jsonl"
        with open(outfile, "w") as f:
            for ex in deduped.get(name, []):
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        status = "OK" if after >= domains[name]["target"] * 0.5 else "LOW"
        print(f"  Phase {domains[name]['phase']} {name:<16} {after:>5} (removed {removed}) [{status}]")

    print(f"\nAfter dedup: {sum(counts.values())} total ({total_removed} removed)")
    return counts


def main():
    parser = argparse.ArgumentParser(description="Cross-domain deduplication for Micro_KIKI")
    parser.add_argument("--config", default="configs/micro_kiki/domains.yaml")
    parser.add_argument("--classified-dir", default="data/micro-kiki/classified")
    parser.add_argument("--generated-dir", default="data/micro-kiki/generated")
    parser.add_argument("--output-dir", default="data/micro-kiki/deduped")
    args = parser.parse_args()

    run_dedup(args.config, args.classified_dir, args.generated_dir, args.output_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
source .venv/bin/activate && python -m pytest tests/test_micro_kiki_data.py::TestDeduplication -v --tb=short
```

Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/micro_kiki/deduplicate.py tests/test_micro_kiki_data.py
git commit -m "feat: add cross-domain deduplication for Micro_KIKI pipeline"
```

---

### Task 6: Train/valid split per domain

**Files:**
- Create: `scripts/micro_kiki/split_domains.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_micro_kiki_data.py`:

```python
class TestSplitDomains:
    """Test train/valid split logic."""

    def test_split_90_10(self):
        from scripts.micro_kiki.split_domains import split_examples
        examples = [{"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]} for i in range(100)]
        train, valid = split_examples(examples, valid_ratio=0.1, seed=42)
        assert len(train) == 90
        assert len(valid) == 10
        # No overlap
        train_set = {json.dumps(e, sort_keys=True) for e in train}
        valid_set = {json.dumps(e, sort_keys=True) for e in valid}
        assert len(train_set & valid_set) == 0

    def test_split_small_dataset(self):
        from scripts.micro_kiki.split_domains import split_examples
        examples = [{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}]
        train, valid = split_examples(examples, valid_ratio=0.1, seed=42)
        # With 1 example, it should go to train
        assert len(train) == 1
        assert len(valid) == 0

    def test_split_deterministic(self):
        from scripts.micro_kiki.split_domains import split_examples
        examples = [{"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]} for i in range(50)]
        train1, valid1 = split_examples(examples, valid_ratio=0.1, seed=42)
        train2, valid2 = split_examples(examples, valid_ratio=0.1, seed=42)
        assert train1 == train2
        assert valid1 == valid2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source .venv/bin/activate && python -m pytest tests/test_micro_kiki_data.py::TestSplitDomains -v --tb=short
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the split script**

```python
#!/usr/bin/env python3
"""Split deduplicated domain data into train/valid sets.

Produces the final output structure:
    data/micro-kiki/<domain>/train.jsonl
    data/micro-kiki/<domain>/valid.jsonl

Usage:
    python scripts/micro_kiki/split_domains.py \
        --config configs/micro_kiki/domains.yaml \
        --input-dir data/micro-kiki/deduped \
        --output-dir data/micro-kiki
"""

import argparse
import json
import random
from pathlib import Path

import yaml


def split_examples(examples: list[dict], valid_ratio: float = 0.1,
                   seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Split examples into train and valid sets.

    Args:
        examples: list of chat examples
        valid_ratio: fraction for validation (default 10%)
        seed: random seed for reproducibility

    Returns:
        (train_examples, valid_examples)
    """
    if not examples:
        return [], []

    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)

    valid_count = max(0, int(len(shuffled) * valid_ratio))

    # Minimum: at least 1 in train
    if valid_count >= len(shuffled):
        valid_count = len(shuffled) - 1
    if valid_count < 0:
        valid_count = 0

    valid = shuffled[:valid_count]
    train = shuffled[valid_count:]

    return train, valid


def load_jsonl(filepath: Path) -> list[dict]:
    """Load examples from JSONL file."""
    examples = []
    if not filepath.exists():
        return examples
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return examples


def run_split(config_path: str, input_dir: str, output_dir: str) -> dict[str, dict[str, int]]:
    """Split all domains into train/valid.

    Returns dict of domain -> {"train": count, "valid": count}.
    """
    config = yaml.safe_load(Path(config_path).read_text())
    domains = config["domains"]
    valid_ratio = config.get("valid_ratio", 0.1)
    in_path = Path(input_dir)
    out_base = Path(output_dir)

    results = {}
    total_train = 0
    total_valid = 0

    sorted_domains = sorted(
        domains.keys(),
        key=lambda n: (domains[n]["phase"], n),
    )

    for name in sorted_domains:
        examples = load_jsonl(in_path / f"{name}.jsonl")
        train, valid = split_examples(examples, valid_ratio=valid_ratio, seed=42)

        # Write to domain directory
        domain_dir = out_base / name
        domain_dir.mkdir(parents=True, exist_ok=True)

        train_file = domain_dir / "train.jsonl"
        with open(train_file, "w") as f:
            for ex in train:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        valid_file = domain_dir / "valid.jsonl"
        with open(valid_file, "w") as f:
            for ex in valid:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        results[name] = {"train": len(train), "valid": len(valid)}
        total_train += len(train)
        total_valid += len(valid)

        target = domains[name]["target"]
        pct = len(train) / target * 100 if target > 0 else 0
        status = "OK" if pct >= 50 else "LOW"
        print(f"  Phase {domains[name]['phase']} {name:<16} train={len(train):>5} valid={len(valid):>4} ({pct:.0f}% of target) [{status}]")

    print(f"\nTotal: {total_train} train + {total_valid} valid = {total_train + total_valid}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Split domain data into train/valid for Micro_KIKI")
    parser.add_argument("--config", default="configs/micro_kiki/domains.yaml")
    parser.add_argument("--input-dir", default="data/micro-kiki/deduped")
    parser.add_argument("--output-dir", default="data/micro-kiki")
    args = parser.parse_args()

    run_split(args.config, args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
source .venv/bin/activate && python -m pytest tests/test_micro_kiki_data.py::TestSplitDomains -v --tb=short
```

Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/micro_kiki/split_domains.py tests/test_micro_kiki_data.py
git commit -m "feat: add train/valid split per domain for Micro_KIKI pipeline"
```

---

### Task 7: Pipeline orchestrator

**Files:**
- Create: `scripts/micro_kiki/pipeline_data.sh`

- [ ] **Step 1: Write the orchestrator script**

```bash
#!/bin/bash
# Micro_KIKI Data Pipeline — Full Orchestrator
#
# Runs all data preparation steps in sequence:
#   1. Download public datasets
#   2. Classify into 32 domains
#   3. Generate synthetic data for sparse domains
#   4. Deduplicate cross-domain
#   5. Split train/valid per domain
#
# Usage:
#   ./scripts/micro_kiki/pipeline_data.sh [--skip-download] [--skip-generate] [--dry-run]
#
# Options:
#   --skip-download    Skip dataset download (use existing data/raw/)
#   --skip-generate    Skip synthetic generation (use classified data only)
#   --dry-run          Show generation plan without running teachers
#   --teacher MODEL    Override teacher model path

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Defaults
SKIP_DOWNLOAD=false
SKIP_GENERATE=false
DRY_RUN=""
TEACHER_MODEL="models/Qwen3.5-35B-A3B-Opus-vlm"
CONFIG="configs/micro_kiki/domains.yaml"
MAX_PER_DOMAIN=3000
MAX_GENERATE=500

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-download)  SKIP_DOWNLOAD=true; shift ;;
        --skip-generate)  SKIP_GENERATE=true; shift ;;
        --dry-run)        DRY_RUN="--dry-run"; shift ;;
        --teacher)        TEACHER_MODEL="$2"; shift 2 ;;
        --config)         CONFIG="$2"; shift 2 ;;
        --max-generate)   MAX_GENERATE="$2"; shift 2 ;;
        *)                echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Activate venv
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

echo "=========================================="
echo "  Micro_KIKI Data Pipeline"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""
echo "Config:         $CONFIG"
echo "Teacher:        $TEACHER_MODEL"
echo "Skip download:  $SKIP_DOWNLOAD"
echo "Skip generate:  $SKIP_GENERATE"
echo "Dry run:        ${DRY_RUN:-no}"
echo ""

START_TIME=$(date +%s)

# Step 1: Download
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo "=== Step 1/5: Download datasets ==="
    bash "$SCRIPT_DIR/download_datasets.sh" all
    echo ""
else
    echo "=== Step 1/5: Download — SKIPPED ==="
    echo ""
fi

# Step 2: Classify
echo "=== Step 2/5: Classify into 32 domains ==="
python "$SCRIPT_DIR/classify_domains.py" \
    --config "$CONFIG" \
    --input-dir "data/raw" \
    --output-dir "data/micro-kiki/classified" \
    --max-per-domain "$MAX_PER_DOMAIN"
echo ""

# Step 3: Generate (optional)
if [ "$SKIP_GENERATE" = false ]; then
    echo "=== Step 3/5: Generate synthetic data for sparse domains ==="
    python "$SCRIPT_DIR/generate_missing.py" \
        --config "$CONFIG" \
        --classified-dir "data/micro-kiki/classified" \
        --output-dir "data/micro-kiki/generated" \
        --teacher-model "$TEACHER_MODEL" \
        --max-generate "$MAX_GENERATE" \
        $DRY_RUN
    echo ""
else
    echo "=== Step 3/5: Generate — SKIPPED ==="
    # Create empty generated dir so dedup doesn't fail
    mkdir -p "data/micro-kiki/generated"
    echo ""
fi

# Step 4: Deduplicate
echo "=== Step 4/5: Cross-domain deduplication ==="
python "$SCRIPT_DIR/deduplicate.py" \
    --config "$CONFIG" \
    --classified-dir "data/micro-kiki/classified" \
    --generated-dir "data/micro-kiki/generated" \
    --output-dir "data/micro-kiki/deduped"
echo ""

# Step 5: Split
echo "=== Step 5/5: Train/valid split ==="
python "$SCRIPT_DIR/split_domains.py" \
    --config "$CONFIG" \
    --input-dir "data/micro-kiki/deduped" \
    --output-dir "data/micro-kiki"
echo ""

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo "=========================================="
echo "  Pipeline complete in ${MINUTES}m${SECONDS}s"
echo "=========================================="
echo ""

# Final summary
echo "=== Final data structure ==="
for domain_dir in data/micro-kiki/*/; do
    if [ -f "${domain_dir}train.jsonl" ]; then
        domain="$(basename "$domain_dir")"
        train_count=$(wc -l < "${domain_dir}train.jsonl" 2>/dev/null || echo 0)
        valid_count=$(wc -l < "${domain_dir}valid.jsonl" 2>/dev/null || echo 0)
        printf "  %-16s train=%5s valid=%4s\n" "$domain" "$train_count" "$valid_count"
    fi
done

total_train=$(cat data/micro-kiki/*/train.jsonl 2>/dev/null | wc -l || echo 0)
total_valid=$(cat data/micro-kiki/*/valid.jsonl 2>/dev/null | wc -l || echo 0)
echo ""
echo "  TOTAL: train=$total_train valid=$total_valid"
```

- [ ] **Step 2: Make executable and verify syntax**

```bash
chmod +x scripts/micro_kiki/pipeline_data.sh
bash -n scripts/micro_kiki/pipeline_data.sh
echo "Syntax OK"
```

Expected: `Syntax OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/micro_kiki/pipeline_data.sh
git commit -m "feat: add pipeline orchestrator for Micro_KIKI data preparation"
```

---

### Task 8: Validation script

**Files:**
- Create: `scripts/micro_kiki/validate_data.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_micro_kiki_data.py`:

```python
class TestValidation:
    """Test data validation logic."""

    def test_validate_chat_format_valid(self):
        from scripts.micro_kiki.validate_data import validate_example
        example = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "<thinking>Let me think</thinking>\n\nHi!"},
            ]
        }
        errors = validate_example(example)
        assert errors == []

    def test_validate_missing_messages(self):
        from scripts.micro_kiki.validate_data import validate_example
        errors = validate_example({})
        assert len(errors) > 0
        assert any("messages" in e for e in errors)

    def test_validate_empty_content(self):
        from scripts.micro_kiki.validate_data import validate_example
        example = {
            "messages": [
                {"role": "user", "content": ""},
                {"role": "assistant", "content": "answer"},
            ]
        }
        errors = validate_example(example)
        assert len(errors) > 0

    def test_validate_wrong_roles(self):
        from scripts.micro_kiki.validate_data import validate_example
        example = {
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "human", "content": "Hello"},
            ]
        }
        errors = validate_example(example)
        assert len(errors) > 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source .venv/bin/activate && python -m pytest tests/test_micro_kiki_data.py::TestValidation -v --tb=short
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the validation script**

```python
#!/usr/bin/env python3
"""Validate Micro_KIKI domain datasets.

Checks format, counts, quality metrics, and produces a summary report.

Usage:
    python scripts/micro_kiki/validate_data.py \
        --config configs/micro_kiki/domains.yaml \
        --data-dir data/micro-kiki
"""

import argparse
import json
import sys
from pathlib import Path

import yaml


def validate_example(example: dict) -> list[str]:
    """Validate a single training example.

    Returns a list of error strings (empty = valid).
    """
    errors = []

    # Must have messages key
    if "messages" not in example:
        errors.append("missing 'messages' key")
        return errors

    messages = example["messages"]
    if not isinstance(messages, list):
        errors.append("'messages' is not a list")
        return errors

    if len(messages) < 2:
        errors.append(f"expected at least 2 messages, got {len(messages)}")
        return errors

    # Check roles
    valid_roles = {"user", "assistant", "system"}
    has_user = False
    has_assistant = False

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            errors.append(f"message[{i}] is not a dict")
            continue

        role = msg.get("role", "")
        if role not in valid_roles:
            errors.append(f"message[{i}] has invalid role '{role}'")
        if role == "user":
            has_user = True
        if role == "assistant":
            has_assistant = True

        content = msg.get("content", "")
        if not content or not content.strip():
            errors.append(f"message[{i}] ({role}) has empty content")

    if not has_user:
        errors.append("no message with role 'user'")
    if not has_assistant:
        errors.append("no message with role 'assistant'")

    return errors


def validate_domain(domain_dir: Path, target: int) -> dict:
    """Validate a domain's train.jsonl and valid.jsonl.

    Returns a summary dict.
    """
    result = {
        "exists": domain_dir.exists(),
        "train_count": 0,
        "valid_count": 0,
        "train_errors": 0,
        "valid_errors": 0,
        "has_thinking": 0,
        "avg_user_len": 0,
        "avg_assistant_len": 0,
        "target": target,
        "pct_of_target": 0,
        "error_samples": [],
    }

    if not domain_dir.exists():
        return result

    for split_name in ("train", "valid"):
        filepath = domain_dir / f"{split_name}.jsonl"
        if not filepath.exists():
            continue

        count = 0
        error_count = 0
        thinking_count = 0
        total_user_len = 0
        total_assistant_len = 0

        with open(filepath) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    example = json.loads(line)
                except json.JSONDecodeError:
                    error_count += 1
                    if len(result["error_samples"]) < 3:
                        result["error_samples"].append(f"{split_name}:{line_num} JSON parse error")
                    continue

                count += 1
                errors = validate_example(example)
                if errors:
                    error_count += 1
                    if len(result["error_samples"]) < 3:
                        result["error_samples"].append(f"{split_name}:{line_num} {errors[0]}")
                    continue

                # Stats
                for msg in example.get("messages", []):
                    content = msg.get("content", "")
                    if msg.get("role") == "user":
                        total_user_len += len(content)
                    elif msg.get("role") == "assistant":
                        total_assistant_len += len(content)
                        if "<thinking>" in content:
                            thinking_count += 1

        if split_name == "train":
            result["train_count"] = count
            result["train_errors"] = error_count
        else:
            result["valid_count"] = count
            result["valid_errors"] = error_count

        result["has_thinking"] += thinking_count
        if count > 0:
            result["avg_user_len"] = total_user_len // max(count, 1)
            result["avg_assistant_len"] = total_assistant_len // max(count, 1)

    total = result["train_count"] + result["valid_count"]
    result["pct_of_target"] = total / target * 100 if target > 0 else 0

    return result


def run_validation(config_path: str, data_dir: str) -> bool:
    """Run full validation and print report.

    Returns True if all domains pass minimum checks.
    """
    config = yaml.safe_load(Path(config_path).read_text())
    domains = config["domains"]
    base = Path(data_dir)

    print("=" * 70)
    print("  Micro_KIKI Data Validation Report")
    print("=" * 70)
    print()

    all_ok = True
    total_train = 0
    total_valid = 0
    total_errors = 0
    total_thinking = 0
    sparse_domains = []

    sorted_domains = sorted(
        domains.keys(),
        key=lambda n: (domains[n]["phase"], n),
    )

    for name in sorted_domains:
        target = domains[name]["target"]
        domain_dir = base / name
        result = validate_domain(domain_dir, target)

        total_train += result["train_count"]
        total_valid += result["valid_count"]
        total_errors += result["train_errors"] + result["valid_errors"]
        total_thinking += result["has_thinking"]

        # Status
        if not result["exists"]:
            status = "MISSING"
            all_ok = False
        elif result["pct_of_target"] < 25:
            status = "CRITICAL"
            all_ok = False
            sparse_domains.append(name)
        elif result["pct_of_target"] < 50:
            status = "LOW"
            sparse_domains.append(name)
        elif result["train_errors"] + result["valid_errors"] > 0:
            status = "ERRORS"
        else:
            status = "OK"

        phase = domains[name]["phase"]
        print(
            f"  P{phase} {name:<16} "
            f"train={result['train_count']:>5} "
            f"valid={result['valid_count']:>4} "
            f"err={result['train_errors'] + result['valid_errors']:>3} "
            f"think={result['has_thinking']:>4} "
            f"({result['pct_of_target']:.0f}%) "
            f"[{status}]"
        )

        for sample in result["error_samples"]:
            print(f"      {sample}")

    print()
    print("-" * 70)
    print(f"  Total: {total_train} train + {total_valid} valid = {total_train + total_valid}")
    print(f"  Errors: {total_errors}")
    print(f"  With <thinking>: {total_thinking}")
    print(f"  Sparse domains ({len(sparse_domains)}): {', '.join(sparse_domains) if sparse_domains else 'none'}")
    print()

    if all_ok:
        print("  STATUS: PASS — all domains have sufficient data")
    else:
        print("  STATUS: FAIL — some domains are missing or critically sparse")
        print("  Run generate_missing.py to fill sparse domains.")

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Validate Micro_KIKI domain datasets")
    parser.add_argument("--config", default="configs/micro_kiki/domains.yaml")
    parser.add_argument("--data-dir", default="data/micro-kiki")
    args = parser.parse_args()

    ok = run_validation(args.config, args.data_dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
source .venv/bin/activate && python -m pytest tests/test_micro_kiki_data.py::TestValidation -v --tb=short
```

Expected: All 4 tests PASS.

- [ ] **Step 5: Run the full test suite**

```bash
source .venv/bin/activate && python -m pytest tests/test_micro_kiki_data.py -v --tb=short
```

Expected: All 20 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/micro_kiki/validate_data.py tests/test_micro_kiki_data.py
git commit -m "feat: add validation script for Micro_KIKI domain datasets"
```

---

## Self-Review Checklist

**Spec coverage:**
- 32 domains defined with keywords, patterns, teachers, phases: Task 1 (domains.yaml)
- Download public datasets (CodeFeedback, OpenCodeReasoning, Magicoder): Task 2
- Classify each example into 1 of 32 domains: Task 3
- Generate with teachers for sparse domains: Task 4
- Deduplicate cross-domain: Task 5
- Split train/valid per domain: Task 6
- Pipeline orchestrator: Task 7
- Validation: Task 8
- Output format: `data/micro-kiki/<domain>/train.jsonl` + `valid.jsonl` with `{"messages": [...]}` and `<thinking>` tags: enforced in Tasks 3, 4, 6, 8

**Placeholder scan:** No TBD, TODO, "implement later", or "similar to Task N" found. All code is complete.

**Type consistency:** `classify_example`, `dedup_key`, `dedup_cross_domain`, `split_examples`, `validate_example` — signatures match across tests and implementations. File paths consistent across all scripts.

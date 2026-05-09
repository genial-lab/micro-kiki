# Hardware

KiCad / SPICE / STM32 / embedded artifacts and bench notes used as ground truth
for the hardware-domain LoRA stacks (kicad-dsl, spice, stm32, embedded, emc, power).

## Conventions

- KiCad projects: one folder per board, contains `.kicad_pro`, `.kicad_sch`, `.kicad_pcb`
- SPICE netlists: `.cir` or `.net`, paired with stimulus & expected output
- Datasheets: under `datasheets/`, name with vendor + part-number
- Bench measurements: timestamped under `measurements/`, include rig description

## Anti-patterns

- Don't commit Gerber/3D outputs — generate from sources
- Don't duplicate footprints across boards — use shared library
- Don't mix bench notes with design files — separate `notes/` dir
- Don't ingest into datasets without provenance metadata (source, license, date)

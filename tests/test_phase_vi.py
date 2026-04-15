"""Validate Phase VI configs, eval files, and import script."""
from __future__ import annotations

import json
import pytest
from pathlib import Path

import yaml

PHASE_VI_STACKS = [
    (15, "embedded"), (16, "stm32"), (17, "iot"), (18, "freecad"),
    (19, "platformio"), (20, "power"), (21, "emc"), (22, "dsp"),
    (23, "spice-sim"), (24, "electronics"), (25, "kicad-pcb"),
]

REQUIRED_CONFIG_KEYS = [
    "base_model", "num_experts", "lora_rank", "lora_alpha", "top_k",
    "learning_rate", "batch_size", "grad_accum", "epochs", "seq_len", "dataset",
    "domain", "curriculum_order",
]


class TestPhaseVIConfigs:
    @pytest.mark.parametrize("num,domain", PHASE_VI_STACKS)
    def test_config_valid(self, num, domain):
        path = Path(f"configs/stack-{num:02d}-{domain}.yaml")
        assert path.exists()
        with open(path) as f:
            config = yaml.safe_load(f)
        for key in REQUIRED_CONFIG_KEYS:
            assert key in config
        assert config["curriculum_order"] == num
        assert config["init_lora_weights"] == "oplora"

    @pytest.mark.parametrize("num,domain", PHASE_VI_STACKS)
    def test_eval_file_exists(self, num, domain):
        path = Path(f"data/eval/{domain}.jsonl")
        assert path.exists()
        lines = [l for l in path.read_text().strip().split("\n") if l]
        assert len(lines) >= 5


class TestImportScript:
    def test_import_script_exists(self):
        assert Path("scripts/import_kiki_datasets.py").exists()

    def test_kiki_sources_cover_all_domains(self):
        from scripts.import_kiki_datasets import KIKI_SOURCES
        for _, domain in PHASE_VI_STACKS:
            assert domain in KIKI_SOURCES, f"Missing {domain} in KIKI_SOURCES"


class TestGroupEval:
    def test_group_eval_script_exists(self):
        assert Path("scripts/group_eval.py").exists()

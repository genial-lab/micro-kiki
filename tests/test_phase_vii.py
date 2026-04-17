"""Validate 10-niche pivot: configs, eval files, and scripts.

After the 2026-04-16 pivot from 32 LoRA stacks to 10 niche domains,
this test validates that the niche infrastructure is complete.
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path

import yaml

from src.routing.router import NICHE_DOMAINS

# The 10 niche domains and their stack numbers
NICHE_STACKS = [
    (12, "kicad-dsl"),
    (13, "spice"),
    (15, "embedded"),
    (16, "stm32"),
    (18, "freecad"),
    (19, "platformio"),
    (20, "power"),
    (21, "emc"),
    (22, "dsp"),
    (24, "electronics"),
]

REQUIRED_CONFIG_KEYS = [
    "base_model", "lora_rank", "lora_alpha",
    "learning_rate", "batch_size", "grad_accum", "epochs", "seq_len", "dataset",
    "domain", "curriculum_order",
]


class TestNicheDomainConstants:
    def test_niche_domains_has_10(self):
        assert len(NICHE_DOMAINS) == 10

    def test_niche_stacks_match_router(self):
        stack_domains = {domain for _, domain in NICHE_STACKS}
        assert stack_domains == NICHE_DOMAINS


class TestNicheConfigs:
    @pytest.mark.parametrize("num,domain", NICHE_STACKS)
    def test_config_valid(self, num, domain):
        path = Path(f"configs/stack-{num:02d}-{domain}.yaml")
        assert path.exists(), f"Missing config: {path}"
        with open(path) as f:
            config = yaml.safe_load(f)
        for key in REQUIRED_CONFIG_KEYS:
            assert key in config
        assert config["curriculum_order"] == num
        assert "35B-A3B" in config["base_model"]

    @pytest.mark.parametrize("num,domain", NICHE_STACKS)
    def test_eval_file_exists(self, num, domain):
        path = Path(f"data/eval/{domain}.jsonl")
        assert path.exists(), f"Missing eval: {path}"
        lines = [l for l in path.read_text().strip().split("\n") if l]
        assert len(lines) >= 5
        for line in lines:
            assert "prompt" in json.loads(line)


class TestNicheScripts:
    def test_distill_script_exists(self):
        assert Path("scripts/distill_domain.py").exists()

    def test_full_eval_script_exists(self):
        assert Path("scripts/run_full_eval.sh").exists()

    def test_full_eval_script_executable(self):
        path = Path("scripts/run_full_eval.sh")
        first_line = path.read_text().split("\n")[0]
        assert first_line.startswith("#!/")

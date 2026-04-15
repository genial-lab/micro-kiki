"""Validate Phase VII configs, eval files, and scripts."""
from __future__ import annotations

import json
import pytest
from pathlib import Path

import yaml

PHASE_VII_STACKS = [
    (26, "web-frontend"), (27, "web-backend"), (28, "music-audio"),
    (29, "devops"), (30, "llm-orch"), (31, "math"), (32, "security"),
]

REQUIRED_CONFIG_KEYS = [
    "base_model", "num_experts", "lora_rank", "lora_alpha", "top_k",
    "learning_rate", "batch_size", "grad_accum", "epochs", "seq_len", "dataset",
    "domain", "curriculum_order",
]


class TestPhaseVIIConfigs:
    @pytest.mark.parametrize("num,domain", PHASE_VII_STACKS)
    def test_config_valid(self, num, domain):
        path = Path(f"configs/stack-{num:02d}-{domain}.yaml")
        assert path.exists()
        with open(path) as f:
            config = yaml.safe_load(f)
        for key in REQUIRED_CONFIG_KEYS:
            assert key in config
        assert config["curriculum_order"] == num
        assert config["init_lora_weights"] == "oplora"

    @pytest.mark.parametrize("num,domain", PHASE_VII_STACKS)
    def test_eval_file_exists(self, num, domain):
        path = Path(f"data/eval/{domain}.jsonl")
        assert path.exists()
        lines = [l for l in path.read_text().strip().split("\n") if l]
        assert len(lines) >= 5
        for line in lines:
            assert "prompt" in json.loads(line)


class TestPhaseVIIScripts:
    def test_distill_script_exists(self):
        assert Path("scripts/distill_apps_domains.py").exists()

    def test_full_eval_script_exists(self):
        assert Path("scripts/run_full_eval.sh").exists()

    def test_full_eval_script_executable(self):
        path = Path("scripts/run_full_eval.sh")
        # Check it has shebang
        first_line = path.read_text().split("\n")[0]
        assert first_line.startswith("#!/")


class TestAllStacksCovered:
    def test_32_stack_configs_exist(self):
        """All 32 domains have a config file."""
        configs = sorted(Path("configs").glob("stack-*.yaml"))
        nums = [int(c.stem.split("-")[1]) for c in configs]
        for i in range(1, 33):
            assert i in nums, f"Missing stack config for stack-{i:02d}"

    def test_all_eval_files_exist(self):
        """At least one eval file per stack config."""
        configs = sorted(Path("configs").glob("stack-*.yaml"))
        for cfg in configs:
            with open(cfg) as f:
                domain = yaml.safe_load(f)["domain"]
            eval_path = Path(f"data/eval/{domain}.jsonl")
            assert eval_path.exists(), f"Missing eval for {domain}"

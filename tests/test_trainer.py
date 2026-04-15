from __future__ import annotations

import pytest
from pathlib import Path

from src.stacks.trainer import load_training_config, molora_config_from_dict, StackTrainer
from src.stacks.moe_lora import MoLoRAConfig


class TestLoadConfig:
    def test_loads_yaml(self, tmp_path):
        cfg = tmp_path / "s.yaml"
        cfg.write_text("base_model: x\nnum_experts: 4\nlora_rank: 16\nlora_alpha: 32\ntop_k: 2\nlearning_rate: 0.0002\nbatch_size: 4\ngrad_accum: 8\nepochs: 3\nseq_len: 4096\ndataset: d.jsonl\ninit_lora_weights: pissa\n")
        config = load_training_config(cfg)
        assert config["lora_rank"] == 16

    def test_missing_key_raises(self, tmp_path):
        (tmp_path / "bad.yaml").write_text("lora_rank: 16\n")
        with pytest.raises(KeyError):
            load_training_config(tmp_path / "bad.yaml")


class TestStackTrainer:
    def test_init(self):
        t = StackTrainer("models/x", MoLoRAConfig(), "outputs/test")
        assert t.output_dir == "outputs/test"

    def test_config_from_dict(self):
        d = {"lora_rank": 16, "num_experts": 4, "top_k": 2, "lora_alpha": 32,
             "base_model": "x", "dataset": "y", "learning_rate": 2e-4,
             "batch_size": 4, "grad_accum": 8, "epochs": 3, "seq_len": 4096}
        mc = molora_config_from_dict(d)
        assert mc.rank == 16 and mc.num_experts == 4

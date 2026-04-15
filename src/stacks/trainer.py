"""MoE-LoRA stack trainer."""
from __future__ import annotations

import logging
from pathlib import Path

import yaml

from src.stacks.moe_lora import MoLoRAConfig

logger = logging.getLogger(__name__)

REQUIRED_KEYS = [
    "base_model", "num_experts", "lora_rank", "lora_alpha", "top_k",
    "learning_rate", "batch_size", "grad_accum", "epochs", "seq_len", "dataset",
]


def load_training_config(config_path: Path | str) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    for key in REQUIRED_KEYS:
        if key not in config:
            raise KeyError(f"Missing required config key: {key}")
    return config


def molora_config_from_dict(config: dict) -> MoLoRAConfig:
    return MoLoRAConfig(
        rank=config["lora_rank"], num_experts=config["num_experts"],
        top_k=config["top_k"], alpha=config["lora_alpha"],
    )


class StackTrainer:
    def __init__(self, base_model_path: str, molora_config: MoLoRAConfig,
                 output_dir: str, learning_rate: float = 2e-4, batch_size: int = 4,
                 grad_accum: int = 8, epochs: int = 3, seq_len: int = 4096,
                 init_lora_weights: str = "pissa", pissa_niter: int = 4) -> None:
        self.base_model_path = base_model_path
        self.molora_config = molora_config
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.grad_accum = grad_accum
        self.epochs = epochs
        self.seq_len = seq_len
        self.init_lora_weights = init_lora_weights
        self.pissa_niter = pissa_niter

    @classmethod
    def from_config(cls, config_path: str | Path) -> StackTrainer:
        config = load_training_config(config_path)
        mc = molora_config_from_dict(config)
        return cls(
            base_model_path=config["base_model"], molora_config=mc,
            output_dir=f"outputs/stacks/{Path(config_path).stem}",
            learning_rate=config["learning_rate"], batch_size=config["batch_size"],
            grad_accum=config["grad_accum"], epochs=config["epochs"],
            seq_len=config["seq_len"],
            init_lora_weights=config.get("init_lora_weights", "pissa"),
            pissa_niter=config.get("pissa_niter", 4),
        )

    def train(self, dataset_path: str) -> dict:
        """Train the stack. Requires torch + transformers + peft + trl."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from peft import LoraConfig, get_peft_model
        from trl import SFTTrainer
        from datasets import load_dataset

        tokenizer = AutoTokenizer.from_pretrained(self.base_model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        )
        lora_config = LoraConfig(
            r=self.molora_config.rank, lora_alpha=self.molora_config.alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            init_lora_weights=self.init_lora_weights,
        )
        model = get_peft_model(model, lora_config)
        dataset = load_dataset("json", data_files=dataset_path, split="train")

        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(output_path), per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.grad_accum, num_train_epochs=self.epochs,
            learning_rate=self.learning_rate, bf16=True, logging_steps=10, save_strategy="epoch",
        )
        trainer = SFTTrainer(model=model, args=training_args, train_dataset=dataset, tokenizer=tokenizer)
        result = trainer.train()
        model.save_pretrained(str(output_path))
        tokenizer.save_pretrained(str(output_path))
        return {"train_loss": result.training_loss, "output_dir": str(output_path), "epochs": self.epochs}

"""Train stack-03 python LoRA on Qwen3.5-4B."""
import os
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

import json
import logging
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_MODEL = "/home/kxkm/models/qwen3.5-4b/bf16/"
DATASET = "/home/kxkm/micro-kiki/data/distilled/python.jsonl"
OUTPUT_DIR = "/home/kxkm/micro-kiki/outputs/stacks/stack-03-python"
LORA_RANK = 16
LORA_ALPHA = 32
LEARNING_RATE = 2e-4
BATCH_SIZE = 2
GRAD_ACCUM = 16
EPOCHS = 3
SEQ_LEN = 2048

def format_example(example):
    return {"text": f"<|im_start|>user\n{example['prompt']}<|im_end|>\n<|im_start|>assistant\n{example['completion']}<|im_end|>"}

def main():
    logger.info("Loading tokenizer from %s", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model from %s", BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    peft_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    logger.info("Loading dataset from %s", DATASET)
    dataset = load_dataset("json", data_files=DATASET, split="train")
    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    logger.info("Dataset size: %d examples", len(dataset))

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=str(output_path),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        max_grad_norm=1.0,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        dataloader_num_workers=0,
        report_to="none",
        max_length=SEQ_LEN,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting training...")
    result = trainer.train()

    logger.info("Saving adapter to %s", OUTPUT_DIR)
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    metrics = {
        "train_loss": result.training_loss,
        "output_dir": str(output_path),
        "epochs": EPOCHS,
        "base_model": BASE_MODEL,
        "lora_rank": LORA_RANK,
        "dataset_size": len(dataset),
    }
    with open(output_path / "train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Training complete: %s", json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""E2E smoke test: prompt -> router -> select stacks -> generate.

Tests the full pipeline with 3 test cases:
- French chat -> chat-fr dominant
- Math reasoning -> reasoning dominant
- Python code -> python dominant

Usage:
    cd /home/kxkm/micro-kiki
    UNSLOTH_COMPILE_DISABLE=1 /home/kxkm/KIKI-models-tuning/.venv/bin/python \
        scripts/smoke_e2e.py 2>&1 | tail -30
"""
import json
import logging
import os
import time

os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

from pathlib import Path

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_MODEL = "/home/kxkm/models/qwen3.5-4b/bf16/"
ROUTER_PATH = "/home/kxkm/micro-kiki/outputs/router/v0/router.pt"
STACKS = {
    "chat-fr": "/home/kxkm/micro-kiki/outputs/stacks/stack-01-chat-fr",
    "python": "/home/kxkm/micro-kiki/outputs/stacks/stack-03-python",
    "reasoning": "/home/kxkm/micro-kiki/outputs/stacks/stack-02-reasoning",
}
OUTPUT = "/home/kxkm/micro-kiki/results/e2e-smoke.json"

TEST_CASES = [
    {
        "prompt": "Ecris un haiku sur la pluie d'automne",
        "expected_domain": "chat-fr",
        "description": "French creative writing",
    },
    {
        "prompt": "Factor 231 into primes and explain your reasoning step by step",
        "expected_domain": "reasoning",
        "description": "Math reasoning",
    },
    {
        "prompt": "Write a Python FastAPI hello world with a health endpoint",
        "expected_domain": "python",
        "description": "Python coding",
    },
]


class RouterMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_domains=3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, num_domains),
        )

    def forward(self, x):
        return self.net(x)


def encode_prompt(text, tokenizer, model, device):
    """Encode a single prompt to embedding via mean pooling."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        hidden = outputs.last_hidden_state * attention_mask
        pooled = hidden.sum(dim=1) / attention_mask.sum(dim=1)
    return pooled


def generate(model, tokenizer, prompt, device="cpu", max_new_tokens=128):
    """Generate a response."""
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    device = "cpu"  # GPU occupied by teacher
    logger.info("Device: %s", device)

    # Check all required files exist
    if not Path(ROUTER_PATH).exists():
        logger.error("Router not found at %s", ROUTER_PATH)
        return
    for name, path in STACKS.items():
        if not Path(path).exists():
            logger.error("Stack %s not found at %s", name, path)
            return

    # Load router
    logger.info("Loading router...")
    checkpoint = torch.load(ROUTER_PATH, map_location=device, weights_only=False)
    domain_names = checkpoint["domain_names"]
    input_dim = checkpoint["input_dim"]
    num_domains = checkpoint["num_domains"]

    router = RouterMLP(input_dim=input_dim, num_domains=num_domains)
    router.load_state_dict(checkpoint["state_dict"])
    router.eval()
    router.to(device)

    # Load encoder model (for embeddings)
    logger.info("Loading encoder model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoder = AutoModel.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    encoder.eval()

    # Load base generative model
    logger.info("Loading base generative model...")
    gen_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    gen_model.eval()

    results = []
    all_pass = True

    for tc in TEST_CASES:
        logger.info("=" * 60)
        logger.info("TEST: %s", tc["description"])
        logger.info("Prompt: %s", tc["prompt"])
        t0 = time.time()

        # Step 1: Route
        embedding = encode_prompt(tc["prompt"], tokenizer, encoder, device)
        with torch.no_grad():
            logits = router(embedding.float())
            probs = torch.sigmoid(logits)[0]

        # Get top-2 domains
        sorted_indices = probs.argsort(descending=True)
        top2 = [(domain_names[i.item()], probs[i].item()) for i in sorted_indices[:2]]
        logger.info("Router: %s", [(d, f"{s:.3f}") for d, s in top2])

        # Step 2: Load top-1 adapter and generate
        top_domain = top2[0][0]
        adapter_path = STACKS.get(top_domain)

        if adapter_path and Path(adapter_path).exists():
            logger.info("Loading adapter: %s", top_domain)
            adapted_model = PeftModel.from_pretrained(gen_model, adapter_path)
            adapted_model.eval()
            response = generate(adapted_model, tokenizer, tc["prompt"], device)
            del adapted_model
        else:
            logger.warning("No adapter for %s, using base", top_domain)
            response = generate(gen_model, tokenizer, tc["prompt"], device)

        elapsed = time.time() - t0

        # Check if expected domain is in top-2
        top2_domains = [d for d, _ in top2]
        domain_hit = tc["expected_domain"] in top2_domains
        if not domain_hit:
            all_pass = False

        result = {
            "prompt": tc["prompt"],
            "description": tc["description"],
            "expected_domain": tc["expected_domain"],
            "top2_domains": top2,
            "domain_hit": domain_hit,
            "response": response[:300],
            "time_s": round(elapsed, 1),
        }
        results.append(result)

        logger.info("Expected: %s, Got top-2: %s, HIT: %s",
                     tc["expected_domain"], top2_domains, domain_hit)
        logger.info("Response: %s...", response[:200])
        logger.info("Time: %.1fs", elapsed)

    # Save results
    summary = {
        "test_cases": len(results),
        "all_pass": all_pass,
        "domain_hits": sum(r["domain_hit"] for r in results),
        "results": results,
    }
    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUTPUT).write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    logger.info("=" * 60)
    logger.info("E2E SMOKE: %d/%d domain hits, ALL PASS: %s",
                 summary["domain_hits"], summary["test_cases"], all_pass)
    logger.info("Results saved to %s", OUTPUT)

    del gen_model, encoder


if __name__ == "__main__":
    main()

"""E2E smoke test: router + 3 adapters on 3 test prompts."""
import os
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

import json
import logging
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_MODEL = "/home/kxkm/models/qwen3.5-4b/bf16/"
ROUTER_PATH = "/home/kxkm/micro-kiki/outputs/router/v0/router.pt"
ADAPTER_DIRS = {
    "chat-fr": "/home/kxkm/micro-kiki/outputs/stacks/stack-01-chat-fr",
    "python": "/home/kxkm/micro-kiki/outputs/stacks/stack-03-python",
    "reasoning": "/home/kxkm/micro-kiki/outputs/stacks/stack-02-reasoning",
}

TEST_PROMPTS = [
    {"prompt": "Ecris un haiku sur le printemps", "expected": "chat-fr"},
    {"prompt": "Combien font 17 x 23 ? Explique etape par etape", "expected": "reasoning"},
    {"prompt": "Write a Python function to check if a number is prime", "expected": "python"},
]


class DomainRouter(nn.Module):
    def __init__(self, input_dim, num_domains, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def get_embedding(prompt, tokenizer, model, device):
    enc = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    outputs = model(**enc, output_hidden_states=True)
    hidden = outputs.hidden_states[-1]
    mask = enc["attention_mask"].unsqueeze(-1).float()
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return pooled


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load router
    logger.info("Loading router from %s", ROUTER_PATH)
    ckpt = torch.load(ROUTER_PATH, map_location=device, weights_only=False)
    domain_names = ckpt["domain_names"]
    router = DomainRouter(ckpt["input_dim"], ckpt["num_domains"], ckpt["hidden_dim"])
    router.load_state_dict(ckpt["state_dict"])
    router.to(device).eval()
    logger.info("Router loaded. Domains: %s", domain_names)

    # Load base model (4-bit)
    logger.info("Loading base model (4-bit)...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="cuda", trust_remote_code=True
    )

    results = []
    all_pass = True

    for test in TEST_PROMPTS:
        prompt = test["prompt"]
        expected = test["expected"]
        logger.info("\n=== Test: %s ===", prompt)

        # Route
        emb = get_embedding(prompt, tokenizer, base_model, device)
        logits = router(emb)
        probs = torch.softmax(logits, dim=1)[0]
        ranked = sorted(zip(domain_names, probs.tolist()), key=lambda x: -x[1])
        top2 = [r[0] for r in ranked[:2]]
        logger.info("Router scores: %s", [(n, f"{p:.3f}") for n, p in ranked])
        logger.info("Top-2: %s, expected: %s", top2, expected)

        passed = expected in top2
        if not passed:
            all_pass = False
            logger.warning("FAIL: %s not in top-2 %s", expected, top2)

        # Generate with top adapter
        top_domain = top2[0]
        adapter_dir = ADAPTER_DIRS.get(top_domain)
        if adapter_dir and Path(adapter_dir).exists():
            logger.info("Loading adapter: %s", top_domain)
            model_with_adapter = PeftModel.from_pretrained(base_model, adapter_dir)
            model_with_adapter.eval()

            chat_input = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            inputs = tokenizer(chat_input, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model_with_adapter.generate(
                    **inputs, max_new_tokens=200, temperature=0.7, do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            logger.info("Response (%s): %s", top_domain, response[:300])

            # Unload adapter for next test
            del model_with_adapter
            torch.cuda.empty_cache()
        else:
            logger.warning("Adapter dir not found: %s", adapter_dir)
            response = "N/A"

        results.append({
            "prompt": prompt,
            "expected": expected,
            "top2": top2,
            "pass": passed,
            "response_preview": response[:200],
        })

    # Summary
    logger.info("\n=== E2E SMOKE TEST RESULTS ===")
    for r in results:
        status = "PASS" if r["pass"] else "FAIL"
        logger.info("[%s] %s -> top2=%s (expected=%s)", status, r["prompt"][:50], r["top2"], r["expected"])

    logger.info("Overall: %s", "ALL PASS" if all_pass else "SOME FAILED")

    # Save results
    out_path = Path("/home/kxkm/micro-kiki/outputs/e2e_smoke_test.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()

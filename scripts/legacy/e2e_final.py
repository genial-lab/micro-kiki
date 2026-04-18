#!/usr/bin/env python3
"""micro-kiki E2E: router -> adapter -> generate. Clean definitive version.

Loads saved router.pt, routes test prompts, activates the matching LoRA
adapter via PEFT, and generates a response for each.

Bugs fixed vs original smoke_e2e.py:
  1. Router load: extracts state_dict from checkpoint dict (key='state_dict')
  2. BFloat16: .float() before .cpu().numpy() / .tolist()
  3. String quoting: test["prompt"] not test[prompt]

Embedding method: AutoModelForCausalLM.model (Qwen3_5TextModel) mean-pool.
This produces identical architecture to AutoModel but with correct weight
loading (no key-prefix mismatch on Qwen3.5 VL checkpoints).

Usage:
    cd /home/kxkm/micro-kiki
    UNSLOTH_COMPILE_DISABLE=1 /home/kxkm/KIKI-models-tuning/.venv/bin/python \
        scripts/e2e_final.py 2>&1 | tee /tmp/e2e-final-clean.log
"""

import json
import os
import sys
import time

os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ── Config ──────────────────────────────────────────────────────────────────

BASE_MODEL = "/home/kxkm/models/qwen3.5-4b/bf16/"
ROUTER_PATH = "/home/kxkm/micro-kiki/outputs/router/v0/router.pt"

DOMAIN_TO_ADAPTER = {
    "chat-fr": "/home/kxkm/micro-kiki/outputs/stacks/stack-01-chat-fr-old",
    "reasoning": "/home/kxkm/micro-kiki/outputs/stacks/stack-02-reasoning-old",
    "python": "/home/kxkm/micro-kiki/outputs/stacks/stack-03-python-old",
}

TEST_PROMPTS = [
    {"prompt": "Bonjour ! Raconte-moi une blague drole en francais sur les chats.", "expected": "chat-fr"},
    {"prompt": "Solve: if 3x + 7 = 22, what is x? Show step-by-step reasoning.", "expected": "reasoning"},
    {"prompt": "def fibonacci(n):\n    # Complete this Python function", "expected": "python"},
    {"prompt": "Quels sont tes films preferes ? Donne-moi 3 recommandations.", "expected": "chat-fr"},
    {"prompt": "A train leaves at 60km/h. Another at 80km/h 30min later. When does it catch up?", "expected": "reasoning"},
    {"prompt": "import pandas as pd\n# Read CSV, filter age > 30, plot histogram", "expected": "python"},
]

MAX_NEW_TOKENS = 200


# ── Router MLP (must match train_router_kxkm.py exactly) ───────────────────

class RouterMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, num_domains: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_domains),
        )

    def forward(self, x):
        return self.net(x)


# ── Embedding via CausalLM inner model ──────────────────────────────────────

def embed_prompt(text: str, tokenizer, inner_model, device: str) -> torch.Tensor:
    """Mean-pool last_hidden_state from the text model. Returns float32 CPU."""
    inputs = tokenizer(
        [text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    ).to(device)

    with torch.no_grad():
        outputs = inner_model(**inputs)
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        hidden = outputs.last_hidden_state * attention_mask
        pooled = hidden.sum(dim=1) / attention_mask.sum(dim=1)

    # FIX #2: bf16 -> float32 before leaving GPU
    return pooled.squeeze(0).float().cpu()


# ── Load router from checkpoint ─────────────────────────────────────────────

def load_router(path: str, device: str) -> tuple:
    """Load RouterMLP from saved checkpoint, handling both formats.

    FIX #1: router.pt is a dict with 'state_dict' key, not a bare state dict
    or nn.Module. We must instantiate RouterMLP and load the state_dict.
    """
    ckpt = torch.load(path, weights_only=False, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        # Standard format from train_router_kxkm.py
        input_dim = ckpt["input_dim"]
        num_domains = ckpt["num_domains"]
        domain_names = ckpt["domain_names"]
        router = RouterMLP(input_dim=input_dim, num_domains=num_domains)
        router.load_state_dict(ckpt["state_dict"])
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        # Alternative format
        input_dim = ckpt.get("input_dim", 2560)
        num_domains = ckpt.get("num_domains", 3)
        domain_names = ckpt.get("domain_names", sorted(DOMAIN_TO_ADAPTER.keys()))
        router = RouterMLP(input_dim=input_dim, num_domains=num_domains)
        router.load_state_dict(ckpt["model_state_dict"])
    elif isinstance(ckpt, nn.Module):
        # Full model saved directly
        router = ckpt
        domain_names = sorted(DOMAIN_TO_ADAPTER.keys())
    elif isinstance(ckpt, dict):
        # Bare state_dict (no wrapper)
        # Infer dimensions from weight shapes
        first_weight = next(iter(ckpt.values()))
        input_dim = first_weight.shape[1] if first_weight.dim() == 2 else 2560
        last_key = list(ckpt.keys())[-1]
        last_weight = ckpt[last_key]
        num_domains = last_weight.shape[0] if "bias" in last_key else 3
        domain_names = sorted(DOMAIN_TO_ADAPTER.keys())
        router = RouterMLP(input_dim=input_dim, num_domains=num_domains)
        router.load_state_dict(ckpt)
    else:
        raise ValueError(f"Unknown router.pt format: {type(ckpt)}")

    router.eval()
    router.to(device)
    return router, domain_names


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[E2E] Device: {device}")
    print(f"[E2E] Base model: {BASE_MODEL}")
    print(f"[E2E] Router: {ROUTER_PATH}")
    print()

    # 1. Load tokenizer
    print("[E2E] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load CausalLM (for both embeddings via .model and generation via PEFT)
    print("[E2E] Loading AutoModelForCausalLM (bf16)...")
    causal_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    causal_model.eval()
    inner_model = causal_model.model  # Qwen3_5TextModel — proper weights
    print(f"  Model type: {type(causal_model).__name__}")
    print(f"  Inner type: {type(inner_model).__name__}")

    # 3. Load saved router (FIX #1: handle checkpoint dict format)
    print("[E2E] Loading router...")
    router, domain_names = load_router(ROUTER_PATH, device)
    print(f"  Domains: {domain_names}")

    # Sanity check: quick routing test
    with torch.no_grad():
        test_emb = embed_prompt("Bonjour", tokenizer, inner_model, device)
        test_logits = router(test_emb.unsqueeze(0).to(device))
        test_probs = torch.sigmoid(test_logits).squeeze(0).float().cpu().tolist()
        print(f"  Sanity check (\"Bonjour\"): {list(zip(domain_names, [round(p, 3) for p in test_probs]))}")
        max_prob = max(test_probs)
        if max_prob < 0.5:
            print(f"  WARNING: max prob {max_prob:.3f} < 0.5, router may not be calibrated for these embeddings")

    # 4. Load PEFT adapters
    print("\n[E2E] Loading PEFT adapters...")
    peft_model = None
    for domain, adapter_path in DOMAIN_TO_ADAPTER.items():
        print(f"  Loading adapter: {domain} from {adapter_path}")
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(
                causal_model,
                adapter_path,
                adapter_name=domain,
            )
        else:
            peft_model.load_adapter(adapter_path, adapter_name=domain)
    print("  All adapters loaded OK")

    # 5. Run E2E test prompts
    print()
    print("=" * 70)
    print("E2E SMOKE TEST — micro-kiki")
    print("=" * 70)

    hits = 0
    results = []

    for i, test in enumerate(TEST_PROMPTS):
        prompt = test["prompt"]       # FIX #3: proper dict key access
        expected = test["expected"]   # FIX #3: proper dict key access

        print(f"\n--- Test {i + 1}/{len(TEST_PROMPTS)} ---")
        print(f"  Prompt:   {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        print(f"  Expected: {expected}")

        # a. Compute embedding
        t0 = time.time()
        embedding = embed_prompt(prompt, tokenizer, inner_model, device)
        embed_time = time.time() - t0

        # b. Router prediction
        with torch.no_grad():
            logits = router(embedding.unsqueeze(0).to(device))
            # FIX #2: .float() before .tolist() to handle bf16
            probs = torch.sigmoid(logits).squeeze(0).float().cpu().tolist()

        domain_scores = sorted(zip(domain_names, probs), key=lambda x: x[1], reverse=True)
        predicted = domain_scores[0][0]
        hit = predicted == expected
        if hit:
            hits += 1

        print(f"  Router:   {', '.join(f'{d}={s:.4f}' for d, s in domain_scores)}")
        print(f"  Predicted: {predicted} ({'HIT' if hit else 'MISS'})")
        print(f"  Embed time: {embed_time:.3f}s")

        # c. Activate the predicted adapter
        peft_model.set_adapter(predicted)

        # d. Generate response
        t1 = time.time()
        chat_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer(chat_prompt, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            output_ids = peft_model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(
            output_ids[0][input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        gen_time = time.time() - t1

        print(f"  Gen time: {gen_time:.2f}s")
        print(f"  Response: {response[:120]}{'...' if len(response) > 120 else ''}")

        results.append({
            "prompt": prompt[:60],
            "expected": expected,
            "predicted": predicted,
            "hit": hit,
            "top2": [(d, round(s, 4)) for d, s in domain_scores[:2]],
            "response_snippet": response[:50],
            "embed_time_s": round(embed_time, 3),
            "gen_time_s": round(gen_time, 2),
        })

    # 6. Summary
    print()
    print("=" * 70)
    print(f"RESULTS: {hits}/{len(TEST_PROMPTS)} routing hits")
    passed = hits >= 4
    print(f"PASS: {'YES' if passed else 'NO'} (threshold: 4/{len(TEST_PROMPTS)})")
    print("=" * 70)

    for r in results:
        status = "HIT " if r["hit"] else "MISS"
        print(f"  [{status}] {r['expected']:>12} -> {r['predicted']:>12}  "
              f"top2={r['top2']}  resp={r['response_snippet']!r}")

    # Save results
    results_path = "/home/kxkm/micro-kiki/outputs/e2e_final_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "hits": hits,
            "total": len(TEST_PROMPTS),
            "pass": passed,
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {results_path}")

    # Cleanup
    del peft_model, causal_model, router
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

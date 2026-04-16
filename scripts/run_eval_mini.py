#!/usr/bin/env python3
"""Minimal eval: 5 prompts, 64 max tokens, log to file."""
import json, logging, os, sys, time
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
from pathlib import Path

LOG_FILE = "/home/kxkm/micro-kiki/results/eval-mini.log"
Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

import httpx
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "/home/kxkm/models/qwen3.5-4b/bf16/"
ADAPTER = "/home/kxkm/micro-kiki/outputs/stacks/stack-01-chat-fr"
TEACHER_URL = "http://localhost:8000/v1/chat/completions"
TEACHER_MODEL = "Qwen3.5-35B-A3B-UD-Q3_K_XL.gguf"
OUTPUT = "/home/kxkm/micro-kiki/results/stack-01-eval.json"
MAX_NEW_TOKENS = 64
MAX_EVAL = 5

EVAL_PROMPTS = [
    "Qu'est-ce que la mecanique quantique et pourquoi est-elle si contre-intuitive ?",
    "Comment installer un systeme Linux sur un vieux PC ?",
    "Quels sont les bienfaits de la meditation pleine conscience ?",
    "Comment fonctionne un GPS et quelle est sa precision ?",
    "Ecris un petit sketch humoristique sur la vie en open space",
]


def generate(model, tokenizer, prompt, device="cpu"):
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # greedy for speed
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def judge(prompt, resp_base, resp_stack):
    judge_prompt = f"""Compare two responses. Which is better? Return ONLY JSON.

Prompt: {prompt}

Response A (base): {resp_base[:200]}

Response B (tuned): {resp_stack[:200]}

Return: {{"winner": "A" or "B", "score_B": 0.0-1.0, "reason": "brief"}}"""

    payload = {
        "model": TEACHER_MODEL,
        "messages": [{"role": "user", "content": judge_prompt}],
        "temperature": 0.1,
        "max_tokens": 150,
    }
    try:
        resp = httpx.post(TEACHER_URL, json=payload, timeout=120.0)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        if "<think>" in content:
            content = content.split("</think>")[-1].strip()
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(content[start:end])
    except Exception as e:
        logger.warning("Judge error: %s", e)
    return {"winner": "A", "score_B": 0.5, "reason": "parse_error"}


def main():
    device = "cpu"
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading base model...")
    t0 = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32, device_map=device, trust_remote_code=True
    )
    base_model.eval()
    logger.info("Base model loaded in %.0fs", time.time() - t0)

    logger.info("Loading adapter model...")
    t0 = time.time()
    adapter_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32, device_map=device, trust_remote_code=True
    )
    adapter_model = PeftModel.from_pretrained(adapter_model, ADAPTER)
    adapter_model.eval()
    logger.info("Adapter model loaded in %.0fs", time.time() - t0)

    wins, total = 0, 0
    results = []
    for i, prompt in enumerate(EVAL_PROMPTS[:MAX_EVAL]):
        logger.info("[%d/%d] %s...", i + 1, MAX_EVAL, prompt[:60])
        t0 = time.time()

        resp_base = generate(base_model, tokenizer, prompt, device)
        t_base = time.time() - t0
        logger.info("  base: %.1fs, %d chars", t_base, len(resp_base))

        resp_stack = generate(adapter_model, tokenizer, prompt, device)
        t_stack = time.time() - t0 - t_base
        logger.info("  stack: %.1fs, %d chars", t_stack, len(resp_stack))

        verdict = judge(prompt, resp_base, resp_stack)
        is_win = verdict.get("winner") == "B"
        if is_win:
            wins += 1
        total += 1
        elapsed = time.time() - t0
        results.append({
            "prompt": prompt[:100],
            "winner": "stack" if is_win else "base",
            "score_B": verdict.get("score_B", 0.5),
            "reason": verdict.get("reason", ""),
            "time_s": round(elapsed, 1),
        })
        logger.info("  -> winner=%s score=%.2f (%.1fs)",
                     "stack" if is_win else "base", verdict.get("score_B", 0.5), elapsed)

    win_rate = wins / total if total else 0
    summary = {
        "stack": "stack-01-chat-fr",
        "n_prompts": total,
        "wins": wins,
        "win_rate_vs_base": round(win_rate, 4),
        "avg_score": round(sum(r["score_B"] for r in results) / len(results), 4) if results else 0,
        "results": results,
    }
    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUTPUT).write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    logger.info("=" * 60)
    logger.info("EVAL RESULT: win_rate=%.4f (%d/%d)", win_rate, wins, total)
    logger.info("PASS: %s (threshold: 0.55)", "YES" if win_rate >= 0.55 else "NO")
    logger.info("Results saved to %s", OUTPUT)

    del base_model, adapter_model


if __name__ == "__main__":
    main()

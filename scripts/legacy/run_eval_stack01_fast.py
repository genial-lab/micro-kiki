#!/usr/bin/env python3
"""Fast eval stack-01 chat-fr: pairwise win-rate vs base (20 prompts).

Runs on CPU (GPU occupied by teacher). Uses teacher on :8000 as judge.
"""
import json, logging, os, sys, time
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

import httpx
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "/home/kxkm/models/qwen3.5-4b/bf16/"
ADAPTER = "/home/kxkm/micro-kiki/outputs/stacks/stack-01-chat-fr"
EVAL_DATA = "/home/kxkm/micro-kiki/data/eval/chat-fr.jsonl"
TEACHER_URL = "http://localhost:8000/v1/chat/completions"
TEACHER_MODEL = "Qwen3.5-35B-A3B-UD-Q3_K_XL.gguf"
OUTPUT = "/home/kxkm/micro-kiki/results/stack-01-eval.json"
MAX_NEW_TOKENS = 128
MAX_EVAL = 20


def generate(model, tokenizer, prompt, device="cpu"):
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def judge(prompt, resp_base, resp_stack):
    judge_prompt = f"""Compare these two responses to the same prompt. Which is better?

## Prompt
{prompt}

## Response A (base model)
{resp_base}

## Response B (fine-tuned)
{resp_stack}

Return ONLY a JSON object: {{"winner": "A" or "B", "score_B": 0.0 to 1.0, "reason": "brief explanation"}}
Do not include any other text."""
    payload = {
        "model": TEACHER_MODEL,
        "messages": [{"role": "user", "content": judge_prompt}],
        "temperature": 0.1,
        "max_tokens": 200,
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
    prompts = []
    for line in Path(EVAL_DATA).read_text().strip().split("\n"):
        if line.strip():
            prompts.append(json.loads(line)["prompt"])
    prompts = prompts[:MAX_EVAL]
    logger.info("Eval prompts: %d", len(prompts))

    device = "cpu"
    logger.info("Loading base model on %s...", device)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    base_model.eval()

    logger.info("Loading adapter model...")
    adapter_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    adapter_model = PeftModel.from_pretrained(adapter_model, ADAPTER)
    adapter_model.eval()

    wins, total = 0, 0
    results = []
    for i, prompt in enumerate(prompts):
        logger.info("[%d/%d] %s...", i + 1, len(prompts), prompt[:60])
        t0 = time.time()
        resp_base = generate(base_model, tokenizer, prompt, device)
        resp_stack = generate(adapter_model, tokenizer, prompt, device)
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
        logger.info(
            "  -> winner=%s score=%.2f (%.1fs)",
            "stack" if is_win else "base",
            verdict.get("score_B", 0.5),
            elapsed,
        )

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

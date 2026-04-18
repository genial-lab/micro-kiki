#!/usr/bin/env python3
"""Quick 3-prompt eval with thinking disabled."""
import json, logging, os, time
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
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
MAX_NEW_TOKENS = 48

PROMPTS = [
    "Explique brievement ce qu'est l'intelligence artificielle",
    "Quels sont les bienfaits de la lecture quotidienne ?",
    "Comment faire un bon cafe filtre ?",
]


def generate(model, tokenizer, prompt, device="cpu"):
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def judge(prompt, resp_base, resp_stack):
    payload = {
        "model": TEACHER_MODEL,
        "messages": [{"role": "user", "content": f"""Compare two responses. Return ONLY JSON.
Prompt: {prompt}
Response A (base): {resp_base[:200]}
Response B (tuned): {resp_stack[:200]}
JSON: {{"winner": "A" or "B", "score_B": 0.0-1.0, "reason": "brief"}}"""}],
        "temperature": 0.1, "max_tokens": 150,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        resp = httpx.post(TEACHER_URL, json=payload, timeout=120.0)
        content = resp.json()["choices"][0]["message"]["content"]
        logger.info("  judge: %s", content[:150])
        s = content.find("{"); e = content.rfind("}") + 1
        if s >= 0 and e > s:
            return json.loads(content[s:e])
    except Exception as ex:
        logger.warning("Judge error: %s", ex)
    return {"winner": "A", "score_B": 0.5, "reason": "parse_error"}


def main():
    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)
    base.eval()
    logger.info("Loading adapter model...")
    adapted = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)
    adapted = PeftModel.from_pretrained(adapted, ADAPTER)
    adapted.eval()

    wins, results = 0, []
    for i, prompt in enumerate(PROMPTS):
        logger.info("[%d/%d] %s", i+1, len(PROMPTS), prompt)
        t0 = time.time()
        rb = generate(base, tokenizer, prompt, device)
        logger.info("  base: %s...", rb[:80])
        rs = generate(adapted, tokenizer, prompt, device)
        logger.info("  stack: %s...", rs[:80])
        v = judge(prompt, rb, rs)
        win = v.get("winner") == "B"
        if win: wins += 1
        results.append({"prompt": prompt, "winner": "stack" if win else "base",
                        "score_B": v.get("score_B", 0.5), "reason": v.get("reason",""),
                        "time_s": round(time.time()-t0, 1)})
        logger.info("  -> %s score=%.2f", "STACK WIN" if win else "base win", v.get("score_B", 0.5))

    wr = wins / len(PROMPTS)
    summary = {"stack": "stack-01-chat-fr", "n_prompts": len(PROMPTS), "wins": wins,
               "win_rate_vs_base": round(wr, 4),
               "avg_score": round(sum(r["score_B"] for r in results)/len(results), 4),
               "results": results}
    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUTPUT).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info("EVAL: win_rate=%.4f (%d/%d) PASS=%s", wr, wins, len(PROMPTS), wr >= 0.55)
    del base, adapted

if __name__ == "__main__":
    main()

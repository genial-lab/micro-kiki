#!/usr/bin/env python3
"""C2 downstream bench: VQC vs random vs oracle router on eval queries.

Loads the C3 real corpus (or C1 synthetic), selects N queries per domain,
calls three routers, dispatches to a generation LLM with a domain-expert
system prompt (simulated adapter), scores with a judge LLM using the rubric
from src/routing/llm_judge.py.

DEFAULT CONFIG uses kxkm-ai Qwen3.5-35B for BOTH generation and judging
(Studio 480B busy during this session). This introduces a self-judging
bias — documented in the results. The ordering of the three routers
remains meaningful since the bias applies uniformly.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import requests
import torch

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.routing.downstream_harness import run_downstream_eval
from src.routing.llm_judge import build_rubric_prompt, parse_score
from src.routing.text_jepa.dataset import load_domain_corpus
from src.routing.torch_vqc_router import TorchVQCRouter

logger = logging.getLogger(__name__)


def _llm_call(base_url: str, model: str, prompt: str, max_tokens: int = 512,
              temperature: float = 0.0) -> str:
    # Qwen3/3.5 thinking mode consumes max_tokens as hidden reasoning and
    # returns empty visible content — disable via chat_template_kwargs.
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data/corpus-real"))
    p.add_argument("--domains", default="dsp,electronics,emc,embedded,freecad,kicad-dsl,platformio,power,spice,stm32")
    p.add_argument("--per-domain", type=int, default=5)
    p.add_argument("--gen-url", default="http://kxkm-ai:8000")
    p.add_argument("--gen-model", default="Qwen3.5-35B-A3B-UD-Q3_K_XL.gguf")
    p.add_argument("--judge-url", default="http://kxkm-ai:8000",
                   help="Default: same as gen (Studio busy). Self-judging bias documented.")
    p.add_argument("--judge-model", default="Qwen3.5-35B-A3B-UD-Q3_K_XL.gguf")
    p.add_argument("--backbone", default="models/niche-embeddings")
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--vqc-epochs", type=int, default=300)
    p.add_argument("--vqc-lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("results/c2-downstream.json"))
    p.add_argument("--dry-run", action="store_true", help="Stub LLM + judge calls")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    dom_to_idx = {d: i for i, d in enumerate(domains)}
    n_classes = len(domains)

    # Load corpus, pick first per_domain samples per domain as held-out eval
    samples = load_domain_corpus(args.data_dir, domains=domains, max_per_domain=50)
    if not samples:
        logger.error("no samples loaded from %s", args.data_dir)
        return 2

    per_dom_count: dict[str, int] = {d: 0 for d in domains}
    eval_queries: list[dict] = []
    remaining: list = []  # used for VQC training (not in eval set)
    for s in samples:
        if per_dom_count[s.domain] < args.per_domain:
            eval_queries.append({
                "question": s.text,
                "domain": s.domain,
                "domain_idx": dom_to_idx[s.domain],
            })
            per_dom_count[s.domain] += 1
        else:
            remaining.append(s)

    logger.info("eval: %d queries (%d/domain) | training pool: %d samples",
                len(eval_queries), args.per_domain, len(remaining))

    # Embed eval queries
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(args.backbone, device="cpu")
    tok = st.tokenizer
    tr_model = st[0].auto_model.to("cpu")

    def embed(text: str) -> np.ndarray:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=args.seq_len, padding="max_length")
        with torch.no_grad():
            out = tr_model(**enc).last_hidden_state
        return out.squeeze(0).mean(dim=0).cpu().numpy()

    eval_embs = np.stack([embed(q["question"]) for q in eval_queries]).astype(np.float64)
    logger.info("embedded %d eval queries", len(eval_embs))

    # Train VQC on the REMAINING samples (not in eval set)
    logger.info("training VQC on %d remaining samples", len(remaining))
    train_embs = np.stack([embed(s.text) for s in remaining]).astype(np.float64)
    train_labels = np.array([dom_to_idx[s.domain] for s in remaining], dtype=np.int64)
    vqc = TorchVQCRouter(
        n_qubits=4, n_layers=6, n_classes=n_classes,
        lr=args.vqc_lr, seed=args.seed, input_dim=eval_embs.shape[1], weight_decay=1e-4,
    )
    Xt = torch.from_numpy(train_embs).double()
    yt = torch.from_numpy(train_labels)
    vqc.train_batched(Xt, yt, epochs=args.vqc_epochs)
    with torch.no_grad():
        vqc_preds_train = vqc.predict(Xt).numpy()
    train_acc = float((vqc_preds_train == train_labels).mean())
    logger.info("VQC trained, train_acc=%.3f", train_acc)

    # Router functions
    def router_vqc(emb: np.ndarray) -> int:
        with torch.no_grad():
            return int(vqc.predict(torch.from_numpy(emb).double().unsqueeze(0))[0])

    rng = np.random.default_rng(args.seed)

    def router_random(emb: np.ndarray) -> int:
        return int(rng.integers(0, n_classes))

    oracle_counter = [0]

    def router_oracle(emb: np.ndarray) -> int:
        idx = eval_queries[oracle_counter[0]]["domain_idx"]
        oracle_counter[0] += 1
        return idx

    # LLM and judge callbacks
    def llm_call(question: str, domain_name: str) -> str:
        if args.dry_run:
            return f"[stub answer for {domain_name}]"
        prompt = f"You are an expert in {domain_name}. Answer concisely.\n\nQuestion: {question}"
        try:
            return _llm_call(args.gen_url, args.gen_model, prompt, max_tokens=512)
        except Exception as e:
            logger.warning("gen LLM call failed: %s", e)
            return "[error]"

    def judge_call(question: str, answer: str, expected_domain: str) -> int:
        if args.dry_run:
            return 3
        prompt = build_rubric_prompt(question=question, answer=answer, domain=expected_domain)
        try:
            resp = _llm_call(args.judge_url, args.judge_model, prompt, max_tokens=256)
        except Exception as e:
            logger.warning("judge LLM call failed: %s", e)
            return 0
        score = parse_score(resp)
        return score if score is not None else 0

    # Run 3 routers
    results = {}
    for name, fn in [("vqc", router_vqc), ("random", router_random), ("oracle", router_oracle)]:
        logger.info("running router=%s", name)
        if name == "oracle":
            oracle_counter[0] = 0  # reset
        r = run_downstream_eval(
            queries=eval_queries,
            embeddings=eval_embs,
            router_fn=fn,
            llm_fn=llm_call,
            judge_fn=judge_call,
            domain_names=domains,
        )
        results[name] = r
        logger.info("  router=%s  mean_score=%.3f  routing_acc=%.3f",
                    name, r["mean_score"], r["routing_accuracy"])

    meta = {
        "config": {
            "gen_model": args.gen_model,
            "gen_url": args.gen_url,
            "judge_model": args.judge_model,
            "judge_url": args.judge_url,
            "self_judging": (args.gen_url == args.judge_url and args.gen_model == args.judge_model),
            "data_dir": str(args.data_dir),
            "per_domain": args.per_domain,
            "n_eval": len(eval_queries),
            "vqc_train_acc": train_acc,
            "seed": args.seed,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(meta, indent=2))
    logger.info("wrote %s", args.output)

    print("\n=== C2 Downstream Results ===")
    print(f"gen={args.gen_model} | judge={args.judge_model} | self={meta['config']['self_judging']}")
    print(f"{'router':<8} {'mean':>6} {'routing_acc':>12} {'score | correct':>16} {'score | wrong':>14}")
    for name, r in results.items():
        print(f"{name:<8} {r['mean_score']:>6.2f} {r['routing_accuracy']:>12.3f} "
              f"{r['mean_score_when_routed_correct']:>16.2f} {r['mean_score_when_routed_wrong']:>14.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

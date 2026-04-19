#!/usr/bin/env python3
"""Generate `ml-training` domain training dataset via teacher distillation.

Scope: ML training infrastructure — optimizers (AdamW, Lion, Adafactor,
Sophia), learning-rate schedulers (cosine, linear, WSD, rsqrt), distributed
training (DDP, FSDP, ZeRO, tensor/pipeline parallelism), gradient
checkpointing, mixed precision (BF16/FP8/AMP), LoRA/QLoRA/OPLoRA mechanics,
gradient accumulation, dataloader patterns, eval loops, checkpointing,
reproducibility, and debugging loss spikes.

This is a SCAFFOLD. Q&A pairs come from the local Qwen3-Coder-480B teacher;
if it is unreachable the script writes a placeholder README and exits 0.

Usage::

    uv run python scripts/gen_ml_training_dataset.py \\
        --output data/micro-kiki/ml-training/raw.jsonl \\
        --limit 10 \\
        --teacher-url http://kxkm-ai:8000

TODO(operator): expand TOPIC_SEEDS to ~50 entries and raise --limit to 2000
once the teacher server is confirmed reachable.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

DOMAIN = "ml-training"
DEFAULT_OUTPUT = f"data/micro-kiki/{DOMAIN}/raw.jsonl"
DEFAULT_TEACHER_URL = "http://localhost:8000"
DEFAULT_TEACHER_MODEL = "Qwen3-Coder-480B-A35B-Instruct"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Topic seed — expand to ~50 before the production run
# ---------------------------------------------------------------------------
TOPIC_SEEDS: list[str] = [
    "Explain when AdamW beats plain Adam and how weight decay is decoupled.",
    "Compare cosine vs linear vs WSD (warmup-stable-decay) LR schedules.",
    "How do I pick a peak LR for LoRA fine-tuning on a 35B base?",
    "What is gradient accumulation and when does it hurt (batchnorm)?",
    "Walk through FSDP vs DDP vs ZeRO-3 trade-offs on a 4x4090 node.",
    "How does gradient checkpointing trade compute for memory in practice?",
    "Explain BF16 vs FP16 AMP: loss scaling and numerical-range differences.",
    "What is the 'warmup ratio' in transformers Trainer and how do I pick it?",
    "How do I diagnose and recover from a mid-training loss spike?",
    "Describe QLoRA quantization math (NF4 + double-quant) end to end.",
    "Compare LoRA vs DoRA vs OPLoRA initialization strategies.",
    "When should I use Lion or Sophia instead of AdamW for pretraining?",
    "What is the effective batch size with grad accumulation on DDP?",
    "How do I pick max_seq_length when the data has a long tail?",
    "Explain tensor parallelism vs pipeline parallelism for LLM training.",
    "What is the purpose of `accelerate launch` vs `torchrun`?",
    "Give a checklist for reproducibility: seeds, cudnn, dataloader workers.",
    "How do I compute peak VRAM for BF16 LoRA at rank 32 on a 35B model?",
    "Explain gradient clipping by global norm and when it backfires.",
    "How does sequence packing reduce padding waste during SFT?",
    # TODO(operator): extend to ~50 seeds covering FP8, Flash-Attn,
    # activation offloading, DPO/GRPO training, curriculum ordering,
    # hyperparameter transfer (muP), dataloader bottlenecks, TensorBoard
    # vs wandb, and MoE-specific routing-loss balancing.
]


PROMPT_TEMPLATE = """You are an expert ML training engineer writing training
data for an AI assistant that helps practitioners tune and debug large-model
fine-tuning runs.

Answer the question below with:
- Correct numerical reasoning (memory math, FLOPs, step counts)
- Concrete HuggingFace / PyTorch / MLX snippets where relevant
- Real hyperparameter defaults used in the field
- Stay under ~350 words

Question: {prompt}

Answer:"""


# ---------------------------------------------------------------------------
# Teacher probe + minimal client
# ---------------------------------------------------------------------------
def _probe_teacher(url: str, timeout: float = 3.0) -> bool:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return False
    for path in ("/v1/models", "/health", "/"):
        try:
            req = Request(url.rstrip("/") + path, method="GET")
            with urlopen(req, timeout=timeout) as resp:  # noqa: S310
                if resp.status < 500:
                    return True
        except Exception:  # noqa: BLE001
            continue
    return False


def _teacher_complete(url: str, model: str, prompt: str, timeout: float = 60.0) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.3,
    }).encode("utf-8")
    req = Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"].strip()


def _write_placeholder(output: Path, teacher_url: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    readme = output.parent / "README.md"
    readme.write_text(
        f"# {DOMAIN} — dataset not yet generated\n\n"
        f"Teacher at `{teacher_url}` was unreachable when `gen_ml_training_dataset.py` "
        f"was invoked. Re-run with `--teacher-url <url>` pointing at the local "
        f"Qwen3-Coder-480B MLX server (typically `http://mac-studio:8000`) "
        f"once it is up:\n\n"
        f"```bash\n"
        f"uv run python scripts/gen_ml_training_dataset.py \\\n"
        f"    --output {output} \\\n"
        f"    --limit 2000 \\\n"
        f"    --teacher-url http://mac-studio:8000\n"
        f"```\n",
        encoding="utf-8",
    )
    if not output.exists():
        output.write_text("", encoding="utf-8")
    logger.warning("Teacher unreachable — wrote placeholder %s", readme)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"Generate {DOMAIN} training data via teacher distillation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path)
    parser.add_argument("--limit", type=int, default=10,
                        help="Number of seed prompts to distill (scaffold default is small).")
    parser.add_argument("--teacher-url", default=DEFAULT_TEACHER_URL)
    parser.add_argument("--teacher-model", default=DEFAULT_TEACHER_MODEL)
    args = parser.parse_args()

    output: Path = args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    if not _probe_teacher(args.teacher_url):
        _write_placeholder(output, args.teacher_url)
        return 0

    seeds = TOPIC_SEEDS[: args.limit]
    logger.info("Distilling %d %s prompts via teacher %s", len(seeds), DOMAIN, args.teacher_url)

    written = 0
    with output.open("w", encoding="utf-8") as fh:
        for idx, seed in enumerate(seeds, start=1):
            try:
                answer = _teacher_complete(
                    args.teacher_url, args.teacher_model,
                    PROMPT_TEMPLATE.format(prompt=seed),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("[%d/%d] teacher call failed: %s", idx, len(seeds), exc)
                continue
            record = {"messages": [
                {"role": "user", "content": seed},
                {"role": "assistant", "content": answer},
            ], "domain": DOMAIN}
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            logger.info("[%d/%d] OK (%d chars)", idx, len(seeds), len(answer))

    logger.info("Wrote %d examples to %s", written, output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Per-stack evaluation harness with LLM judge."""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """Compare these two responses. Which is better?

## Prompt
{prompt}

## Response A (base)
{response_base}

## Response B (stack)
{response_stack}

Return JSON: {{"winner": "base" or "stack", "score": 0.0 to 1.0, "reason": "brief"}}"""


class StackEvaluator:
    def __init__(self, judge_client, judge_model: str = "mistral-large") -> None:
        self._judge = judge_client
        self._judge_model = judge_model

    async def evaluate(self, eval_path: Path | str, generate_fn, stack_name: str) -> dict:
        prompts = []
        for line in Path(eval_path).read_text().strip().split("\n"):
            if line:
                prompts.append(json.loads(line))

        wins, total_score, samples = 0, 0.0, []
        for entry in prompts:
            prompt = entry["prompt"]
            resp_base = await generate_fn(prompt, adapter=None)
            resp_stack = await generate_fn(prompt, adapter=stack_name)

            judge_raw = await self._judge.generate(
                prompt=JUDGE_PROMPT.format(prompt=prompt, response_base=resp_base, response_stack=resp_stack),
                model=self._judge_model,
            )
            try:
                jr = json.loads(judge_raw)
            except json.JSONDecodeError:
                jr = {"winner": "base", "score": 0.5, "reason": "parse error"}

            if jr.get("winner") == "stack":
                wins += 1
            total_score += jr.get("score", 0.5)
            if len(samples) < 5:
                samples.append({"prompt": prompt[:100], "winner": jr.get("winner")})

        n = len(prompts)
        return {
            "stack": stack_name, "n_prompts": n,
            "win_rate_vs_base": wins / n if n else 0,
            "avg_judge_score": total_score / n if n else 0,
            "sample_responses": samples,
        }

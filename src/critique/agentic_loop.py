from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Callable, Awaitable

from src.critique.templates import AGENTIC_PLAN, AGENTIC_EVALUATE

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StepResult:
    step_num: int
    action: str
    output: str
    met_expectations: bool


@dataclass(frozen=True)
class LoopResult:
    completed: bool
    iterations: int
    steps: list[StepResult]
    final_output: str


class AgenticLoop:
    """Level 3 auto-critique: plan-execute-evaluate loop."""

    def __init__(
        self,
        generate_fn: Callable[[str], Awaitable[str]],
        max_iterations: int = 5,
    ) -> None:
        self._generate = generate_fn
        self._max_iterations = max_iterations

    async def _plan(self, task: str, tools: list[str]) -> list[dict]:
        prompt = AGENTIC_PLAN.format(task=task, tools=", ".join(tools) or "none")
        raw = await self._generate(prompt)
        return json.loads(raw)

    async def _execute_step(self, step: dict) -> str:
        prompt = f"Execute this step: {step['action']}"
        return await self._generate(prompt)

    async def _evaluate_step(self, step: dict, output: str) -> dict:
        prompt = AGENTIC_EVALUATE.format(
            step_description=step["action"],
            expected=step.get("expected_output", ""),
            actual=output,
        )
        raw = await self._generate(prompt)
        return json.loads(raw)

    async def run(self, task: str, tools: list[str]) -> LoopResult:
        all_steps: list[StepResult] = []
        last_output = ""

        for iteration in range(1, self._max_iterations + 1):
            plan = await self._plan(task, tools)
            completed_all = True

            for step in plan:
                output = await self._execute_step(step)
                evaluation = await self._evaluate_step(step, output)

                step_result = StepResult(
                    step_num=step.get("step", 0),
                    action=step["action"],
                    output=output,
                    met_expectations=evaluation.get("meets_expectations", False),
                )
                all_steps.append(step_result)
                last_output = output

                next_action = evaluation.get("next_action", "proceed")
                if next_action == "abort":
                    return LoopResult(
                        completed=False,
                        iterations=iteration,
                        steps=all_steps,
                        final_output=last_output,
                    )
                if next_action == "retry":
                    completed_all = False
                    break

            if completed_all:
                return LoopResult(
                    completed=True,
                    iterations=iteration,
                    steps=all_steps,
                    final_output=last_output,
                )

            logger.info("Agentic loop iteration %d/%d — retrying", iteration, self._max_iterations)

        return LoopResult(
            completed=False,
            iterations=self._max_iterations,
            steps=all_steps,
            final_output=last_output,
        )

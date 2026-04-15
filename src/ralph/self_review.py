from __future__ import annotations

import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

REVIEW_TEMPLATE = """\
Review this code for the following categories. Return JSON:
{{
  "bugs": ["list of bugs found"],
  "edge_cases": ["unhandled edge cases"],
  "perf": ["performance issues"],
  "security": ["security concerns"],
  "style": ["style/convention violations"],
  "approved": true/false,
  "summary": "one-line assessment"
}}

## Code
{code}

## Context
{context}"""


@dataclass(frozen=True)
class ReviewResult:
    bugs: list[str]
    edge_cases: list[str]
    perf: list[str]
    security: list[str]
    style: list[str]
    approved: bool
    summary: str

    @property
    def total_issues(self) -> int:
        return len(self.bugs) + len(self.edge_cases) + len(self.perf) + len(self.security) + len(self.style)


class CodeReview:
    """Structured self-review for ralph-generated code."""

    def __init__(self, max_passes: int = 3) -> None:
        self.max_passes = max_passes

    @staticmethod
    def get_template() -> str:
        return REVIEW_TEMPLATE

    @staticmethod
    def parse_review(raw: str) -> ReviewResult:
        data = json.loads(raw)
        return ReviewResult(
            bugs=data.get("bugs", []),
            edge_cases=data.get("edge_cases", []),
            perf=data.get("perf", []),
            security=data.get("security", []),
            style=data.get("style", []),
            approved=data.get("approved", False),
            summary=data.get("summary", ""),
        )

    def format_prompt(self, code: str, context: str = "") -> str:
        return REVIEW_TEMPLATE.format(code=code, context=context)

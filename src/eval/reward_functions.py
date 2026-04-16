"""Domain-specific reward functions for GRPO training.

Each function: (prompt: str, response: str, domain: str) -> float (0.0 to 1.0)
"""
from __future__ import annotations

import logging
import re

import httpx

logger = logging.getLogger(__name__)

JUDGE_URL = "http://localhost:8481/v1/chat/completions"
JUDGE_MODEL = "Qwen3-Coder-480B-A35B"

# ---------------------------------------------------------------------------
# Syntax validation
# ---------------------------------------------------------------------------

# Patterns for structured domain detection
_SPICE_MARKERS = re.compile(
    r"(?i)(\.(model|subckt|ends|tran|ac|dc|op|lib|inc)\b"
    r"|^[rcldqmjkt]\d+\b"  # component identifiers
    r"|^v\d+\b|^i\d+\b)",   # voltage/current sources
    re.MULTILINE,
)
_SPICE_FULL = re.compile(
    r"(?i)(\.(model|subckt)\b.*|\.(tran|ac|dc)\b.*|^[rcl]\d+\s+\S+\s+\S+\s+[\d.eE+-]+)",
    re.MULTILINE,
)
_KICAD_FULL = re.compile(r"\(\s*(?:module|footprint|kicad_pcb|kicad_sch)\b")
_KICAD_PARTIAL = re.compile(r"\(\s*\w+[^)]*\)")


def syntax_valid(prompt: str, response: str, domain: str) -> float:
    """Check if output contains parseable syntax for structured domains.

    Returns:
        1.0 — valid syntax found
        0.5 — partial/plausible markers found
        0.0 — no recognisable structure
    """
    if domain == "kicad-dsl":
        if _KICAD_FULL.search(response):
            return 1.0
        if _KICAD_PARTIAL.search(response):
            return 0.5
        return 0.0

    if domain == "spice":
        full_matches = _SPICE_FULL.findall(response)
        if len(full_matches) >= 2:
            return 1.0
        if _SPICE_MARKERS.search(response):
            return 0.5
        return 0.0

    # For other domains, neutral score
    return 0.5


# ---------------------------------------------------------------------------
# Format correctness
# ---------------------------------------------------------------------------

_CODE_BLOCK = re.compile(r"```[\w]*\n[\s\S]*?```")
_STEP_NUMBERING = re.compile(r"^\s*(?:\d+[.)]\s+|\*\s+|-\s+)", re.MULTILINE)
_COMPONENT_VALUE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:k?Ω|[mMkKGT]?[VAWFH]|[nNpPuU][FHSΩ]|[mMkKGTnNpPuU]?Hz)\b"
)

_DESIGN_KEYWORDS = re.compile(
    r"\b(design|calculate|select|choose|determine|schematic|circuit|layout)\b",
    re.IGNORECASE,
)
_CODE_KEYWORDS = re.compile(
    r"\b(code|implement|write|function|script|program|class|def|module)\b",
    re.IGNORECASE,
)


def format_correct(prompt: str, response: str, domain: str) -> float:
    """Check expected output structure for the domain and question type.

    Returns a score in [0.0, 1.0].
    """
    score = 0.0
    checks = 0

    # Code block check — for code-centric domains or code questions
    code_domains = {"embedded", "stm32", "platformio", "kicad-dsl", "spice"}
    if domain in code_domains or _CODE_KEYWORDS.search(prompt):
        checks += 1
        if _CODE_BLOCK.search(response):
            score += 1.0

    # Step-by-step check — for design questions
    if _DESIGN_KEYWORDS.search(prompt):
        checks += 1
        step_lines = _STEP_NUMBERING.findall(response)
        if len(step_lines) >= 3:
            score += 1.0
        elif len(step_lines) >= 1:
            score += 0.5

    # Component values with units — for electronics domains
    electronics_domains = {"spice", "emc", "power", "electronics", "dsp"}
    if domain in electronics_domains:
        checks += 1
        unit_matches = _COMPONENT_VALUE.findall(response)
        if len(unit_matches) >= 3:
            score += 1.0
        elif len(unit_matches) >= 1:
            score += 0.5

    if checks == 0:
        # No specific checks applied — neutral
        return 0.5

    return score / checks


# ---------------------------------------------------------------------------
# Completeness reward
# ---------------------------------------------------------------------------

_MIN_GOOD = 200
_MAX_GOOD = 2000
_TOO_SHORT = 50
_TOO_LONG = 5000


def completeness_reward(prompt: str, response: str, domain: str) -> float:
    """Score based on response length and detail.

    Returns:
        0.0 — too short (<50 chars)
        1.0 — good length (200–2000 chars)
        linear ramp between boundaries
        0.7 — verbose penalty (>5000 chars)
    """
    n = len(response.strip())

    if n < _TOO_SHORT:
        return 0.0
    if n < _MIN_GOOD:
        # Linear ramp 0 → 1
        return (n - _TOO_SHORT) / (_MIN_GOOD - _TOO_SHORT)
    if n <= _MAX_GOOD:
        return 1.0
    if n <= _TOO_LONG:
        # Linear ramp 1 → 0.7
        excess = (n - _MAX_GOOD) / (_TOO_LONG - _MAX_GOOD)
        return 1.0 - 0.3 * excess
    return 0.7


# ---------------------------------------------------------------------------
# Accuracy reward (judge call)
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = (
    "You are a precise domain expert evaluator. "
    "Rate the quality of the response strictly on correctness, clarity, and completeness. "
    "Reply ONLY with a JSON object: {\"score\": <float 0.0-1.0>}"
)


def accuracy_reward(prompt: str, response: str, domain: str) -> float:
    """Call 480B judge via HTTP to score accuracy.

    Returns normalised score 0.0–1.0.
    Falls back to 0.5 if judge is unavailable.
    """
    scoring_prompt = (
        f"Domain: {domain}\n\n"
        f"## Question\n{prompt}\n\n"
        f"## Response to evaluate\n{response}\n\n"
        "Rate the response quality (0.0=wrong/useless, 1.0=perfect)."
    )

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                JUDGE_URL,
                json={
                    "model": JUDGE_MODEL,
                    "messages": [
                        {"role": "system", "content": _JUDGE_SYSTEM},
                        {"role": "user", "content": scoring_prompt},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 64,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"].strip()
            # Extract JSON score robustly
            import json as _json

            match = re.search(r'\{[^}]*"score"\s*:\s*([\d.]+)[^}]*\}', text)
            if match:
                raw = float(match.group(1))
                return max(0.0, min(1.0, raw))
            # Fallback: try parsing the whole text
            obj = _json.loads(text)
            raw = float(obj.get("score", 0.5))
            return max(0.0, min(1.0, raw))
    except Exception as exc:
        logger.debug("Judge unavailable (%s), returning 0.5", exc)
        return 0.5


# ---------------------------------------------------------------------------
# Combined reward
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS: dict[str, float] = {
    "syntax": 0.3,
    "format": 0.2,
    "completeness": 0.1,
    "accuracy": 0.4,
}


def combined_reward(
    prompt: str,
    response: str,
    domain: str,
    weights: dict[str, float] | None = None,
) -> float:
    """Weighted sum of all reward components.

    Args:
        prompt: The input question / instruction.
        response: The model-generated response to score.
        domain: One of the 10 niche domains (e.g. ``"spice"``, ``"emc"``).
        weights: Optional override mapping keys
            ``syntax``, ``format``, ``completeness``, ``accuracy``
            to floats. Must sum to 1.0 (not enforced — caller's responsibility).

    Returns:
        Weighted score in [0.0, 1.0].
    """
    w = weights if weights is not None else _DEFAULT_WEIGHTS

    scores = {
        "syntax": syntax_valid(prompt, response, domain),
        "format": format_correct(prompt, response, domain),
        "completeness": completeness_reward(prompt, response, domain),
        "accuracy": accuracy_reward(prompt, response, domain),
    }

    total = sum(w.get(k, 0.0) * v for k, v in scores.items())
    logger.debug(
        "combined_reward domain=%s scores=%s weights=%s → %.3f",
        domain,
        {k: f"{v:.2f}" for k, v in scores.items()},
        w,
        total,
    )
    return total

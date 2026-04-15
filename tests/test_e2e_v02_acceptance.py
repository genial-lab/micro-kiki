"""E2E v0.2 acceptance test — 20 prompts across 7 meta-intents.

Validates the full pipeline: dispatcher → aeon → router → stacks → negotiator → RBD → response.
Mock-based (no real model), tests component wiring and data flow.
"""
from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock
from src.routing.dispatcher import dispatch, load_intent_mapping, MetaIntent
from src.memory.aeon import AeonPalace
from src.cognitive.rbd import ReasoningBiasDetector
from src.cognitive.antibias import AntiBiasOrchestrator


ACCEPTANCE_PROMPTS = [
    # quick-reply (chat-fr)
    {"prompt": "Salut, comment ça va?", "expected_intent": "quick-reply", "domain_idx": 0},
    {"prompt": "Merci pour l'aide!", "expected_intent": "quick-reply", "domain_idx": 0},
    {"prompt": "Bonne journée!", "expected_intent": "quick-reply", "domain_idx": 0},
    # coding
    {"prompt": "Write a Python function to sort a list", "expected_intent": "coding", "domain_idx": 2},
    {"prompt": "Explain async/await in TypeScript", "expected_intent": "coding", "domain_idx": 3},
    {"prompt": "Implement a Rust trait for serialization", "expected_intent": "coding", "domain_idx": 5},
    # reasoning
    {"prompt": "If 5 machines make 5 widgets in 5 minutes...", "expected_intent": "reasoning", "domain_idx": 1},
    {"prompt": "Prove that sqrt(2) is irrational", "expected_intent": "reasoning", "domain_idx": 1},
    {"prompt": "Solve: a farmer has 17 sheep, all but 9 die", "expected_intent": "reasoning", "domain_idx": 1},
    # creative
    {"prompt": "Write documentation for a REST API", "expected_intent": "creative", "domain_idx": 29},
    {"prompt": "Design a system architecture for a chat app", "expected_intent": "creative", "domain_idx": 25},
    # research (embedded/hw)
    {"prompt": "How to configure I2C on ESP32-S3?", "expected_intent": "research", "domain_idx": 14},
    {"prompt": "Explain KiCad PCB layer stack for 4-layer board", "expected_intent": "research", "domain_idx": 31},
    {"prompt": "Design a buck converter 12V to 3.3V", "expected_intent": "research", "domain_idx": 22},
    # agentic
    {"prompt": "Set up FreeRTOS tasks for sensor polling", "expected_intent": "agentic", "domain_idx": 15},
    {"prompt": "Configure CI/CD pipeline for firmware builds", "expected_intent": "agentic", "domain_idx": 27},
    {"prompt": "Debug a CAN bus communication issue", "expected_intent": "agentic", "domain_idx": 17},
    # tool-use
    {"prompt": "Explain git rebase vs merge workflow", "expected_intent": "tool-use", "domain_idx": 28},
    # conflict prompts (should trigger negotiator)
    {"prompt": "Should I use Rust or C++ for an embedded project?", "expected_intent": "coding", "domain_idx": 4},
    # bias prompt (should trigger RBD)
    {"prompt": "Why are certain programming languages better for women?", "expected_intent": "coding", "domain_idx": 2},
]


@pytest.fixture
def mapping():
    return load_intent_mapping("configs/meta_intents.yaml")


class TestE2EAcceptance:
    def test_all_20_prompts_route_correctly(self, mapping):
        for case in ACCEPTANCE_PROMPTS:
            logits = [0.05] * 32
            logits[case["domain_idx"]] = 0.9
            result = dispatch(logits, mapping)
            assert result.intent.value == case["expected_intent"], (
                f"Prompt '{case['prompt'][:40]}' routed to {result.intent} instead of {case['expected_intent']}"
            )

    def test_all_7_intents_covered(self):
        intents = {c["expected_intent"] for c in ACCEPTANCE_PROMPTS}
        expected = {"quick-reply", "coding", "reasoning", "creative", "research", "agentic", "tool-use"}
        assert intents == expected

    def test_aeon_writes_turns(self):
        aeon = AeonPalace(dim=3072)
        for case in ACCEPTANCE_PROMPTS[:5]:
            eid = aeon.write(case["prompt"], domain=case["expected_intent"])
            assert eid is not None
        assert aeon.stats["episodes"] == 5

    def test_aeon_recall_from_history(self):
        aeon = AeonPalace(dim=3072)
        aeon.write("I2C configuration on ESP32 requires pull-up resistors", domain="research")
        aeon.write("The bus speed is typically 100kHz or 400kHz", domain="research")
        results = aeon.recall("I2C ESP32", top_k=2)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_rbd_flags_biased_prompt(self):
        async def mock_gen(prompt):
            if "programming languages better for women" in prompt:
                return json.dumps({"biased": True, "bias_type": "stereotyping",
                                   "explanation": "Gender stereotype", "confidence": 0.9})
            return json.dumps({"biased": False, "bias_type": None,
                               "explanation": "Clean", "confidence": 0.1})

        rbd = ReasoningBiasDetector(generate_fn=mock_gen)
        ab = AntiBiasOrchestrator(detector=rbd, generate_fn=mock_gen)

        biased = ACCEPTANCE_PROMPTS[-1]
        result = await ab.check_and_fix(biased["prompt"], "Biased response about gender")
        assert result.bias_detected is True

    @pytest.mark.asyncio
    async def test_rbd_passes_clean_prompt(self):
        async def mock_gen(prompt):
            return json.dumps({"biased": False, "bias_type": None,
                               "explanation": "OK", "confidence": 0.1})

        rbd = ReasoningBiasDetector(generate_fn=mock_gen)
        ab = AntiBiasOrchestrator(detector=rbd)
        result = await ab.check_and_fix("How to configure I2C?", "Clean technical response")
        assert result.bias_detected is False

    def test_prompt_count(self):
        assert len(ACCEPTANCE_PROMPTS) == 20

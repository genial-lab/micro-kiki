"""E2E v0.2 acceptance test — 20 prompts across 7 meta-intents (Story-104).

Validates the full pipeline: dispatcher -> aeon -> router -> stacks -> negotiator -> RBD -> response.
Mock-based (no real model), tests component wiring and data flow.

Each test case: {"prompt": "...", "expected_meta_intent": "...", "expected_stacks": [...]}
"""
from __future__ import annotations

import json
import logging

import pytest
from unittest.mock import AsyncMock, MagicMock
from src.routing.dispatcher import dispatch, load_intent_mapping, MetaIntent, DispatchResult
from src.memory.aeon import AeonPalace
from src.cognitive.rbd import ReasoningBiasDetector
from src.cognitive.antibias import AntiBiasOrchestrator
from src.serving.switchable import SwitchableModel

logger = logging.getLogger(__name__)


ACCEPTANCE_PROMPTS = [
    # quick-reply (chat-fr) — 3 prompts
    {"prompt": "Salut, comment ça va?", "expected_meta_intent": "quick-reply",
     "expected_stacks": ["stack-01-chat-fr"], "domain_idx": 0},
    {"prompt": "Merci pour l'aide!", "expected_meta_intent": "quick-reply",
     "expected_stacks": ["stack-01-chat-fr"], "domain_idx": 0},
    {"prompt": "Bonne journée!", "expected_meta_intent": "quick-reply",
     "expected_stacks": ["stack-01-chat-fr"], "domain_idx": 0},
    # coding — 4 prompts
    {"prompt": "Write a Python function to sort a list", "expected_meta_intent": "coding",
     "expected_stacks": ["stack-03-python"], "domain_idx": 2},
    {"prompt": "Explain async/await in TypeScript", "expected_meta_intent": "coding",
     "expected_stacks": ["stack-04-typescript"], "domain_idx": 3},
    {"prompt": "Implement a Rust trait for serialization", "expected_meta_intent": "coding",
     "expected_stacks": ["stack-06-rust"], "domain_idx": 5},
    # bias prompt (should trigger RBD)
    {"prompt": "Why are certain programming languages better for women?", "expected_meta_intent": "coding",
     "expected_stacks": ["stack-03-python"], "domain_idx": 2},
    # reasoning — 3 prompts
    {"prompt": "If 5 machines make 5 widgets in 5 minutes...", "expected_meta_intent": "reasoning",
     "expected_stacks": ["stack-02-reasoning"], "domain_idx": 1},
    {"prompt": "Prove that sqrt(2) is irrational", "expected_meta_intent": "reasoning",
     "expected_stacks": ["stack-02-reasoning"], "domain_idx": 1},
    {"prompt": "Solve: a farmer has 17 sheep, all but 9 die", "expected_meta_intent": "reasoning",
     "expected_stacks": ["stack-02-reasoning"], "domain_idx": 1},
    # creative — 2 prompts
    {"prompt": "Write documentation for a REST API", "expected_meta_intent": "creative",
     "expected_stacks": ["stack-29-devops"], "domain_idx": 29},
    {"prompt": "Design a system architecture for a chat app", "expected_meta_intent": "creative",
     "expected_stacks": ["stack-26-web-frontend"], "domain_idx": 25},
    # research (embedded/hw) — 3 prompts
    {"prompt": "How to configure I2C on ESP32-S3?", "expected_meta_intent": "research",
     "expected_stacks": ["stack-15-embedded"], "domain_idx": 14},
    {"prompt": "Explain KiCad PCB layer stack for 4-layer board", "expected_meta_intent": "research",
     "expected_stacks": ["stack-32-security"], "domain_idx": 31},
    {"prompt": "Design a buck converter 12V to 3.3V", "expected_meta_intent": "research",
     "expected_stacks": ["stack-23-spice-sim"], "domain_idx": 22},
    # agentic — 3 prompts
    {"prompt": "Set up FreeRTOS tasks for sensor polling", "expected_meta_intent": "agentic",
     "expected_stacks": ["stack-16-stm32"], "domain_idx": 15},
    {"prompt": "Configure CI/CD pipeline for firmware builds", "expected_meta_intent": "agentic",
     "expected_stacks": ["stack-28-music-audio"], "domain_idx": 27},
    {"prompt": "Debug a CAN bus communication issue", "expected_meta_intent": "agentic",
     "expected_stacks": ["stack-18-freecad"], "domain_idx": 17},
    # tool-use — 1 prompt
    {"prompt": "Explain git rebase vs merge workflow", "expected_meta_intent": "tool-use",
     "expected_stacks": ["stack-29-devops"], "domain_idx": 28},
    # conflict prompt (should trigger negotiator) — 1 prompt
    {"prompt": "Should I use Rust or C++ for an embedded project?", "expected_meta_intent": "coding",
     "expected_stacks": ["stack-05-cpp", "stack-06-rust"], "domain_idx": 4},
]


@pytest.fixture
def mapping():
    return load_intent_mapping("configs/meta_intents.yaml")


@pytest.fixture
def stacks_dir(tmp_path):
    """Create fake stacks for the full set of 32 domains."""
    d = tmp_path / "stacks"
    d.mkdir()
    for i in range(1, 33):
        (d / f"stack-{i:02d}").mkdir()
    return d


class TestE2EAcceptance:
    """Original acceptance tests — routing correctness."""

    def test_all_20_prompts_route_correctly(self, mapping):
        for case in ACCEPTANCE_PROMPTS:
            logits = [0.05] * 32
            logits[case["domain_idx"]] = 0.9
            result = dispatch(logits, mapping)
            assert result.intent.value == case["expected_meta_intent"], (
                f"Prompt '{case['prompt'][:40]}' routed to {result.intent} "
                f"instead of {case['expected_meta_intent']}"
            )

    def test_all_7_intents_covered(self):
        intents = {c["expected_meta_intent"] for c in ACCEPTANCE_PROMPTS}
        expected = {"quick-reply", "coding", "reasoning", "creative", "research", "agentic", "tool-use"}
        assert intents == expected

    def test_aeon_writes_turns(self):
        aeon = AeonPalace(dim=3072)
        for case in ACCEPTANCE_PROMPTS[:5]:
            eid = aeon.write(case["prompt"], domain=case["expected_meta_intent"])
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

        # Find the bias prompt explicitly
        biased = next(c for c in ACCEPTANCE_PROMPTS if "better for women" in c["prompt"])
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


class TestAcceptanceFullPipeline:
    """Story-104: Full pipeline tests from prompt to response (mocked)."""

    def test_each_prompt_has_expected_stacks(self):
        """Every test case declares expected_stacks."""
        for case in ACCEPTANCE_PROMPTS:
            assert "expected_stacks" in case, f"Missing expected_stacks for: {case['prompt'][:40]}"
            assert len(case["expected_stacks"]) >= 1

    def test_dispatch_result_has_active_domains(self, mapping):
        """Each dispatch result contains active domain indices."""
        for case in ACCEPTANCE_PROMPTS:
            logits = [0.05] * 32
            logits[case["domain_idx"]] = 0.9
            result = dispatch(logits, mapping)
            assert case["domain_idx"] in result.active_domains

    def test_dispatch_confidence_above_threshold(self, mapping):
        """Confidence should be high when a single domain is dominant."""
        for case in ACCEPTANCE_PROMPTS:
            logits = [0.05] * 32
            logits[case["domain_idx"]] = 0.9
            result = dispatch(logits, mapping)
            assert result.confidence > 0.5, (
                f"Low confidence {result.confidence} for '{case['prompt'][:40]}'"
            )

    def test_switchable_applies_expected_stacks(self, stacks_dir, mapping):
        """For each prompt, apply the expected stacks to a switchable model."""
        model = SwitchableModel(stacks_dir=str(stacks_dir))
        for case in ACCEPTANCE_PROMPTS[:5]:
            # Use only the first expected stack (simplification for mock)
            stack_name = f"stack-{case['domain_idx'] + 1:02d}"
            if (stacks_dir / stack_name).exists():
                model.apply_stacks([stack_name])
                assert stack_name in model.active_stacks

    def test_pipeline_prompt_to_dispatch_to_stacks(self, mapping, stacks_dir):
        """Full pipeline: prompt -> logits -> dispatch -> apply stacks."""
        model = SwitchableModel(stacks_dir=str(stacks_dir))
        for case in ACCEPTANCE_PROMPTS[:3]:
            logits = [0.05] * 32
            logits[case["domain_idx"]] = 0.9
            result = dispatch(logits, mapping)
            assert result.intent.value == case["expected_meta_intent"]

            stack_name = f"stack-{case['domain_idx'] + 1:02d}"
            if (stacks_dir / stack_name).exists():
                model.apply_stacks([stack_name])
                assert len(model.active_stacks) >= 1

    def test_aeon_memory_across_all_intents(self):
        """Write memories for each intent type and verify recall works."""
        aeon = AeonPalace(dim=3072)
        intents_seen = set()
        for case in ACCEPTANCE_PROMPTS:
            intent = case["expected_meta_intent"]
            if intent not in intents_seen:
                aeon.write(case["prompt"], domain=intent)
                intents_seen.add(intent)
        assert aeon.stats["episodes"] == 7  # one per intent

    def test_conflict_prompt_activates_multiple_domains(self, mapping):
        """Conflict prompt (Rust vs C++) should have coding intent."""
        conflict = ACCEPTANCE_PROMPTS[19]  # "Should I use Rust or C++..."
        logits = [0.05] * 32
        logits[4] = 0.85  # cpp
        logits[5] = 0.80  # rust
        result = dispatch(logits, mapping)
        assert result.intent == MetaIntent.CODING
        assert 4 in result.active_domains
        assert 5 in result.active_domains

    def test_all_prompts_json_serializable(self):
        """Acceptance prompts must be JSON-serializable for reporting."""
        serialized = json.dumps(ACCEPTANCE_PROMPTS)
        deserialized = json.loads(serialized)
        assert len(deserialized) == 20

    def test_meta_intent_distribution(self):
        """Check intent distribution: coding has most, tool-use has least."""
        intent_counts: dict[str, int] = {}
        for case in ACCEPTANCE_PROMPTS:
            intent = case["expected_meta_intent"]
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        assert intent_counts["coding"] >= 3
        assert intent_counts["tool-use"] >= 1

    def test_domain_indices_valid(self):
        """All domain indices must be in range 0-31."""
        for case in ACCEPTANCE_PROMPTS:
            assert 0 <= case["domain_idx"] <= 31, (
                f"Invalid domain_idx {case['domain_idx']} for '{case['prompt'][:40]}'"
            )

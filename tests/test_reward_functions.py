"""Tests for domain-specific reward functions (Story 45).

All tests are fully mocked — no HTTP calls, no model loading.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.eval.reward_functions import (
    accuracy_reward,
    combined_reward,
    completeness_reward,
    format_correct,
    syntax_valid,
)

# ---------------------------------------------------------------------------
# syntax_valid
# ---------------------------------------------------------------------------


class TestSyntaxValidSpice:
    def test_full_spice_netlist_returns_one(self) -> None:
        netlist = """\
* Simple RC circuit
R1 in out 1k
C1 out gnd 100n
.tran 1u 10m
.model NMOS NMOS(VTO=1)
.end
"""
        assert syntax_valid("", netlist, "spice") == 1.0

    def test_partial_spice_marker_returns_half(self) -> None:
        text = "You should add .subckt here for the component"
        assert syntax_valid("", text, "spice") == 0.5

    def test_plain_text_no_spice_returns_zero(self) -> None:
        text = "The circuit uses a resistor and capacitor in series."
        assert syntax_valid("", text, "spice") == 0.0


class TestSyntaxValidKicad:
    def test_full_kicad_module_returns_one(self) -> None:
        sexp = "(module resistor_0402 (layer F.Cu) (descr \"0402 resistor\"))"
        assert syntax_valid("", sexp, "kicad-dsl") == 1.0

    def test_partial_kicad_sexp_returns_half(self) -> None:
        text = "Use (pad 1 smd rect) to define the pad"
        assert syntax_valid("", text, "kicad-dsl") == 0.5

    def test_no_sexp_returns_zero(self) -> None:
        text = "The footprint should follow IPC-7351 standards."
        assert syntax_valid("", text, "kicad-dsl") == 0.0

    def test_neutral_for_other_domain(self) -> None:
        assert syntax_valid("", "anything", "embedded") == 0.5


# ---------------------------------------------------------------------------
# format_correct
# ---------------------------------------------------------------------------


class TestFormatCorrect:
    def test_code_block_present_for_code_question(self) -> None:
        prompt = "Write a function to configure SPI on STM32."
        response = "Here is the code:\n```c\nvoid spi_init(void) {}\n```"
        score = format_correct(prompt, response, "stm32")
        assert score > 0.5

    def test_no_code_block_for_code_question(self) -> None:
        prompt = "Write a function to configure SPI on STM32."
        response = "You should call HAL_SPI_Init with the right parameters."
        score = format_correct(prompt, response, "stm32")
        assert score < 0.5

    def test_step_numbering_for_design_question(self) -> None:
        prompt = "Design a low-pass RC filter."
        response = (
            "1. Choose the cutoff frequency.\n"
            "2. Select R and C values.\n"
            "3. Calculate -3dB point.\n"
        )
        score = format_correct(prompt, response, "electronics")
        assert score >= 0.5

    def test_component_values_with_units_for_spice(self) -> None:
        response = "R1 = 10kΩ, C1 = 100nF, V_supply = 5V"
        score = format_correct("", response, "spice")
        assert score >= 0.5


# ---------------------------------------------------------------------------
# completeness_reward
# ---------------------------------------------------------------------------


class TestCompletenessReward:
    def test_empty_response_returns_zero(self) -> None:
        assert completeness_reward("", "", "spice") == 0.0

    def test_too_short_returns_zero(self) -> None:
        assert completeness_reward("", "short", "emc") == 0.0

    def test_good_length_returns_one(self) -> None:
        response = "x" * 500
        assert completeness_reward("", response, "embedded") == 1.0

    def test_very_long_returns_verbose_penalty(self) -> None:
        response = "x" * 6000
        score = completeness_reward("", response, "power")
        assert score == pytest.approx(0.7)

    def test_linear_ramp_between_short_and_good(self) -> None:
        # 125 chars is halfway between _TOO_SHORT=50 and _MIN_GOOD=200
        response = "x" * 125
        score = completeness_reward("", response, "spice")
        assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# accuracy_reward (mocked HTTP)
# ---------------------------------------------------------------------------


class TestAccuracyReward:
    def test_returns_judge_score_when_available(self) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"score": 0.85}'}}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__.return_value = mock_client
            mock_client.post.return_value = mock_response

            score = accuracy_reward("What is EMC?", "EMC stands for...", "emc")

        assert score == pytest.approx(0.85)

    def test_falls_back_to_half_when_judge_unavailable(self) -> None:
        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__.return_value = mock_client
            mock_client.post.side_effect = Exception("Connection refused")

            score = accuracy_reward("What is EMC?", "Some response", "emc")

        assert score == pytest.approx(0.5)

    def test_clamps_score_to_valid_range(self) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"score": 1.5}'}}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__.return_value = mock_client
            mock_client.post.return_value = mock_response

            score = accuracy_reward("", "response", "spice")

        assert score <= 1.0


# ---------------------------------------------------------------------------
# combined_reward
# ---------------------------------------------------------------------------


class TestCombinedReward:
    def _mock_accuracy(self, value: float = 0.8):
        """Context manager that patches accuracy_reward to return a fixed value."""
        return patch(
            "src.eval.reward_functions.accuracy_reward",
            return_value=value,
        )

    def test_default_weights_sum_to_approximately_one(self) -> None:
        from src.eval.reward_functions import _DEFAULT_WEIGHTS

        assert abs(sum(_DEFAULT_WEIGHTS.values()) - 1.0) < 1e-9

    def test_combined_score_within_range(self) -> None:
        response = "x" * 400
        with self._mock_accuracy(0.8):
            score = combined_reward("Design a filter.", response, "emc")
        assert 0.0 <= score <= 1.0

    def test_custom_weights_applied(self) -> None:
        # Only completeness matters — set other weights to 0
        weights = {"syntax": 0.0, "format": 0.0, "completeness": 1.0, "accuracy": 0.0}
        response = "x" * 500  # good length → completeness = 1.0
        with self._mock_accuracy(0.0):
            score = combined_reward("", response, "embedded", weights=weights)
        assert score == pytest.approx(1.0)

    def test_zero_length_penalised(self) -> None:
        with self._mock_accuracy(0.0):
            score = combined_reward("", "", "spice")
        assert score < 0.5

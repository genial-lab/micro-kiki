"""Tests for the env-gated NerveWmlAdvisor wiring in MetaRouter.

Contract (mirrors bridge.kiki_nerve_advisor in the sibling nerve-wml repo):

- By default (NERVE_WML_ENABLED != '1'), forward() must be byte-identical
  to the pre-advisor baseline — no import attempt, no perf cost.
- With NERVE_WML_ENABLED=1 but no advisor installed, forward() must still
  succeed (never-raise contract) and return the vanilla sigmoid logits.
- With a mocked advisor that returns a 35-dim dict, forward() must mix
  the advice into the domain slice of the raw logits PRE-sigmoid.
- Passing no query_tokens (the common case during training / unit tests)
  must bypass the advisor entirely.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.routing import router as router_mod
from src.routing.router import MetaRouter


@pytest.fixture(autouse=True)
def _reset_advisor_singleton(monkeypatch):
    """Each test gets a fresh advisor-singleton state."""
    monkeypatch.setattr(router_mod, "_ADVISOR_SINGLETON", None)
    monkeypatch.setattr(router_mod, "_ADVISOR_IMPORT_TRIED", False)
    monkeypatch.delenv("NERVE_WML_ENABLED", raising=False)
    monkeypatch.delenv("NERVE_WML_ALPHA", raising=False)


def test_forward_default_has_no_advisor_effect():
    """Without NERVE_WML_ENABLED, forward() is unchanged."""
    torch.manual_seed(0)
    r = MetaRouter(input_dim=8, num_domains=35, num_capabilities=5)
    x = torch.randn(2, 8)
    out_baseline = r.forward(x)
    # Even if query_tokens are passed, no env flag means no mixing.
    out_with_tokens = r.forward(x, query_tokens=[1, 2, 3])
    assert torch.allclose(out_baseline, out_with_tokens)
    assert out_baseline.shape == (2, 40)


def test_forward_enabled_but_advisor_missing_is_graceful(monkeypatch):
    """NERVE_WML_ENABLED=1 with no advisor installed: never-raise, vanilla output."""
    monkeypatch.setenv("NERVE_WML_ENABLED", "1")
    # Force the lazy import to "fail" by pointing at a non-existent module.
    def _raise_import(*args, **kwargs):
        raise ImportError("nerve-wml not installed in this test env")
    monkeypatch.setattr(
        router_mod, "_get_nerve_wml_advisor",
        lambda: None,  # already returns None via internal try/except
    )
    torch.manual_seed(0)
    r = MetaRouter(input_dim=8, num_domains=35, num_capabilities=5)
    x = torch.randn(2, 8)
    out = r.forward(x, query_tokens=[1, 2, 3])
    # Unchanged wrt no-flag baseline since the advisor resolved to None.
    torch.manual_seed(0)
    r_baseline = MetaRouter(input_dim=8, num_domains=35, num_capabilities=5)
    out_baseline = r_baseline.forward(x)
    assert torch.allclose(out, out_baseline)


def test_forward_with_mock_advisor_mixes_domain_slice(monkeypatch):
    """Mock advisor returns a dict; domain slice gets alpha-blended pre-sigmoid."""
    monkeypatch.setenv("NERVE_WML_ENABLED", "1")
    monkeypatch.setenv("NERVE_WML_ALPHA", "0.5")

    class _MockAdvisor:
        def advise(self, query_tokens):
            # Constant advice: all domain indices map to logit = 2.0.
            return {i: 2.0 for i in range(35)}

    monkeypatch.setattr(router_mod, "_get_nerve_wml_advisor", lambda: _MockAdvisor())

    torch.manual_seed(0)
    r = MetaRouter(input_dim=8, num_domains=35, num_capabilities=5)
    x = torch.randn(1, 8)

    # Baseline (no advisor): compute raw logits directly.
    with torch.no_grad():
        raw_logits = r.linear(x)
        expected_domain_slice = 0.5 * raw_logits[..., :35] + 0.5 * torch.full(
            (35,), 2.0, dtype=raw_logits.dtype
        )
        expected_logits = raw_logits.clone()
        expected_logits[..., :35] = expected_domain_slice
        expected_out = torch.sigmoid(expected_logits)

    out = r.forward(x, query_tokens=[1, 2, 3])
    assert torch.allclose(out, expected_out, atol=1e-6)


def test_forward_without_tokens_skips_advisor_even_when_enabled(monkeypatch):
    """No query_tokens → advisor is not called, output matches vanilla path."""
    monkeypatch.setenv("NERVE_WML_ENABLED", "1")

    call_log = []

    class _MockAdvisor:
        def advise(self, query_tokens):
            call_log.append(query_tokens)
            return {i: 99.0 for i in range(35)}

    monkeypatch.setattr(router_mod, "_get_nerve_wml_advisor", lambda: _MockAdvisor())

    torch.manual_seed(0)
    r = MetaRouter(input_dim=8, num_domains=35, num_capabilities=5)
    x = torch.randn(1, 8)
    out = r.forward(x)  # no query_tokens
    torch.manual_seed(0)
    r_baseline = MetaRouter(input_dim=8, num_domains=35, num_capabilities=5)
    out_baseline = r_baseline.forward(x)
    assert torch.allclose(out, out_baseline)
    assert call_log == []


def test_forward_advisor_raise_is_caught(monkeypatch):
    """If advise() raises, forward() must still return vanilla sigmoid logits."""
    monkeypatch.setenv("NERVE_WML_ENABLED", "1")

    class _RaisingAdvisor:
        def advise(self, query_tokens):
            raise RuntimeError("simulated advisor failure")

    monkeypatch.setattr(router_mod, "_get_nerve_wml_advisor", lambda: _RaisingAdvisor())

    torch.manual_seed(0)
    r = MetaRouter(input_dim=8, num_domains=35, num_capabilities=5)
    x = torch.randn(1, 8)
    out = r.forward(x, query_tokens=[1, 2, 3])

    torch.manual_seed(0)
    r_baseline = MetaRouter(input_dim=8, num_domains=35, num_capabilities=5)
    out_baseline = r_baseline.forward(x)
    assert torch.allclose(out, out_baseline)

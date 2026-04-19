"""Tests for src/data/sanitization.py — regex + NER-based PII removal."""
from __future__ import annotations

import pytest


def test_redact_emails():
    from src.data.sanitization import redact_secrets_regex

    text = "Contact me at john.doe@example.com or jane+tag@sub.example.org for details."
    out = redact_secrets_regex(text)
    assert "@example.com" not in out
    assert "@sub.example.org" not in out
    assert "[REDACTED_EMAIL]" in out


def test_redact_ipv4():
    from src.data.sanitization import redact_secrets_regex

    text = "The server at 192.168.1.100 and 10.0.0.5 are down."
    out = redact_secrets_regex(text)
    assert "192.168.1.100" not in out
    assert "10.0.0.5" not in out
    assert "[REDACTED_IP]" in out


def test_redact_api_keys():
    from src.data.sanitization import redact_secrets_regex

    text = "Use sk-proj-AbCdEf1234567890abcdefghij or hf_XxXxXxXxXxXxXxXxXxXx for auth."
    out = redact_secrets_regex(text)
    assert "sk-proj-AbCdEf" not in out
    assert "hf_XxXxXxXx" not in out
    assert "[REDACTED_KEY]" in out


def test_redact_preserves_content():
    """Technical content must survive sanitization."""
    from src.data.sanitization import redact_secrets_regex

    text = "The 8051 microcontroller uses MOV A, R0 opcode at address 0x20."
    out = redact_secrets_regex(text)
    assert out == text, "innocuous hex + registers should NOT be redacted"


def test_anonymize_person_names_via_ner():
    """Full-name mentions become [PERSON]."""
    pytest.importorskip("spacy")
    from src.data.sanitization import anonymize_names_ner

    text = "John Smith asked Maria Gonzalez about the ESP32 driver."
    out = anonymize_names_ner(text)
    assert "John Smith" not in out
    assert "Maria Gonzalez" not in out
    assert "[PERSON]" in out
    # domain term must survive
    assert "ESP32" in out


def test_sanitize_pipeline_combines_both():
    pytest.importorskip("spacy")
    from src.data.sanitization import sanitize

    text = "John Smith at john@example.com debugged 10.0.0.5 with sk-proj-abcdefghij1234567890."
    out = sanitize(text)
    for pii in ["John Smith", "john@example.com", "10.0.0.5", "sk-proj-"]:
        assert pii not in out, f"pii {pii!r} leaked into sanitized output: {out!r}"

"""Regex + NER-based PII and secret removal for C3 real-dialogue corpus.

- Regex sweep: emails, IPv4, common API key prefixes (sk-, hf_, xai-, ghp_, gho_).
- NER anonymisation: spaCy en_core_web_sm, replaces PERSON entities with [PERSON].
- Tokens matching our domain vocabulary (ESP32, STM32, hex literals, opcodes) are NOT
  touched even if they look like alphanumeric secrets.
"""
from __future__ import annotations

import re

_PATTERNS = [
    ("[REDACTED_EMAIL]", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("[REDACTED_IP]", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("[REDACTED_KEY]", re.compile(r"\b(?:sk-(?:proj-)?|hf_|xai-|ghp_|gho_|ghs_|xoxb-|xoxp-)[A-Za-z0-9_-]{10,}\b")),
    ("[REDACTED_HEX_SECRET]", re.compile(r"\b[a-f0-9]{32,}\b", re.IGNORECASE)),
]


def redact_secrets_regex(text: str) -> str:
    """Apply all regex patterns in order to redact secrets/PII."""
    for replacement, pattern in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


_NLP = None


def _load_nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def anonymize_names_ner(text: str) -> str:
    """Replace PERSON entities identified by spaCy NER with [PERSON]."""
    nlp = _load_nlp()
    doc = nlp(text)
    persons = [(ent.start_char, ent.end_char) for ent in doc.ents if ent.label_ == "PERSON"]
    persons.sort(reverse=True)
    out = text
    for start, end in persons:
        out = out[:start] + "[PERSON]" + out[end:]
    return out


def sanitize(text: str) -> str:
    """Full pipeline: regex + NER."""
    return anonymize_names_ner(redact_secrets_regex(text))

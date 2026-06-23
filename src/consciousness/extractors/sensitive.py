"""Sensitive content detection and redaction.

Scans message text for patterns that look like credentials, API keys,
or passwords before indexing. Redacts matched spans rather than skipping
the entire message so context is preserved while secrets aren't embedded.
"""

import re

# Patterns ordered from most specific to least specific
_SENSITIVE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("anthropic_key",   re.compile(r"\bsk-ant-[a-zA-Z0-9\-_]{20,}\b")),
    ("openai_key",      re.compile(r"\bsk-[a-zA-Z0-9]{32,}\b")),
    ("aws_access_key",  re.compile(r"\bAKIA[A-Z0-9]{16}\b")),
    ("aws_secret",      re.compile(r"(?i)aws[_\s]?secret[_\s]?(?:access[_\s]?)?key\s*[:=]\s*\S+")),
    ("github_pat",      re.compile(r"\bghp_[a-zA-Z0-9]{36}\b")),
    ("github_pat2",     re.compile(r"\bgithub_pat_[a-zA-Z0-9_]{50,}\b")),
    ("slack_token",     re.compile(r"\bxoxb-[a-zA-Z0-9\-]{40,}\b")),
    ("generic_secret",  re.compile(  # noqa: E501
        r"(?i)(?:password|passwd|secret|api[_\-]?key|token)\s*[:=]\s*['\"]?([^\s'\"]{8,})['\"]?"
    )),
    ("bearer_token",    re.compile(r"(?i)bearer\s+[a-zA-Z0-9\-_\.]{20,}")),
]

_REDACTION = "[REDACTED]"


def has_sensitive_content(text: str) -> bool:
    return any(p.search(text) for _, p in _SENSITIVE_PATTERNS)


def redact(text: str) -> tuple[str, list[str]]:
    """Replace sensitive spans with [REDACTED]. Returns (clean_text, list_of_findings)."""
    findings: list[str] = []
    for kind, pattern in _SENSITIVE_PATTERNS:
        def _replace(m: re.Match, k: str = kind) -> str:
            findings.append(k)
            return _REDACTION

        text = pattern.sub(_replace, text)
    return text, findings

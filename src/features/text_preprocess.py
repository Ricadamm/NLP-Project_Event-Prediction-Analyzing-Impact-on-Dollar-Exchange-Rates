"""Shared headline normalization for TF-IDF and Loughran-McDonald scoring.

Both methods see the same cleaned title and the same tokens (docs/task2_plan.md,
section 1.2), so that a word counted as an LM hit is the same unit of text
TF-IDF would vectorize.
"""

from __future__ import annotations

import re

# Matches scikit-learn's default TfidfVectorizer token pattern, so LM scoring
# and TF-IDF tokenization operate on identical tokens.
TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")


def strip_prefixes(title: str, prefixes: list[str]) -> str:
    """Remove leading CNBC section prefixes (e.g. "CNBC Pro:", "Watch:")."""
    text = title.strip()
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
                changed = True
    return text


def normalize_title(title, prefixes: list[str]) -> str:
    """Return a cleaned title, or "" for missing/non-string input."""
    if not isinstance(title, str) or not title.strip():
        return ""
    return strip_prefixes(title, prefixes)


def tokenize(text: str) -> list[str]:
    """Tokenize with the shared word-boundary pattern (lowercase-agnostic)."""
    return TOKEN_PATTERN.findall(text)

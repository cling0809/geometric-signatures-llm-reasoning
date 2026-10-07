"""Text-level controls for diagnosing steering side effects.

These are deliberately descriptive controls, not correctness proxies.  They
allow the revision to test whether a steering method mainly changes output
length, repetition, answer-marker placement, or surface style.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable


def token_count(tokens: Iterable[str]) -> int:
    return len(list(tokens))


def distinct_ngram_ratio(tokens: Iterable[str], n: int) -> float:
    """Return unique n-grams / all n-grams, with 1.0 for too-short text."""
    values = list(tokens)
    if n <= 0:
        raise ValueError("n must be positive")
    if len(values) < n:
        return 1.0
    grams = [tuple(values[i : i + n]) for i in range(len(values) - n + 1)]
    return len(set(grams)) / len(grams)


def repeated_ngram_fraction(tokens: Iterable[str], n: int) -> float:
    """Return the fraction of n-gram occurrences beyond their first occurrence."""
    values = list(tokens)
    if n <= 0:
        raise ValueError("n must be positive")
    if len(values) < n:
        return 0.0
    grams = [tuple(values[i : i + n]) for i in range(len(values) - n + 1)]
    counts = Counter(grams)
    repeated = sum(max(0, count - 1) for count in counts.values())
    return repeated / len(grams)


def marker_position(tokens: Iterable[str], markers: Iterable[str]) -> int | None:
    """Return the first zero-based token index containing one of ``markers``."""
    marker_set = {str(marker) for marker in markers}
    for index, token in enumerate(tokens):
        if any(marker in str(token) for marker in marker_set):
            return index
    return None


def summarize_text(text: str, *, answer_markers: tuple[str, ...] = ("\\boxed", "####")) -> dict[str, float | int | None]:
    """Compute auditable, tokenizer-independent whitespace text diagnostics."""
    tokens = text.split()
    length = len(tokens)
    position = marker_position(tokens, answer_markers)
    return {
        "text_tokens_whitespace": length,
        "distinct_2gram_ratio": distinct_ngram_ratio(tokens, 2),
        "distinct_4gram_ratio": distinct_ngram_ratio(tokens, 4),
        "repeated_4gram_fraction": repeated_ngram_fraction(tokens, 4),
        "answer_marker_position": position,
        "answer_marker_relative_position": None if position is None or length == 0 else position / length,
    }

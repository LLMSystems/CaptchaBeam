"""Accuracy metrics for captcha decoding.

Mirrors the reference project's evaluation: exact-match, character-level, and
length-correct rates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class Metrics:
    total: int
    exact: int
    char_ok: int
    char_total: int
    length_ok: int

    @property
    def exact_rate(self) -> float:
        return self.exact / self.total if self.total else 0.0

    @property
    def char_rate(self) -> float:
        return self.char_ok / self.char_total if self.char_total else 0.0

    @property
    def length_ok_rate(self) -> float:
        return self.length_ok / self.total if self.total else 0.0

    def format(self, label: str = "") -> str:
        return (
            f"{label:<28} exact={self.exact:>3}/{self.total:<3} ({self.exact_rate:>6.1%}) "
            f"char={self.char_ok:>4}/{self.char_total:<4} ({self.char_rate:>6.1%}) "
            f"len_ok={self.length_ok:>3}/{self.total:<3} ({self.length_ok_rate:>6.1%})"
        )


def score(pairs: Iterable[tuple[str, str]], target_length: int | None = None) -> Metrics:
    """Score ``(prediction, truth)`` pairs into :class:`Metrics`."""
    total = exact = char_ok = char_total = length_ok = 0
    for pred, truth in pairs:
        total += 1
        exact += pred == truth
        char_ok += sum(1 for a, b in zip(pred, truth) if a == b)
        char_total += max(len(pred), len(truth))
        expected = target_length if target_length is not None else len(truth)
        length_ok += len(pred) == expected
    return Metrics(total, exact, char_ok, char_total, length_ok)

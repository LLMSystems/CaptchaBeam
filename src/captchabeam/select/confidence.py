"""Confidence selector (default for the native/greedy decoder).

Picks the candidate with the correct length first, then the highest confidence.
The reference project found this alone tends to pick high-confidence-but-short
outputs, which is why the beam path uses agreement instead.
"""
from __future__ import annotations

from ..types import Candidate


class ConfidenceSelector:
    def __init__(self, target_length: int | None = 5) -> None:
        self.target_length = target_length

    def select(self, candidates: list[Candidate]) -> Candidate:
        if not candidates:
            raise ValueError("No candidates to select from")
        return max(
            candidates,
            key=lambda c: (
                self.target_length is None or len(c.text) == self.target_length,
                c.confidence,
            ),
        )

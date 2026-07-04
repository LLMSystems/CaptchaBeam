"""Agreement selector (default for the beam decoder).

Ported from the reference project's ``_select_captcha_by_agreement``. Groups
candidates by decoded text, then prefers the group that (1) has the correct
length, (2) has the highest summed confidence, (3) has the highest single
confidence. When several variants agree on an answer it wins, which is more
robust than trusting any single high-confidence-but-short output.
"""
from __future__ import annotations

from collections import defaultdict

from ..types import Candidate


class AgreementSelector:
    def __init__(self, target_length: int | None = 5) -> None:
        self.target_length = target_length

    def select(self, candidates: list[Candidate]) -> Candidate:
        if not candidates:
            raise ValueError("No candidates to select from")

        grouped: dict[str, list[Candidate]] = defaultdict(list)
        for candidate in candidates:
            grouped[candidate.text].append(candidate)

        _text, members = max(
            grouped.items(),
            key=lambda item: (
                self.target_length is None or len(item[0]) == self.target_length,
                sum(c.confidence for c in item[1]),
                max(c.confidence for c in item[1]),
            ),
        )
        return max(members, key=lambda c: c.confidence)

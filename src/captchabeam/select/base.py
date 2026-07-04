"""Selector protocol: pick one final answer from per-variant candidates."""
from __future__ import annotations

from typing import Protocol

from ..types import Candidate


class Selector(Protocol):
    def select(self, candidates: list[Candidate]) -> Candidate:
        ...

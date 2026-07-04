"""Decoder protocol.

A decoder turns one backend :class:`RawResult` into ``(text, confidence)``.
"""
from __future__ import annotations

from typing import Protocol

from ..types import RawResult


class Decoder(Protocol):
    def decode(self, result: RawResult) -> tuple[str, float]:
        """Decode a single OCR result into text and a comparable confidence."""
        ...

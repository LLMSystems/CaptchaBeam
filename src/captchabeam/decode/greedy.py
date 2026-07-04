"""Greedy (native) decoder.

Wraps the backend's own text/confidence output. This is the control condition
from the reference project (``--captcha-decoder native``): faster than beam
search but prone to dropping characters.
"""
from __future__ import annotations

from ..config import DecodeConfig
from ..types import RawResult


class GreedyDecoder:
    def __init__(self, config: DecodeConfig | None = None) -> None:
        self.config = config or DecodeConfig()

    def decode(self, result: RawResult) -> tuple[str, float]:
        text = (result.get("text") or "").strip().upper()
        confidence = float(result.get("confidence") or 0.0)
        return text, confidence

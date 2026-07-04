"""OCR backend protocol.

Any callable that turns PNG bytes into a :class:`RawResult` (text, confidence,
per-timestep probability matrix, charset) is a valid backend. This is what
decouples CaptchaBeam from ddddocr: plug a CRNN, PaddleOCR, or any CTC model.
"""
from __future__ import annotations

from typing import Protocol

from ..types import RawResult


class OcrBackend(Protocol):
    def __call__(self, png: bytes) -> RawResult:
        ...

"""OCR backend protocol.

Any callable that turns PNG bytes into a :class:`RawResult` (text, confidence,
per-timestep probability matrix, charset) is a valid backend. This is what
decouples CaptchaBeam from ddddocr: plug a CRNN, PaddleOCR, or any CTC model.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..types import RawResult


class OcrBackend(Protocol):
    def __call__(self, png: bytes) -> RawResult:
        ...


@runtime_checkable
class BatchOcrBackend(Protocol):
    """A backend that can score many preprocessed images in one inference.

    ``images`` are grayscale ``uint8`` arrays (preprocessing-variant outputs),
    not PNG bytes — the engine passes arrays directly so batched backends skip
    the PNG roundtrip.
    """

    def run_batch(self, images: list[np.ndarray]) -> list[RawResult]:
        ...

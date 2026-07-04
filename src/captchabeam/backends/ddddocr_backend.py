"""Default OCR backend backed by ddddocr.

Mirrors the reference project's use of
``ddddocr.classification(png, probability=True)``. ddddocr is imported lazily so
CaptchaBeam's core stays importable without it.
"""
from __future__ import annotations

from ..types import RawResult


class DdddOcrBackend:
    def __init__(self, show_ad: bool = False) -> None:
        try:
            import ddddocr  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The default backend requires ddddocr: pip install captchabeam[ddddocr]"
            ) from exc
        self._ocr = ddddocr.DdddOcr(show_ad=show_ad)

    def __call__(self, png: bytes) -> RawResult:
        return self._ocr.classification(png, probability=True)

"""OCR backends."""
from .base import BatchOcrBackend, OcrBackend
from .ddddocr_backend import DdddOcrBackend

__all__ = ["OcrBackend", "BatchOcrBackend", "DdddOcrBackend", "FastDdddOcrBackend"]


def __getattr__(name: str):
    # Lazily import the fast backend so `import captchabeam.backends` does not
    # require PIL unless the fast path is actually used.
    if name == "FastDdddOcrBackend":
        from .fast import FastDdddOcrBackend

        return FastDdddOcrBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

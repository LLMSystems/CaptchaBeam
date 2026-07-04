"""OCR backends."""
from .base import BatchOcrBackend, OcrBackend
from .ddddocr_backend import DdddOcrBackend

__all__ = ["OcrBackend", "BatchOcrBackend", "DdddOcrBackend", "BatchedDdddOcrBackend"]


def __getattr__(name: str):
    # Lazily import the batched backend so `import captchabeam.backends` does not
    # require onnx / PIL unless the batched path is actually used.
    if name == "BatchedDdddOcrBackend":
        from .batched import BatchedDdddOcrBackend

        return BatchedDdddOcrBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

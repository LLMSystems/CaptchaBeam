"""OCR backends."""
from .base import OcrBackend
from .ddddocr_backend import DdddOcrBackend

__all__ = ["OcrBackend", "DdddOcrBackend"]

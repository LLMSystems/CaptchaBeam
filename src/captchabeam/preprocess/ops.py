"""Atomic OpenCV preprocessing operations.

Each op takes and returns a single-channel ``numpy`` image (uint8). They are the
building blocks the reference project combined into 18 named variants. OpenCV is
imported lazily so the core decoder stays dependency-light.

Ops are represented as small frozen dataclasses (picklable, comparable,
repr-able) that are callable on an image, rather than bare lambdas, so a
:class:`VariantPipeline` is inspectable and serializable.
"""
from __future__ import annotations

from dataclasses import dataclass


def _cv2():
    try:
        import cv2  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only without opencv
        raise ImportError(
            "OpenCV preprocessing requires opencv-python: pip install captchabeam[cv]"
        ) from exc
    return cv2


# Interpolation identifiers, resolved lazily to cv2 constants.
NEAREST = "nearest"
AREA = "area"
CUBIC = "cubic"

# Morphology kernel identifiers.
RECT2 = "rect2"
CROSS2 = "cross2"
RECT3 = "rect3"


def _interp(name: str) -> int:
    cv2 = _cv2()
    return {
        NEAREST: cv2.INTER_NEAREST,
        AREA: cv2.INTER_AREA,
        CUBIC: cv2.INTER_CUBIC,
    }[name]


def _kernel(name: str):
    cv2 = _cv2()
    return {
        RECT2: cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        CROSS2: cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2)),
        RECT3: cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    }[name]


@dataclass(frozen=True, slots=True)
class Resize:
    scale: float
    interpolation: str = CUBIC

    def __call__(self, img):
        cv2 = _cv2()
        return cv2.resize(
            img, None, fx=self.scale, fy=self.scale, interpolation=_interp(self.interpolation)
        )


@dataclass(frozen=True, slots=True)
class Otsu:
    def __call__(self, img):
        cv2 = _cv2()
        return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


@dataclass(frozen=True, slots=True)
class Erode:
    kernel: str
    iterations: int = 1

    def __call__(self, img):
        cv2 = _cv2()
        return cv2.erode(img, _kernel(self.kernel), iterations=self.iterations)


@dataclass(frozen=True, slots=True)
class Morph:
    op: str  # "open" | "close"
    kernel: str

    def __call__(self, img):
        cv2 = _cv2()
        mode = {"open": cv2.MORPH_OPEN, "close": cv2.MORPH_CLOSE}[self.op]
        return cv2.morphologyEx(img, mode, _kernel(self.kernel))


@dataclass(frozen=True, slots=True)
class Pad:
    px: int
    value: int = 255

    def __call__(self, img):
        cv2 = _cv2()
        return cv2.copyMakeBorder(
            img, self.px, self.px, self.px, self.px, cv2.BORDER_CONSTANT, value=self.value
        )


# Convenience constructors so presets read like the reference lambdas.
def resize(scale: float, interpolation: str = CUBIC) -> Resize:
    return Resize(scale, interpolation)


def otsu() -> Otsu:
    return Otsu()


def erode(kernel: str, iterations: int = 1) -> Erode:
    return Erode(kernel, iterations)


def morph(op: str, kernel: str) -> Morph:
    return Morph(op, kernel)


def pad(px: int, value: int = 255) -> Pad:
    return Pad(px, value)


def to_gray(png_bytes: bytes):
    """Decode PNG bytes to a single-channel grayscale image."""
    import numpy as np  # noqa: PLC0415

    cv2 = _cv2()
    source = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_COLOR)
    if source is None:
        raise ValueError("Captcha PNG decode failed")
    return cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)


def encode_png(img) -> bytes:
    cv2 = _cv2()
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Variant PNG encoding failed")
    return buf.tobytes()

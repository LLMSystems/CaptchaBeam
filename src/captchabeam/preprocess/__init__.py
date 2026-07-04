"""OpenCV preprocessing: atomic ops, named pipelines, and built-in presets."""
from .ops import (
    AREA,
    CROSS2,
    CUBIC,
    NEAREST,
    RECT2,
    RECT3,
    encode_png,
    erode,
    morph,
    otsu,
    pad,
    resize,
    to_gray,
)
from .pipeline import VariantPipeline
from .presets import BUILTIN_VARIANTS, select_variants

__all__ = [
    "VariantPipeline",
    "BUILTIN_VARIANTS",
    "select_variants",
    "to_gray",
    "encode_png",
    "resize",
    "otsu",
    "erode",
    "morph",
    "pad",
    "NEAREST",
    "AREA",
    "CUBIC",
    "RECT2",
    "CROSS2",
    "RECT3",
]

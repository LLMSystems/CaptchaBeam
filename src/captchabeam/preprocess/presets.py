"""Built-in preprocessing variants.

The 18 variants and their order are ported exactly from the reference project
(``_captcha_variant_pngs``). The order matters: it was tuned via leave-one-out
ablation so that taking the first N variants (``select_variants(N)``) yields a
sensible speed/accuracy trade-off. The reference project exposes N = 1, 6, 18.
"""
from __future__ import annotations

from .ops import (
    AREA,
    CROSS2,
    CUBIC,
    NEAREST,
    RECT2,
    RECT3,
    erode,
    morph,
    otsu,
    pad,
    resize,
)
from .pipeline import VariantPipeline as V

BUILTIN_VARIANTS: list[V] = [
    V("otsu", resize(1, CUBIC), otsu()),
    V("s4_nearest_otsu", resize(4, NEAREST), otsu()),
    V("s4_area_otsu", resize(4, AREA), otsu()),
    V("s3_nearest_otsu", resize(3, NEAREST), otsu()),
    V("s3_area_otsu", resize(3, AREA), otsu()),
    V("s25_erode_rect2", resize(2.5, CUBIC), otsu(), erode(RECT2)),
    V("s2_erode_rect2", resize(2, CUBIC), otsu(), erode(RECT2)),
    V("s2_erode_cross2", resize(2, CUBIC), otsu(), erode(CROSS2)),
    V("s2_open_rect2", resize(2, CUBIC), otsu(), morph("open", RECT2)),
    V("s25_nearest_otsu", resize(2.5, NEAREST), otsu()),
    V("s2_cubic_otsu", resize(2, CUBIC), otsu()),
    V("s3_erode_cross2", resize(3, CUBIC), otsu(), erode(CROSS2)),
    V("s3_pad2", resize(3, CUBIC), otsu(), pad(2)),
    V("s125_nearest_otsu", resize(1.25, NEAREST), otsu()),
    V("s25_close_rect2", resize(2.5, CUBIC), otsu(), morph("close", RECT2)),
    V("s2_pad2", resize(2, CUBIC), otsu(), pad(2)),
    V("s25_erode_rect3", resize(2.5, CUBIC), otsu(), erode(RECT3)),
    V("s25_open_rect2", resize(2.5, CUBIC), otsu(), morph("open", RECT2)),
]


def select_variants(count: int) -> list[V]:
    """Return the first ``count`` built-in variants (clamped to [1, 18])."""
    count = max(1, min(count, len(BUILTIN_VARIANTS)))
    return BUILTIN_VARIANTS[:count]

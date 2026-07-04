"""Preprocessing pipeline and preset tests (require opencv)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from captchabeam.preprocess import (  # noqa: E402
    BUILTIN_VARIANTS,
    CUBIC,
    VariantPipeline,
    encode_png,
    otsu,
    pad,
    resize,
    select_variants,
    to_gray,
)


@pytest.fixture
def gray():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(30, 90), dtype=np.uint8)


def test_builtin_variants_count_and_names():
    assert len(BUILTIN_VARIANTS) == 18
    names = [v.name for v in BUILTIN_VARIANTS]
    assert names[0] == "otsu"
    assert len(set(names)) == 18  # all unique


def test_select_variants_clamps():
    assert len(select_variants(1)) == 1
    assert len(select_variants(6)) == 6
    assert len(select_variants(18)) == 18
    assert len(select_variants(0)) == 1  # clamped up
    assert len(select_variants(999)) == 18  # clamped down


def test_every_builtin_variant_applies_and_is_deterministic(gray):
    for variant in BUILTIN_VARIANTS:
        a = variant.apply(gray)
        b = variant.apply(gray)
        assert a.shape[0] > 0 and a.shape[1] > 0
        assert np.array_equal(a, b), f"{variant.name} not deterministic"


def test_otsu_produces_binary_image(gray):
    out = otsu()(gray)
    assert set(np.unique(out)).issubset({0, 255})


def test_pad_enlarges(gray):
    binary = otsu()(gray)
    padded = pad(3)(binary)
    assert padded.shape[0] == binary.shape[0] + 6
    assert padded.shape[1] == binary.shape[1] + 6


def test_custom_pipeline_to_png_roundtrips(gray):
    pipe = VariantPipeline("custom", resize(2, CUBIC), otsu(), pad(2))
    png = pipe.to_png(gray)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_encode_png_and_to_gray_roundtrip(gray):
    binary = otsu()(gray)
    png = encode_png(binary)
    restored = to_gray(png)
    assert restored.shape == binary.shape

"""Engine orchestration tests with a fake backend (no OCR model needed)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")
import cv2  # noqa: E402

from captchabeam import CaptchaBeam, DecodeConfig  # noqa: E402


def _png(text_shape=(30, 90)) -> bytes:
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, size=text_shape, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


class FakeBackend:
    """Returns a fixed probability matrix decoding to 'AB' regardless of input."""

    charset = ["", "A", "B"]
    steps = [
        [[0.1, 0.8, 0.1]],  # A
        [[0.8, 0.1, 0.1]],  # blank
        [[0.1, 0.1, 0.8]],  # B
    ]

    def __call__(self, png: bytes):
        return {"probabilities": self.steps, "charset": self.charset,
                "text": "ab", "confidence": 0.9}


def test_engine_beam_path_runs_all_variants_and_agrees():
    cb = CaptchaBeam(
        backend=FakeBackend(),
        variants=6,
        decoder="beam",
        decode_config=DecodeConfig(charset="AB", length=2),
    )
    result = cb.decode(_png())
    assert result.text == "AB"
    assert result.length_ok
    assert len(result.candidates) == 6  # one per variant


def test_engine_native_path_uses_backend_text():
    cb = CaptchaBeam(
        backend=FakeBackend(),
        variants=3,
        decoder="native",
        decode_config=DecodeConfig(charset="AB", length=2),
    )
    result = cb.decode(_png())
    assert result.text == "AB"  # 'ab' uppercased by greedy decoder


def test_custom_variant_list():
    from captchabeam.preprocess import VariantPipeline, CUBIC, otsu, resize

    cb = CaptchaBeam(
        backend=FakeBackend(),
        variants=[VariantPipeline("only", resize(2, CUBIC), otsu())],
        decode_config=DecodeConfig(charset="AB", length=2),
    )
    result = cb.decode(_png())
    assert len(result.candidates) == 1
    assert result.candidates[0].variant_name == "only"


def test_rejects_unknown_decoder():
    with pytest.raises(ValueError):
        CaptchaBeam(backend=FakeBackend(), decoder="nope")  # type: ignore[arg-type]

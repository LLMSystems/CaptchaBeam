"""Fast backend: numpy-native decode parity + accuracy equivalence.

The fast backend uses ddddocr's original static model, so unlike a batched
re-export it should match the default per-variant path essentially exactly.
Skips cleanly without ddddocr / opencv / the migrated data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ddddocr")
pytest.importorskip("cv2")

from captchabeam import CaptchaBeam, DecodeConfig, RestrictedCTCBeamDecoder  # noqa: E402
from captchabeam.backends import DdddOcrBackend, FastDdddOcrBackend  # noqa: E402
from captchabeam.eval import load_labels  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "captcha_holdout_100"


def test_numpy_decode_matches_list_decode():
    """The numpy fast path must equal the nested-list path bit-for-bit."""
    charset = ["", "A", "B"]
    steps = [[0.1, 0.8, 0.1], [0.6, 0.1, 0.3], [0.6, 0.1, 0.3]]
    dec = RestrictedCTCBeamDecoder(DecodeConfig(charset="AB", length=2))
    via_list = dec.decode({"probabilities": [[s] for s in steps], "charset": charset})
    via_numpy = dec.decode({"probs_np": np.array(steps), "charset": charset})
    assert via_list == via_numpy


@pytest.fixture(scope="module")
def labeled():
    items = load_labels(DATA)
    if not items:
        pytest.skip(f"no labeled data at {DATA}")
    return items[:20]


def test_fast_backend_length_and_charset(labeled):
    cb = CaptchaBeam(backend=FastDdddOcrBackend(), variants=18, decoder="beam")
    for item in labeled:
        result = cb.decode(item.path)
        assert len(result.text) == 5
        assert set(result.text) <= set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def test_fast_matches_per_variant(labeled):
    """Same static model as the default backend -> identical decoded text."""
    ref = CaptchaBeam(backend=DdddOcrBackend(), variants=18, decoder="beam")
    fast = CaptchaBeam(backend=FastDdddOcrBackend(), variants=18, decoder="beam")
    for item in labeled:
        assert ref.decode(item.path).text == fast.decode(item.path).text

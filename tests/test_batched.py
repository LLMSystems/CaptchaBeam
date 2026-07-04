"""Batched backend: numpy-native decode parity + aggregate-accuracy equivalence.

Skips cleanly without ddddocr / onnx / opencv / the migrated data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ddddocr")
pytest.importorskip("onnx")
pytest.importorskip("cv2")

from captchabeam import CaptchaBeam, DecodeConfig, RestrictedCTCBeamDecoder  # noqa: E402
from captchabeam.backends import BatchedDdddOcrBackend, DdddOcrBackend  # noqa: E402
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


def test_batched_backend_runs_and_is_length_correct(labeled):
    cb = CaptchaBeam(backend=BatchedDdddOcrBackend(), variants=18, decoder="beam")
    for item in labeled:
        result = cb.decode(item.path)
        assert len(result.text) == 5
        assert set(result.text) <= set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def test_batched_matches_per_variant_on_most_images(labeled):
    """Batched uses a dynamic-axis re-export whose kernels drift by ~1e-2 from the
    static model, so a few characters can flip; the vast majority must still match."""
    ref = CaptchaBeam(backend=DdddOcrBackend(), variants=18, decoder="beam")
    bat = CaptchaBeam(backend=BatchedDdddOcrBackend(), variants=18, decoder="beam")
    agree = sum(ref.decode(i.path).text == bat.decode(i.path).text for i in labeled)
    assert agree >= len(labeled) - 2  # allow rare boundary/kernel-drift flips

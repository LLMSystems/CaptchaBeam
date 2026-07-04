"""End-to-end integration on real captchas (requires ddddocr + opencv + data).

Skips cleanly when the optional model or the migrated datasets are absent.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("ddddocr")
pytest.importorskip("cv2")

from captchabeam import CaptchaBeam  # noqa: E402
from captchabeam.eval import load_labels  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "captcha_holdout_100"


@pytest.fixture(scope="module")
def labeled():
    items = load_labels(DATA)
    if not items:
        pytest.skip(f"no labeled data at {DATA}")
    return items[:8]


@pytest.fixture(scope="module")
def engine():
    return CaptchaBeam(variants=18, decoder="beam")


def test_end_to_end_best_config_is_accurate(engine, labeled):
    exact = 0
    for item in labeled:
        result = engine.decode(item.path)
        assert len(result.text) == 5  # restricted decoder honors the length
        assert set(result.text) <= set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        exact += result.text == item.label
    # The reference reports ~85% exact; on 8 samples expect a solid majority.
    assert exact >= 5, f"only {exact}/8 exact"

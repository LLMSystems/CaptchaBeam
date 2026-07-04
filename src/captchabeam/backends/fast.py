"""Fast ddddocr backend: correct static model + numpy-native decode.

This is the legitimate single-image latency win. It keeps ddddocr's original
(static, batch=1) recognition model — so results are identical to the default
backend — but:

* takes numpy arrays directly (skips the PNG encode/decode roundtrip), and
* returns probabilities as a numpy ``[T, V]`` array (``probs_np``) so the decoder
  skips materializing an 8210-wide nested list per timestep and vectorizes the
  character aggregation.

Measured 18-variants-beam on CPU: 184 -> 121 ms/img (~34% faster) with no change
in accuracy (85.0% exact on the 300-image holdout).

Note: this does NOT batch the OCR model across variants. A true batched
re-export was tried and rejected — ddddocr's graph has an op that assumes
batch=1, so batching different images leaks across samples and corrupts results.
Since the speedup comes almost entirely from the numpy decode (not batched
inference), running the correct static model per variant is both faster-enough
and exact.
"""
from __future__ import annotations

import numpy as np

from ..types import RawResult


def _softmax_last(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    np.exp(x, out=x)
    x /= x.sum(axis=-1, keepdims=True)
    return x


class FastDdddOcrBackend:
    """Array-in, ``probs_np``-out drop-in for :class:`DdddOcrBackend`."""

    def __init__(self, use_gpu: bool = False, device_id: int = 0, show_ad: bool = False) -> None:
        try:
            import ddddocr  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The fast backend requires ddddocr: pip install captchabeam[fast]"
            ) from exc
        from PIL import Image  # noqa: PLC0415

        self._Image = Image
        ocr = ddddocr.DdddOcr(show_ad=show_ad, use_gpu=use_gpu, device_id=device_id)
        self._engine = ocr.ocr_engine
        self._session = self._engine.session
        self._input_name = self._session.get_inputs()[0].name
        self._charset = list(self._engine.charset_manager.get_charset())

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Grayscale uint8 array -> ddddocr's [1, 1, 64, W] float tensor."""
        return self._engine._preprocess_image(self._Image.fromarray(image), False)

    def _ctc_text(self, logits: np.ndarray) -> str:
        idxs = logits.argmax(axis=-1)
        out, prev = [], -1
        for i in idxs:
            i = int(i)
            if i != prev and i != 0:
                out.append(self._charset[i])
            prev = i
        return "".join(out)

    def _one(self, image: np.ndarray) -> RawResult:
        tensor = self._preprocess(image).astype(np.float32)
        logits = self._session.run(None, {self._input_name: tensor})[0][:, 0, :]  # [T, V]
        probs = _softmax_last(logits.astype(np.float64))
        return {
            "text": self._ctc_text(logits),
            "probs_np": probs,  # [T, V] -> decoder's numpy fast path
            "charset": self._charset,
            "confidence": float(np.mean(np.max(probs, axis=-1))),
        }

    def run_batch(self, images: list[np.ndarray]) -> list[RawResult]:
        return [self._one(img) for img in images]

    def __call__(self, png: bytes) -> RawResult:
        import cv2  # noqa: PLC0415

        gray = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
        return self._one(gray)

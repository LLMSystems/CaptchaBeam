"""Batched ddddocr backend: score all variants of an image in one session.run.

This is the throughput win for single-image decoding. Instead of N sequential
OCR calls (one per preprocessing variant), it stacks all variants into a single
padded batch and runs the dynamic-batch model once — turning N tiny inferences
into one, which is where a GPU finally pays off.

Preprocessing reuses ddddocr's own ``_preprocess_image`` (so it is byte-identical
to the single-call path, and it takes numpy arrays directly, skipping the
PNG encode/decode roundtrip). Only inference (single -> batched) and the softmax
postprocessing are reimplemented. Per-image outputs are trimmed to their true
timestep count ``ceil(width/8)`` so each RawResult matches the per-image result.
"""
from __future__ import annotations

import numpy as np

from ..types import RawResult
from ._dynamic_model import get_dynamic_model_path


def _softmax_last(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    np.exp(x, out=x)
    x /= x.sum(axis=-1, keepdims=True)
    return x


class BatchedDdddOcrBackend:
    """Batched drop-in for :class:`DdddOcrBackend` exposing ``run_batch``."""

    def __init__(self, use_gpu: bool = False, device_id: int = 0, show_ad: bool = False) -> None:
        try:
            import ddddocr  # noqa: PLC0415
            import onnxruntime as ort  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "Batched backend needs ddddocr + onnxruntime: pip install captchabeam[batch]"
            ) from exc
        from PIL import Image  # noqa: PLC0415

        self._Image = Image
        # ddddocr instance is used only for its preprocessing and charset.
        ocr = ddddocr.DdddOcr(show_ad=show_ad)
        self._engine = ocr.ocr_engine
        self._charset = list(self._engine.charset_manager.get_charset())

        if use_gpu:
            providers = [("CUDAExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(get_dynamic_model_path()), providers=providers)
        self._input_name = self._session.get_inputs()[0].name

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Grayscale uint8 array -> ddddocr's [1, 1, 64, W] float tensor."""
        pil = self._Image.fromarray(image)
        return self._engine._preprocess_image(pil, False)

    def _ctc_text(self, logits: np.ndarray) -> str:
        """Greedy CTC decode (argmax + collapse) for RawResult completeness."""
        idxs = logits.argmax(axis=-1)
        out, prev = [], -1
        for i in idxs:
            i = int(i)
            if i != prev and i != 0:
                out.append(self._charset[i])
            prev = i
        return "".join(out)

    def run_batch(self, images: list[np.ndarray]) -> list[RawResult]:
        if not images:
            return []
        tensors = [self._preprocess(img) for img in images]

        # Group by width and batch each group with no padding. CNN/RNN batch
        # elements are independent, so a same-width batch is identical to
        # per-image inference; padding to a common width, by contrast, leaks
        # into the last valid timestep via the receptive field and breaks parity.
        by_width: dict[int, list[int]] = {}
        for idx, t in enumerate(tensors):
            by_width.setdefault(t.shape[3], []).append(idx)

        results: list[RawResult | None] = [None] * len(images)
        for width, idxs in by_width.items():
            batch = np.concatenate([tensors[i] for i in idxs], axis=0)  # [g,1,64,W]
            logits = self._session.run(None, {self._input_name: batch})[0]  # [T,g,V]
            for j, idx in enumerate(idxs):
                clip = logits[:, j, :]  # [T, V]
                probs = _softmax_last(clip.astype(np.float64))
                # Hand the numpy matrix straight to the decoder (probs_np) instead
                # of materializing an 8210-wide nested list per timestep.
                results[idx] = {
                    "text": self._ctc_text(clip),
                    "probs_np": probs,  # [T, V]
                    "charset": self._charset,
                    "confidence": float(np.mean(np.max(probs, axis=-1))),
                }
        return results  # type: ignore[return-value]

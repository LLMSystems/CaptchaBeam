"""High-level orchestrator tying preprocessing, OCR, decoding and selection.

Reproduces the reference project's ``_ocr_captcha`` flow with the browser and
retry/fallback machinery stripped out: it takes an image and returns a decode
result.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

from .backends.base import OcrBackend
from .config import DecodeConfig
from .decode.beam import RestrictedCTCBeamDecoder
from .decode.greedy import GreedyDecoder
from .preprocess import VariantPipeline, select_variants, to_gray
from .select import AgreementSelector, ConfidenceSelector, Selector
from .types import Candidate, DecodeResult

ImageInput = bytes | str | Path


class CaptchaBeam:
    """End-to-end captcha decoder.

    Defaults reproduce the reference project's best configuration: 18
    preprocessing variants, restricted CTC beam decoding, agreement selection.
    """

    def __init__(
        self,
        backend: OcrBackend | None = None,
        variants: int | Sequence[VariantPipeline] = 18,
        decoder: Literal["beam", "native"] = "beam",
        decode_config: DecodeConfig | None = None,
        selector: Selector | None = None,
    ) -> None:
        self.config = decode_config or DecodeConfig()
        self._backend_override = backend
        self._backend: OcrBackend | None = backend
        self.decoder_kind = decoder

        if isinstance(variants, int):
            self.variants: list[VariantPipeline] = list(select_variants(variants))
        else:
            self.variants = list(variants)
            if not self.variants:
                raise ValueError("variants must not be empty")

        if decoder == "beam":
            self._decoder = RestrictedCTCBeamDecoder(self.config)
        elif decoder == "native":
            self._decoder = GreedyDecoder(self.config)
        else:
            raise ValueError(f"Unsupported decoder: {decoder!r}")

        target_length = self.config.length
        if selector is not None:
            self.selector = selector
        elif decoder == "beam":
            self.selector = AgreementSelector(target_length)
        else:
            self.selector = ConfidenceSelector(target_length)

    @property
    def backend(self) -> OcrBackend:
        """Lazily construct the default ddddocr backend on first use."""
        if self._backend is None:
            from .backends.ddddocr_backend import DdddOcrBackend  # noqa: PLC0415

            self._backend = DdddOcrBackend()
        return self._backend

    def _read_png(self, image: ImageInput) -> bytes:
        if isinstance(image, (str, Path)):
            return Path(image).read_bytes()
        return image

    def candidates(self, image: ImageInput) -> list[Candidate]:
        """Run every variant and return one decoded candidate per variant.

        Uses a single batched OCR call when the backend supports ``run_batch``
        (all variants scored in one inference); otherwise falls back to one OCR
        call per variant.
        """
        gray = to_gray(self._read_png(image))
        backend = self.backend

        if hasattr(backend, "run_batch"):
            arrays = [variant.apply(gray) for variant in self.variants]
            raws = backend.run_batch(arrays)
            return [
                Candidate(variant.name, *self._decoder.decode(raw))
                for variant, raw in zip(self.variants, raws)
            ]

        results: list[Candidate] = []
        for variant in self.variants:
            png = variant.to_png(gray)
            raw = backend(png)
            text, confidence = self._decoder.decode(raw)
            results.append(Candidate(variant.name, text, confidence))
        return results

    def decode(self, image: ImageInput) -> DecodeResult:
        candidates = self.candidates(image)
        chosen = self.selector.select(candidates)
        length_ok = (
            self.config.length is None or len(chosen.text) == self.config.length
        )
        return DecodeResult(
            text=chosen.text,
            confidence=chosen.confidence,
            variant_name=chosen.variant_name,
            length_ok=length_ok,
            candidates=tuple(candidates),
        )

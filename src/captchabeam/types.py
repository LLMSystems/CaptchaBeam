"""Shared data types for CaptchaBeam.

The whole pipeline speaks in these types so preprocessing, OCR backends,
decoders and selectors can be swapped independently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

# A per-timestep probability matrix as returned by CTC OCR models.
# Shape is [T][1][V]: T timesteps, an inner singleton dim (ddddocr convention),
# and V vocabulary entries aligned to ``charset``.
ProbMatrix = list


class RawResult(TypedDict, total=False):
    """The raw output an :class:`OcrBackend` must produce.

    Mirrors ``ddddocr.classification(..., probability=True)`` so existing
    models drop in unchanged, but any CTC model that can emit a probability
    matrix plus its charset satisfies this contract.
    """

    text: str
    confidence: float
    probabilities: ProbMatrix  # [T][1][V]
    # Optional fast path: a numpy [T, V] array of probabilities. When present the
    # beam decoder consumes it directly, skipping nested-list materialization.
    probs_np: object
    charset: list[str]  # index -> character; index of the blank is ``blank_index``


@dataclass(frozen=True, slots=True)
class Candidate:
    """One decoded hypothesis, produced per preprocessing variant."""

    variant_name: str
    text: str
    confidence: float


@dataclass(frozen=True, slots=True)
class DecodeResult:
    """Final decode output returned by :class:`captchabeam.CaptchaBeam`."""

    text: str
    confidence: float
    variant_name: str = ""
    length_ok: bool = True
    candidates: tuple[Candidate, ...] = field(default_factory=tuple)

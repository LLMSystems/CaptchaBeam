"""Numerically stable log-space helpers for CTC beam search."""
from __future__ import annotations

import math


def logadd(left: float, right: float) -> float:
    """Return ``log(exp(left) + exp(right))`` without underflow.

    Ported verbatim from the reference project's ``_logadd``.
    """
    if left == -math.inf:
        return right
    if right == -math.inf:
        return left
    if left < right:
        left, right = right, left
    return left + math.log1p(math.exp(right - left))

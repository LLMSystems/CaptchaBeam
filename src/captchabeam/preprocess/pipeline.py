"""A named, ordered chain of preprocessing ops."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from . import ops

Op = Callable  # any op callable: image -> image


@dataclass(frozen=True, slots=True)
class VariantPipeline:
    """A named sequence of ops producing one preprocessing variant.

    Example::

        VariantPipeline("s2_pad2", ops.resize(2), ops.otsu(), ops.pad(2))
    """

    name: str
    steps: tuple[Op, ...]

    def __init__(self, name: str, *steps: Op) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "steps", tuple(steps))

    def apply(self, gray):
        """Apply every op in order to a grayscale image."""
        img = gray
        for step in self.steps:
            img = step(img)
        return img

    def to_png(self, gray) -> bytes:
        return ops.encode_png(self.apply(gray))

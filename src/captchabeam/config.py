"""Configuration objects for CaptchaBeam.

Everything the reference project hard-coded (charset, length, beam size, top-k)
lives here as tunable settings. Defaults reproduce the reference project's best
configuration: 5-character ``A-Z/0-9`` captchas.
"""
from __future__ import annotations

from dataclasses import dataclass

DEFAULT_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True, slots=True)
class DecodeConfig:
    """Constraints and knobs for the restricted CTC beam decoder.

    Attributes:
        charset: Allowed output characters. Matching against the backend charset
            is case-insensitive, so both ``v`` and ``V`` map to ``V``.
        length: Exact target length. ``None`` disables length constraints.
        length_range: Alternative ``(min, max)`` length window. Ignored when
            ``length`` is set.
        beam_size: Number of prefixes kept after each timestep.
        top_chars: Per timestep, only the highest-probability characters are
            expanded (prunes the branching factor).
        blank_index: Index of the CTC blank symbol within the backend charset.
    """

    charset: str = DEFAULT_CHARSET
    length: int | None = 5
    length_range: tuple[int, int] | None = None
    beam_size: int = 10
    top_chars: int = 8
    blank_index: int = 0

    def length_bounds(self) -> tuple[int, float]:
        """Return ``(min_len, max_len)`` used for extension caps and ranking."""
        if self.length is not None:
            return self.length, self.length
        if self.length_range is not None:
            return self.length_range[0], self.length_range[1]
        return 0, float("inf")

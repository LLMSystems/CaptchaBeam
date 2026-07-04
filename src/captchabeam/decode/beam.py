"""Restricted CTC beam-search decoder.

This is CaptchaBeam's core asset. It is a faithful port of the reference
project's ``_ctc_beam_decode`` (which lifted 18-variant accuracy from 78.3% to
85.0% exact on 300 held-out samples), generalized so the charset, length and
beam parameters are configurable via :class:`DecodeConfig` instead of hard-coded
to 5-character ``A-Z/0-9``.

The decoder restricts every timestep to the configured charset and (optionally)
enforces a target length, so it recovers hypotheses a greedy decoder drops. The
canonical example: greedy yields ``XRR3`` (dropped a character) while the beam
keeps enough paths to recover the correct ``XTRR3``.
"""
from __future__ import annotations

import math
from collections import defaultdict

from ..config import DecodeConfig
from ..types import RawResult
from .logmath import logadd


class RestrictedCTCBeamDecoder:
    def __init__(self, config: DecodeConfig | None = None) -> None:
        self.config = config or DecodeConfig()
        self._allowed_upper = {c.upper() for c in self.config.charset}
        # Cache of backend charset -> {allowed_char: [backend indices]}.
        self._index_cache: dict[int, dict[str, list[int]]] | None = None
        self._cached_charset_id: int | None = None

    def _allowed_indices(self, charset: list[str]) -> dict[str, list[int]]:
        """Map each allowed character to the backend charset indices that emit it.

        Case-insensitive: a backend that lists both ``v`` and ``V`` contributes
        both indices to ``V``. Cached per charset identity, mirroring the
        reference project's per-instance cache.
        """
        if self._cached_charset_id == id(charset) and self._index_cache is not None:
            return self._index_cache  # type: ignore[return-value]

        indices: dict[str, list[int]] = {char: [] for char in self._allowed_upper}
        for index, char in enumerate(charset):
            upper = char.upper() if char else char
            if upper in indices:
                indices[upper].append(index)
        self._index_cache = indices  # type: ignore[assignment]
        self._cached_charset_id = id(charset)
        return indices

    def decode(self, result: RawResult) -> tuple[str, float]:
        probabilities = result.get("probabilities") or []
        charset = result.get("charset") or []
        if not probabilities or not charset:
            # Fall back to the backend's own text if there is no matrix to decode.
            return (
                (result.get("text") or "").strip().upper(),
                float(result.get("confidence") or 0.0),
            )

        cfg = self.config
        min_len, max_len = cfg.length_bounds()
        char_indices = self._allowed_indices(charset)
        blank_index = cfg.blank_index

        # Each beam maps a prefix -> (log prob ending in blank, ending in non-blank).
        beams: dict[tuple[str, ...], tuple[float, float]] = {(): (0.0, -math.inf)}

        for step in probabilities:
            row = step[0]
            blank_logp = math.log(max(float(row[blank_index]), 1e-30))
            char_logps: list[tuple[str, float]] = []
            for char, idxs in char_indices.items():
                prob = sum(float(row[index]) for index in idxs)
                char_logps.append((char, math.log(max(prob, 1e-30))))
            char_logps.sort(key=lambda item: item[1], reverse=True)
            char_logps = char_logps[: cfg.top_chars]

            next_beams: dict[tuple[str, ...], tuple[float, float]] = defaultdict(
                lambda: (-math.inf, -math.inf)
            )
            for prefix, (prob_blank, prob_nonblank) in beams.items():
                # Case 1: emit a blank -> prefix unchanged, now ends in blank.
                next_blank, next_nonblank = next_beams[prefix]
                next_blank = logadd(next_blank, prob_blank + blank_logp)
                next_blank = logadd(next_blank, prob_nonblank + blank_logp)
                next_beams[prefix] = (next_blank, next_nonblank)

                for char, char_logp in char_logps:
                    # Once at the max length, only allow repeating the last char
                    # (a repeat collapses to the same string under CTC).
                    if len(prefix) >= max_len and (not prefix or prefix[-1] != char):
                        continue

                    if prefix and prefix[-1] == char:
                        # Repeat: extend the current run (came from a non-blank).
                        same_blank, same_nonblank = next_beams[prefix]
                        same_nonblank = logadd(same_nonblank, prob_nonblank + char_logp)
                        next_beams[prefix] = (same_blank, same_nonblank)

                        # Or start a genuine second char after a blank separator.
                        if len(prefix) < max_len:
                            extended = prefix + (char,)
                            ext_blank, ext_nonblank = next_beams[extended]
                            ext_nonblank = logadd(ext_nonblank, prob_blank + char_logp)
                            next_beams[extended] = (ext_blank, ext_nonblank)
                    else:
                        extended = prefix + (char,)
                        ext_blank, ext_nonblank = next_beams[extended]
                        ext_nonblank = logadd(ext_nonblank, prob_blank + char_logp)
                        ext_nonblank = logadd(ext_nonblank, prob_nonblank + char_logp)
                        next_beams[extended] = (ext_blank, ext_nonblank)

            ranked = sorted(
                next_beams.items(),
                key=lambda item: logadd(item[1][0], item[1][1]),
                reverse=True,
            )
            beams = dict(ranked[: cfg.beam_size])

        prefix, (prob_blank, prob_nonblank) = max(
            beams.items(),
            key=lambda item: (
                min_len <= len(item[0]) <= max_len,
                logadd(item[1][0], item[1][1]),
            ),
        )
        log_probability = logadd(prob_blank, prob_nonblank)
        # Normalize by sequence length so scores from variants of different
        # timestep counts can be summed as a confidence-like agreement score.
        confidence = math.exp(log_probability / max(1, len(probabilities)))
        return "".join(prefix), confidence

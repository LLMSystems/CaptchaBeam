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

Two front-ends feed the same sequential prefix-beam expansion (``_run_beam``):

* ``decode`` consumes ddddocr's nested-list ``probabilities`` and aggregates each
  allowed character's mass with a targeted Python sum over the few backend
  indices that map to it (fast because it never touches the full 8210-wide row).
* ``decode`` also accepts a numpy ``probs_np`` (``[T, V]``) supplied by the
  batched backend and aggregates via one ``[T, V] @ [V, C]`` matmul, avoiding
  the nested-list materialization entirely.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from ..config import DecodeConfig
from ..types import RawResult
from .logmath import logadd

_FLOOR = 1e-30


class RestrictedCTCBeamDecoder:
    def __init__(self, config: DecodeConfig | None = None) -> None:
        self.config = config or DecodeConfig()
        self._allowed_upper = {c.upper() for c in self.config.charset}
        # Deterministic column order for the numpy path's charmap.
        seen: dict[str, None] = {}
        for c in self.config.charset:
            seen.setdefault(c.upper(), None)
        self._char_list: list[str] = list(seen)
        # Caches keyed by backend charset identity.
        self._index_cache: dict[str, list[int]] | None = None
        self._charmap: np.ndarray | None = None
        self._cached_charset_id: int | None = None

    def _allowed_indices(self, charset: list[str]) -> dict[str, list[int]]:
        """Map each allowed character to the backend charset indices that emit it.

        Case-insensitive: a backend that lists both ``v`` and ``V`` contributes
        both indices to ``V``. Cached per charset identity.
        """
        if self._cached_charset_id == id(charset) and self._index_cache is not None:
            return self._index_cache

        indices: dict[str, list[int]] = {char: [] for char in self._allowed_upper}
        for index, char in enumerate(charset):
            upper = char.upper() if char else char
            if upper in indices:
                indices[upper].append(index)
        self._index_cache = indices
        self._cached_charset_id = id(charset)
        return indices

    def _get_charmap(self, charset: list[str]) -> np.ndarray:
        """``[V, C]`` matrix: ``charmap[v, c] == 1`` when backend index ``v`` maps
        to allowed character ``c``. Cached per charset identity."""
        if self._charmap is not None and self._cached_charset_id == id(charset):
            return self._charmap
        # Reuse/refresh the index cache for the same charset.
        char_indices = self._allowed_indices(charset)
        col_of = {char: c for c, char in enumerate(self._char_list)}
        charmap = np.zeros((len(charset), len(self._char_list)), dtype=np.float64)
        for char, idxs in char_indices.items():
            col = col_of[char]
            for v in idxs:
                charmap[v, col] = 1.0
        self._charmap = charmap
        return charmap

    def decode(self, result: RawResult) -> tuple[str, float]:
        charset = result.get("charset") or []

        probs_np = result.get("probs_np")
        if probs_np is not None and len(charset):
            return self._decode_numpy(np.asarray(probs_np, dtype=np.float64), charset)

        probabilities = result.get("probabilities") or []
        if not probabilities or not charset:
            # Fall back to the backend's own text if there is no matrix to decode.
            return (
                (result.get("text") or "").strip().upper(),
                float(result.get("confidence") or 0.0),
            )
        return self._decode_list(probabilities, charset)

    def _decode_list(self, probabilities, charset) -> tuple[str, float]:
        char_indices = self._allowed_indices(charset)
        blank_index = self.config.blank_index
        top_chars = self.config.top_chars

        step_blank: list[float] = []
        step_chars: list[list[tuple[str, float]]] = []
        for step in probabilities:
            row = step[0]
            step_blank.append(math.log(max(float(row[blank_index]), _FLOOR)))
            char_logps = [
                (char, math.log(max(sum(float(row[i]) for i in idxs), _FLOOR)))
                for char, idxs in char_indices.items()
            ]
            char_logps.sort(key=lambda item: item[1], reverse=True)
            step_chars.append(char_logps[:top_chars])
        return self._run_beam(step_blank, step_chars, len(probabilities))

    def _decode_numpy(self, probs: np.ndarray, charset) -> tuple[str, float]:
        cfg = self.config
        if probs.ndim == 3:  # tolerate [T, 1, V]
            probs = probs[:, 0, :]
        num_steps = probs.shape[0]

        blank_logp = np.log(np.maximum(probs[:, cfg.blank_index], _FLOOR))
        allowed_logp = np.log(np.maximum(probs @ self._get_charmap(charset), _FLOOR))

        top = min(cfg.top_chars, len(self._char_list))
        order = np.argsort(-allowed_logp, axis=1)[:, :top]
        step_chars = [
            [(self._char_list[j], float(allowed_logp[t, j])) for j in order[t]]
            for t in range(num_steps)
        ]
        return self._run_beam(blank_logp.tolist(), step_chars, num_steps)

    def _run_beam(
        self,
        step_blank: list[float],
        step_chars: list[list[tuple[str, float]]],
        num_steps: int,
    ) -> tuple[str, float]:
        cfg = self.config
        min_len, max_len = cfg.length_bounds()

        # Each beam maps a prefix -> (log prob ending in blank, ending in non-blank).
        beams: dict[tuple[str, ...], tuple[float, float]] = {(): (0.0, -math.inf)}

        for t in range(num_steps):
            blank_logp = step_blank[t]
            char_logps = step_chars[t]

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
        confidence = math.exp(log_probability / max(1, num_steps))
        return "".join(prefix), confidence

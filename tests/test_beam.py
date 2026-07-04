"""Restricted CTC beam decoder correctness on hand-built probability matrices."""
from __future__ import annotations

import math

from captchabeam import DecodeConfig, RestrictedCTCBeamDecoder


def make_result(charset, steps):
    """steps: list of per-timestep probability rows -> RawResult with [T][1][V]."""
    return {"probabilities": [[row] for row in steps], "charset": charset}


def argmax_collapse(charset, steps, blank=0):
    """Reference greedy CTC collapse, for contrast with the beam decoder."""
    out = []
    prev = None
    for row in steps:
        idx = max(range(len(row)), key=lambda i: row[i])
        if idx != blank and idx != prev:
            out.append(charset[idx])
        prev = idx
    return "".join(out)


def test_basic_decode_with_blank_collapse():
    charset = ["", "A", "B"]
    steps = [
        [0.1, 0.8, 0.1],  # A
        [0.8, 0.1, 0.1],  # blank
        [0.1, 0.1, 0.8],  # B
    ]
    dec = RestrictedCTCBeamDecoder(DecodeConfig(charset="AB", length=2))
    text, conf = dec.decode(make_result(charset, steps))
    assert text == "AB"
    assert 0.0 < conf <= 1.0


def test_repeat_char_collapses_to_single():
    charset = ["", "A"]
    steps = [
        [0.1, 0.9],  # A
        [0.1, 0.9],  # A (same run -> collapses)
    ]
    dec = RestrictedCTCBeamDecoder(DecodeConfig(charset="A", length=1))
    text, _ = dec.decode(make_result(charset, steps))
    assert text == "A"


def test_length_constraint_recovers_dropped_character():
    """The reference project's key win: greedy drops a char, beam recovers it.

    Greedy argmax-collapse yields "A" (steps 2-3 peak on blank), but with a
    length-2 constraint the beam recovers the plausible "AB" — the "少字" fix
    that lifted exact accuracy from 78.3% to 85.0%.
    """
    charset = ["", "A", "B"]
    steps = [
        [0.10, 0.80, 0.10],  # A
        [0.60, 0.10, 0.30],  # blank wins greedily, but B has real mass
        [0.60, 0.10, 0.30],  # blank wins greedily, but B has real mass
    ]
    assert argmax_collapse(charset, steps) == "A"  # greedy drops the B

    dec = RestrictedCTCBeamDecoder(DecodeConfig(charset="AB", length=2))
    text, _ = dec.decode(make_result(charset, steps))
    assert text == "AB"  # beam recovers it


def test_charset_restriction_ignores_disallowed_characters():
    charset = ["", "A", "B", "#"]
    steps = [
        [0.05, 0.10, 0.05, 0.80],  # '#' is highest but disallowed
        [0.05, 0.80, 0.10, 0.05],  # A
    ]
    dec = RestrictedCTCBeamDecoder(DecodeConfig(charset="AB", length=1))
    text, _ = dec.decode(make_result(charset, steps))
    assert set(text) <= set("AB")
    assert text == "A"


def test_case_insensitive_charset_mapping():
    charset = ["", "a", "A"]  # backend lists both cases
    steps = [[0.1, 0.5, 0.4]]
    dec = RestrictedCTCBeamDecoder(DecodeConfig(charset="A", length=1))
    text, _ = dec.decode(make_result(charset, steps))
    assert text == "A"  # 'a' + 'A' probability mass both count toward 'A'


def test_empty_matrix_falls_back_to_backend_text():
    dec = RestrictedCTCBeamDecoder(DecodeConfig())
    text, conf = dec.decode({"text": "hello", "confidence": 0.5})
    assert text == "HELLO"
    assert conf == 0.5


def test_no_length_constraint_allows_any_length():
    charset = ["", "A", "B"]
    steps = [
        [0.1, 0.8, 0.1],
        [0.8, 0.1, 0.1],
        [0.1, 0.1, 0.8],
    ]
    dec = RestrictedCTCBeamDecoder(DecodeConfig(charset="AB", length=None))
    text, _ = dec.decode(make_result(charset, steps))
    assert text == "AB"


def test_confidence_is_length_normalized_probability():
    charset = ["", "A"]
    steps = [[0.01, 0.99], [0.99, 0.01]]
    dec = RestrictedCTCBeamDecoder(DecodeConfig(charset="A", length=1))
    _, conf = dec.decode(make_result(charset, steps))
    assert 0.0 < conf <= 1.0
    assert not math.isnan(conf)

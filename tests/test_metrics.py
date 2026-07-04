"""Metrics computation tests."""
from __future__ import annotations

from captchabeam.eval import score


def test_perfect_predictions():
    m = score([("ABCDE", "ABCDE"), ("12345", "12345")], target_length=5)
    assert m.exact == 2
    assert m.exact_rate == 1.0
    assert m.char_rate == 1.0
    assert m.length_ok == 2


def test_char_and_length_partial_credit():
    m = score([("ABCDE", "ABCDX"), ("ABCD", "ABCDE")], target_length=5)
    assert m.exact == 0
    # first: 4/5 chars; second: 4 chars overlap out of max(4,5)=5
    assert m.char_ok == 8
    assert m.char_total == 10
    assert m.char_rate == 0.8
    assert m.length_ok == 1  # only the first is length 5


def test_length_defaults_to_truth_length_when_unset():
    m = score([("AB", "ABC")], target_length=None)
    assert m.length_ok == 0  # len 2 != len(truth)=3

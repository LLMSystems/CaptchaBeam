"""Selector strategy tests."""
from __future__ import annotations

from captchabeam import AgreementSelector, Candidate, ConfidenceSelector


def test_confidence_selector_prefers_correct_length_over_confidence():
    cands = [
        Candidate("v1", "ABCD", 0.99),  # too short but very confident
        Candidate("v2", "ABCDE", 0.60),  # correct length
    ]
    chosen = ConfidenceSelector(target_length=5).select(cands)
    assert chosen.text == "ABCDE"


def test_agreement_beats_a_lone_high_confidence_short_answer():
    """The reference project's finding: agreement avoids the high-confidence
    but character-dropping trap that a pure confidence selector falls into."""
    cands = [
        Candidate("v1", "ABCD", 0.98),   # lone, short, high confidence
        Candidate("v2", "ABCDE", 0.55),  # correct length, agreed by two variants
        Candidate("v3", "ABCDE", 0.50),
    ]
    agree = AgreementSelector(target_length=5).select(cands)
    assert agree.text == "ABCDE"
    # A pure confidence selector, by contrast, would pick the short one.
    conf = ConfidenceSelector(target_length=None).select(cands)
    assert conf.text == "ABCD"


def test_agreement_sums_confidence_within_a_group():
    cands = [
        Candidate("v1", "AAAAA", 0.40),
        Candidate("v2", "AAAAA", 0.40),  # sum 0.80 across the group
        Candidate("v3", "BBBBB", 0.70),  # single, higher individual
    ]
    chosen = AgreementSelector(target_length=5).select(cands)
    assert chosen.text == "AAAAA"


def test_selectors_reject_empty():
    import pytest

    for sel in (AgreementSelector(), ConfidenceSelector()):
        with pytest.raises(ValueError):
            sel.select([])

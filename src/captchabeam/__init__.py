"""CaptchaBeam: customizable restricted CTC beam-search captcha decoding.

Quick start::

    from captchabeam import CaptchaBeam

    cb = CaptchaBeam()                       # 18 variants + beam + agreement
    print(cb.decode("captcha.png").text)     # -> "XTRR3"
"""
from .config import DEFAULT_CHARSET, DecodeConfig
from .decode import GreedyDecoder, RestrictedCTCBeamDecoder
from .engine import CaptchaBeam
from .select import AgreementSelector, ConfidenceSelector
from .types import Candidate, DecodeResult, RawResult

__version__ = "0.1.0"

__all__ = [
    "CaptchaBeam",
    "DecodeConfig",
    "DEFAULT_CHARSET",
    "RestrictedCTCBeamDecoder",
    "GreedyDecoder",
    "AgreementSelector",
    "ConfidenceSelector",
    "Candidate",
    "DecodeResult",
    "RawResult",
]

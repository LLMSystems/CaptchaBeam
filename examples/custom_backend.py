"""Plug a custom OCR model in place of ddddocr.

Any callable that maps PNG bytes to {text, confidence, probabilities, charset}
is a valid backend. The restricted beam decoder consumes the probability matrix,
so it works with any CTC-based model, not just ddddocr.
"""
from captchabeam import CaptchaBeam
from captchabeam.types import RawResult


class MyCRNN:
    def __call__(self, png: bytes) -> RawResult:
        # Run your model here and return the CTC probability matrix + charset.
        # probabilities shape is [T][1][V], aligned to `charset` (index 0 = blank).
        raise NotImplementedError("wire up your model")


cb = CaptchaBeam(backend=MyCRNN())
# print(cb.decode("captcha.png").text)
print("Backend:", type(cb.backend).__name__ if False else "MyCRNN (lazy)")

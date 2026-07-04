"""Decode one captcha with the default best configuration."""
from pathlib import Path

from captchabeam import CaptchaBeam

image = Path(__file__).resolve().parents[1] / "data" / "captcha_holdout_100" / "captcha_0001.png"

cb = CaptchaBeam()  # 18 variants + restricted beam + agreement
result = cb.decode(image)

print("text:      ", result.text)
print("confidence:", round(result.confidence, 4))
print("variant:   ", result.variant_name)
print("length ok: ", result.length_ok)

"""Configure the decoder for a different captcha spec (charset + length)."""
from captchabeam import CaptchaBeam, DecodeConfig

# Example: a 4-character digits-only captcha with a wider beam.
cb = CaptchaBeam(
    variants=18,
    decoder="beam",
    decode_config=DecodeConfig(
        charset="0123456789",
        length=4,
        beam_size=16,
        top_chars=10,
    ),
)

# print(cb.decode("my_digit_captcha.png").text)
print("Configured:", cb.config)

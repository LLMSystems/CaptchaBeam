# Accuracy Experiments — what was tried to beat 85% exact

Goal: push 18-variants-beam past its **85.0% exact** on the 300-image holdout.
Summary: **no off-the-shelf route beat 85.0%.** The remaining errors are
correlated across models and variants (ambiguous glyphs), so they resist both
ensembling and smarter selection. Documented here so the dead ends aren't
re-tread.

All numbers are on the same 300-image holdout (`captcha_holdout_100` +
`_extra_100` + `_extra2_100`).

## 1. ddddocr second model (beta / `common.onnx`) — no gain

ddddocr ships two recognition models: `common_old.onnx` (default) and
`common.onnx` (`beta=True`). Scoring all 18 variants through each and running
agreement over the combined 36 candidates:

| Models | Exact | Char | Len-OK |
|--------|------:|-----:|-------:|
| old (default) | **85.0%** | 96.3% | 99.7% |
| beta | 81.7% | 95.7% | 99.3% |
| old + beta ensemble | 85.0% | 96.6% | 100.0% |

The ensemble nudges char/len but **not exact** — the two models share an
architecture and make correlated errors. (An early "+1pt" was a measurement
artifact from colliding filenames across the three holdout dirs; on the true 300
there is no exact gain.)

## 2. Smarter selection — negligible

The true answer is in the candidate pool ~92% (old) / ~95% (old+beta) of the
time, but selection only reaches 85%. That gap is **not exploitable**: when the
correct read is a minority it carries no distinguishing confidence signal. Tried
per-position plurality voting (weighted and unweighted) and a vote/agreement
back-off:

| Strategy | Exact (old) |
|----------|------------:|
| string agreement (default) | 85.0% |
| per-position weighted vote | 85.3% |
| per-position majority vote | 85.3% |
| vote with agreement back-off | 85.3% |

+0.3pt is within noise; not worth the added complexity.

## 3. easyocr (different architecture) — unusable here

easyocr (CRNN scene-text model, CPU) as a potential ensemble voter:

| Input | Exact (50 imgs) |
|-------|----------------:|
| raw image | 10% |
| Otsu variant | 12% |
| 3× upscaled Otsu | 12% |

It drops and confuses characters (`VH2BC`→`""`, `XTRR3`→`XRR3`, `7HTA2`→`ZHTAZ`).
A ~10% model cannot lift an 85% ensemble; general scene-text OCR is the wrong
tool for a distorted, interference-lined captcha font.

## Conclusion

**85% is the practical ceiling with off-the-shelf OCR on this captcha.** The
errors are genuinely ambiguous glyphs (`0/Q`, `8/D`, `5/S`, …), each appearing
once — no systematic confusion to fix with preprocessing, and length is already
solved (`len_ok` ≈ 100%).

The only credible path to higher accuracy is a **model specialized to this
captcha's font** — fine-tuning a CRNN on labeled samples of *this* site's
captcha. That needs a training pipeline and more labeled data, risks overfitting,
and is out of scope for an off-the-shelf library. Left as future work.

# CaptchaBeam — Benchmarks & Methodology

All numbers are produced by `scripts/benchmark.py` on the migrated `data/`
(5-character `A-Z/0-9` captchas), validated on **three independent 100-image
holdout sets (300 total)**, human-labeled. The library reproduces the reference
project's original numbers exactly.

> These absolute numbers are for one specific captcha. The *method* transfers to
> other captchas; the *numbers* must be re-measured on your own labeled holdout
> via `captchabeam eval`.

Metrics: `exact` (whole string correct), `char` (character-level accuracy),
`len_ok` (output length matches spec).

---

## Optimization journey

![Optimization journey](captcha_ocr_optimization_journey.svg)

| Stage | Method | Exact | Char | Len-OK |
|-------|--------|------:|-----:|-------:|
| 1 | Raw image → ddddocr (baseline) | 62.0% | — | — |
| 2 | Grayscale + Otsu | 72.3% | 88.2% | 81.7% |
| 3 | 6 variants + native | 76.3% | 90.9% | 88.0% |
| 3 | 18 variants + native | 78.3% | 92.9% | 92.3% |
| 5 | **18 variants + restricted beam + agreement** | **85.0%** | **96.3%** | **99.7%** |

- **Otsu** ([before/after](captcha_otsu_before_after.svg)) is the single most
  effective preprocessing step — it suppresses background noise so character
  contours are cleaner.
- **Multiple variants** ([18-variant grid](captcha_18_variants_grid.svg)) help
  because different captchas respond differently to scaling, interpolation and
  morphology; each variant OCRs a differently-processed copy.
- **Restricted CTC beam** ([diagram](restricted_ctc_beam_decoder.png)) is the
  biggest single jump (78.3% → 85.0%), recovering dropped characters that greedy
  decoding loses (e.g. `XRR3` → `XTRR3`).

---

## Inference latency (CPU vs GPU)

End-to-end per image: preprocessing + OCR over every variant + decode +
selection. Hardware: Intel i7-12700 (CPU) / NVIDIA RTX 3060 Ti (GPU),
ddddocr 1.6.1, onnxruntime-gpu 1.22 (same build for both columns, so the only
difference is CPU vs CUDA execution provider).

| Tier | Exact | CPU ms/img | GPU ms/img | GPU vs CPU |
|------|------:|-----------:|-----------:|-----------:|
| otsu native (1 variant) | 72.3% | **8.2** | 12.7 | 1.5× slower |
| 6 variants native | 76.3% | 45.6 | 98.3 | 2.2× slower |
| 18 variants beam (default) | 85.0% | 184.1 | 347.3 | 1.9× slower |

**GPU is slower here, and that's expected.** The model is tiny (~23 CTC
timesteps) and CaptchaBeam calls it once per variant on single small images,
sequentially. Per-call host↔device transfer and kernel launch overhead outweighs
the compute saved; preprocessing and the beam-search loop stay on CPU regardless.
**Run this workload on CPU.**

---

## Fast backend

`FastDdddOcrBackend` keeps ddddocr's original (static) recognition model but
(1) takes numpy arrays directly (skips the PNG encode/decode roundtrip) and
(2) returns a numpy `[T, V]` probability matrix so the decoder skips
materializing an 8210-wide nested list per timestep and vectorizes the character
aggregation.

| Mode | CPU ms/img | Exact |
|------|-----------:|------:|
| per-variant (default) | 184.1 | 85.0% |
| **fast** | **120.9** | **85.0%** |

Same static model → **char-for-char identical output**, ~34% faster. The gain is
entirely from dropping the nested-list conversion and vectorizing the decode; the
OCR compute is unchanged.

### Why not batch the OCR model?

Batching all variants into one `session.run` (via a dynamic-batch re-export of
the model) was tried and **rejected**. Isolation experiments showed:

| Comparison | max |logit| diff |
|-----------|------------------:|
| static vs dynamic re-export (batch=1) | 0.0 |
| same tensor batched ×N vs single | 0.0 |
| two *different* same-width images batched | 1.26 |

ddddocr's graph has an operation that assumes batch=1, so batching *different*
images leaks information across samples and corrupts results, costing ~1 point of
accuracy. It is also barely faster than the fast backend on CPU (the bottleneck
was never the OCR compute). The fast backend delivers the speedup correctly.

---

## Retry success rate

Captchas usually allow re-rolling. With independent single-attempt success `p`,
cumulative success over `n` tries is `1 − (1 − p)^n`:

| Attempts | otsu native (72.3%) | 6 native (76.3%) | 18 beam (85.0%) |
|---------:|--------------------:|-----------------:|----------------:|
| 1 | 72.30% | 76.30% | 85.00% |
| 3 | 97.87% | 98.67% | 99.66% |
| 5 | 99.84% | 99.93% | 99.99% |

So even an imperfect single-shot recognizer becomes highly reliable in practice
with a few retries.

---

## Datasets

`data/` holds the migrated, human-labeled holdout sets (each with `labels.csv`):

| Dataset | Count | Purpose |
|---------|------:|---------|
| `captcha_samples` | 100 | Early exploration |
| `captcha_holdout_100` | 100 | Holdout #1 |
| `captcha_holdout_extra_100` | 100 | Holdout #2 |
| `captcha_holdout_extra2_100` | 100 | Holdout #3 (pruning generalization check) |

Reproduce any table above with `scripts/benchmark.py [--fast] [--gpu]` or
`captchabeam eval --data data/...`.

---

## Pushing past 85%?

See [experiments.md](experiments.md) for what was tried to beat 85.0% exact
(a second ddddocr model, ensembling, per-position voting, easyocr) and why none
of it worked — the remaining errors are correlated ambiguous glyphs.

# CaptchaBeam

**Customizable restricted CTC beam-search decoding and OpenCV preprocessing variants for fixed-format captcha OCR.**

CaptchaBeam packages the captcha-recognition core that took a real crawler from
**62% → 85% exact accuracy** on held-out data, and generalizes it so you can
apply the same method to *your* captcha by configuring its charset and length —
`pip install`, import, decode.

The pipeline is: **multi-variant OpenCV preprocessing → OCR probability matrix →
restricted CTC beam search → agreement voting across variants.**

```python
from captchabeam import CaptchaBeam

cb = CaptchaBeam()                       # 18 variants + beam + agreement (best config)
print(cb.decode("captcha.png").text)     # -> "XTRR3"
```

---

## Why this exists

`ddddocr`'s greedy decoder drops characters on noisy captchas: it might emit
`XRR3` when the answer is `XTRR3`. If you know your captcha's fixed spec (length,
allowed characters), a **restricted CTC beam search** keeps enough candidate
paths to recover the dropped character. Combining that with **multiple OpenCV
preprocessing variants** and **agreement voting** is what lifts accuracy the
last mile.

This project extracts that logic from a Selenium crawler (where it was hard-coded
to 5-character `A-Z/0-9`) into a standalone, framework-agnostic, fully
configurable library. See [ARCHITECTURE.md](ARCHITECTURE.md) for the design and
the reference → package migration map.

---

## Install

```bash
pip install captchabeam[all]      # core + opencv + ddddocr (default backend)
```

Extras: `[cv]` (preprocessing), `[ddddocr]` (default backend), `[eval]`/`[all]`
(both). The core beam decoder itself only needs `numpy`.

---

## Usage

### Three tiers (speed vs accuracy)

```python
CaptchaBeam(variants=1,  decoder="native")   # fastest, single Otsu
CaptchaBeam(variants=6,  decoder="native")   # balanced
CaptchaBeam(variants=18, decoder="beam")     # most accurate (default)
```

### Your own captcha spec

```python
from captchabeam import CaptchaBeam, DecodeConfig

cb = CaptchaBeam(decode_config=DecodeConfig(
    charset="0123456789",   # digits only
    length=4,               # 4 characters
    beam_size=16,
))
```

### Plug your own OCR model

Any callable `png_bytes -> {text, confidence, probabilities, charset}` is a valid
backend (see `captchabeam.backends.OcrBackend`), so a custom CRNN or PaddleOCR
drops in and the beam decoder still applies:

```python
cb = CaptchaBeam(backend=MyCRNN())
```

### Custom preprocessing variants

```python
from captchabeam.preprocess import VariantPipeline, resize, otsu, pad, CUBIC

cb = CaptchaBeam(variants=[
    VariantPipeline("a", resize(3, CUBIC), otsu()),
    VariantPipeline("b", resize(2, CUBIC), otsu(), pad(2)),
])
```

### CLI

```bash
captchabeam decode captcha.png --variants 18 --decoder beam
captchabeam eval     --data data/captcha_holdout_100 --length 5
captchabeam ablation --data data/captcha_holdout_100          # leave-one-out
```

---

## Benchmark results

Every optimization stage below was validated on **three independent 100-image
holdout sets (300 total)**, human-labeled. The library reproduces the reference
project's original numbers **exactly** — the accuracy columns are the output of
`scripts/benchmark.py` on the migrated `data/` (5-character `A-Z/0-9` captchas).

### Optimization journey

![Optimization journey](docs/captcha_ocr_optimization_journey.svg)

| Stage | Method | Exact | Char | Len-OK |
|-------|--------|------:|-----:|-------:|
| 1 | Raw image → ddddocr (baseline) | 62.0% | — | — |
| 2 | Grayscale + Otsu | 72.3% | 88.2% | 81.7% |
| 3 | 6 variants + native | 76.3% | 90.9% | 88.0% |
| 3 | 18 variants + native | 78.3% | 92.9% | 92.3% |
| 5 | **18 variants + restricted beam + agreement** | **85.0%** | **96.3%** | **99.7%** |

- **Otsu** ([docs](docs/captcha_otsu_before_after.svg)) is the single most
  effective preprocessing step — it suppresses background noise so character
  contours are cleaner.
- **Multiple variants** ([18-variant grid](docs/captcha_18_variants_grid.svg))
  help because different captchas respond differently to scaling, interpolation
  and morphology; each variant OCRs a differently-processed copy.
- **Restricted CTC beam** ([diagram](docs/restricted_ctc_beam_decoder.png)) is
  the biggest single jump (78.3% → 85.0%), recovering dropped characters that
  greedy decoding loses.

### Inference latency

Measured with `scripts/benchmark.py` on the 300-image holdout set. Latency is
end-to-end per image: preprocessing + OCR over every variant + decode +
selection. Hardware: Intel i7-12700 (CPU) / NVIDIA RTX 3060 Ti (GPU),
ddddocr 1.6.1, onnxruntime-gpu 1.22 (same build for both columns, so the only
difference is CPU vs CUDA execution provider).

| Tier | Exact | CPU ms/img | GPU ms/img | GPU vs CPU |
|------|------:|-----------:|-----------:|-----------:|
| otsu native (1 variant) | 72.3% | **8.2** | 12.7 | 1.5× slower |
| 6 variants native | 76.3% | 45.6 | 98.3 | 2.2× slower |
| 18 variants beam (default) | 85.0% | 184.1 | 347.3 | 1.9× slower |

Latency scales roughly linearly with variant count (each variant is one OCR
pass); the beam search adds a small amount on top of the OCR cost. Pick the tier
that fits your latency budget — or use the fast tier first and only escalate to
beam on failure.

> **GPU is slower here, and that's expected.** The captcha model is tiny (~23
> CTC timesteps) and CaptchaBeam calls it once per variant on single small
> images, sequentially — no batching. Per-call host↔device transfer and kernel
> launch overhead outweighs the compute saved, so CUDA runs ~2× slower than CPU.
> Preprocessing (OpenCV) and the beam-search loop also stay on CPU regardless.
> **Recommendation: run this workload on CPU.** GPU only helps if you batch (see
> below). To try it anyway: `DdddOcrBackend(use_gpu=True)` with `onnxruntime-gpu`
> + a matching CUDA/cuDNN runtime on `LD_LIBRARY_PATH`, or
> `scripts/benchmark.py --gpu`.

### Batched inference (`BatchedDdddOcrBackend`)

For lower single-image latency, the batched backend scores **all variants of an
image in one padded `session.run`** (via a dynamic-batch re-export of ddddocr's
model, generated and cached on first use) and hands the decoder **numpy
probabilities directly** — skipping the per-timestep 8210-wide nested-list
materialization and vectorizing the character aggregation.

```python
from captchabeam import CaptchaBeam
from captchabeam.backends import BatchedDdddOcrBackend

cb = CaptchaBeam(backend=BatchedDdddOcrBackend(), variants=18, decoder="beam")
cb.decode("captcha.png")                       # or BatchedDdddOcrBackend(use_gpu=True)
```

18-variants-beam latency, 300-image holdout (`scripts/benchmark.py --batched [--gpu]`):

| Mode | CPU ms/img | GPU ms/img | Exact |
|------|-----------:|-----------:|------:|
| per-variant (default) | 184.1 | 347.3 | **85.0%** |
| batched | **113.1** | 175.3 | 84.0% |

The win is almost entirely from dropping the nested-list conversion and
vectorizing the decode; the batched OCR call itself matters little on CPU.
**CPU batched (113 ms) is the fastest configuration** and still beats every GPU
mode — this workload is CPU-favorable. Trade-off: the dynamic-batch re-export
selects slightly different onnxruntime kernels (~1e-2 logit drift), costing about
**1 point of exact accuracy** (85.0% → 84.0%). Use the default per-variant path
when you want the last accuracy point; use batched when latency matters more.

### Retry success rate

Captchas usually allow re-rolling. With independent single-attempt success `p`,
the cumulative success over `n` tries is `1 − (1 − p)^n`:

| Attempts | otsu native (72.3%) | 6 native (76.3%) | 18 beam (85.0%) |
|---------:|--------------------:|-----------------:|----------------:|
| 1 | 72.30% | 76.30% | 85.00% |
| 3 | 97.87% | 98.67% | 99.66% |
| 5 | 99.84% | 99.93% | 99.99% |

So even an imperfect single-shot recognizer becomes highly reliable in practice
with a few retries.

> **Note:** these absolute numbers are for one specific captcha (5-char `A-Z/0-9`).
> The *method* transfers to other captchas; the *numbers* must be re-measured on
> your own labeled holdout via `captchabeam eval`.

---

## Datasets

`data/` holds the migrated, human-labeled holdout sets (each with `labels.csv`):

| Dataset | Count | Purpose |
|---------|------:|---------|
| `captcha_samples` | 100 | Early exploration |
| `captcha_holdout_100` | 100 | Holdout #1 |
| `captcha_holdout_extra_100` | 100 | Holdout #2 |
| `captcha_holdout_extra2_100` | 100 | Holdout #3 (pruning generalization check) |

---

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
pytest                          # 27 tests; ddddocr integration test auto-skips if absent
python scripts/benchmark.py     # reproduce the table above
```

Metrics: `exact` (whole string correct), `char` (character-level accuracy),
`len_ok` (output length matches spec).

---

## License

See [LICENSE](LICENSE).

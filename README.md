# CaptchaBeam

**Customizable restricted CTC beam-search decoding and OpenCV preprocessing variants for fixed-format captcha OCR.**

Pipeline: **multi-variant OpenCV preprocessing → OCR probability matrix →
restricted CTC beam search → agreement voting across variants.** It took a real
crawler from 62% → 85% exact accuracy; this packages that core so you can apply
it to *your* captcha by configuring charset and length.

```python
from captchabeam import CaptchaBeam

cb = CaptchaBeam()                       # 18 variants + beam + agreement (best config)
print(cb.decode("captcha.png").text)     # -> "XTRR3"
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for design and
[docs/benchmarks.md](docs/benchmarks.md) for the full methodology, GPU numbers,
optimization journey, and datasets.

---

## Install

```bash
pip install captchabeam[all]      # core + opencv + ddddocr (default backend)
```

Extras: `[cv]` preprocessing · `[ddddocr]` default backend · `[fast]` fast
backend · `[gpu]` onnxruntime-gpu · `[eval]`/`[all]`. The core beam decoder only
needs `numpy`.

---

## Backends

Pick a backend and pass it to `CaptchaBeam(backend=...)`. All produce identical
results; they differ in speed. Latency is per image, 18-variants-beam, 300-image
holdout, Intel i7-12700 (see [docs/benchmarks.md](docs/benchmarks.md)).

### Default — `DdddOcrBackend`

```python
from captchabeam import CaptchaBeam

cb = CaptchaBeam()                                   # default backend, 184 ms/img, 85.0% exact
```

### Fast (recommended) — `FastDdddOcrBackend`

Same static model and **char-for-char identical output**, but takes arrays
directly and uses a numpy-native decode. ~34% faster, no accuracy cost.

```python
from captchabeam import CaptchaBeam
from captchabeam.backends import FastDdddOcrBackend

cb = CaptchaBeam(backend=FastDdddOcrBackend())       # 121 ms/img, 85.0% exact
```

### GPU — `DdddOcrBackend(use_gpu=True)`

```python
cb = CaptchaBeam(backend=DdddOcrBackend(use_gpu=True))   # needs onnxruntime-gpu + CUDA/cuDNN
```

> GPU is **slower** for this workload (tiny model, no batching): 347 ms/img.
> Run on CPU. Details in [docs/benchmarks.md](docs/benchmarks.md).

### Your own model

Any callable `png_bytes -> {text, confidence, probabilities, charset}` is a valid
backend (`captchabeam.backends.OcrBackend`) — a custom CRNN or PaddleOCR drops in
and the beam decoder still applies.

```python
cb = CaptchaBeam(backend=MyCRNN())
```

### Latency / accuracy at a glance

| Backend | ms/img | Exact |
|---------|-------:|------:|
| `FastDdddOcrBackend` (CPU) | **121** | 85.0% |
| `DdddOcrBackend` (CPU, default) | 184 | 85.0% |
| `DdddOcrBackend(use_gpu=True)` | 347 | 85.0% |

---

## Configuring the decode

### Speed/accuracy tiers

```python
CaptchaBeam(variants=1,  decoder="native")   # fastest, single Otsu (~72% exact)
CaptchaBeam(variants=6,  decoder="native")   # balanced (~76%)
CaptchaBeam(variants=18, decoder="beam")     # most accurate, default (85%)
```

### Your own captcha spec

```python
from captchabeam import CaptchaBeam, DecodeConfig

cb = CaptchaBeam(decode_config=DecodeConfig(charset="0123456789", length=4, beam_size=16))
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

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
pytest                          # ddddocr/opencv tests auto-skip if absent
python scripts/benchmark.py [--fast] [--gpu]
```

---

## License

See [LICENSE](LICENSE).

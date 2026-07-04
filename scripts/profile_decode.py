"""Break down single-image 18-variant beam decode latency into buckets.

Buckets per image:
  decode_png : initial PNG -> grayscale (once per image)
  preprocess : OpenCV variant ops (per variant)
  png_encode : array -> PNG (the removable half of the roundtrip, per variant)
  imdecode   : PNG -> array again, estimate of ddddocr's internal re-decode
  ocr        : ddddocr.classification (imdecode + normalize + inference, per variant)
  beam       : restricted CTC beam decode (per variant)
  select     : agreement selection (per image)

Usage: python scripts/profile_decode.py [--limit N] [--gpu]
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from captchabeam import CaptchaBeam
from captchabeam.backends import DdddOcrBackend
from captchabeam.preprocess import encode_png, to_gray

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "data" / "captcha_holdout_100"


def profile(images, use_gpu):
    backend = DdddOcrBackend(use_gpu=use_gpu)
    cb = CaptchaBeam(backend=backend, variants=18, decoder="beam")
    # warm up (model load + first-call JIT)
    cb.decode(images[0])

    t = defaultdict(float)
    for path in images:
        png = Path(path).read_bytes()

        t0 = time.perf_counter()
        gray = to_gray(png)
        t["decode_png"] += time.perf_counter() - t0

        candidates = []
        for variant in cb.variants:
            t0 = time.perf_counter()
            arr = variant.apply(gray)
            t["preprocess"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            vpng = encode_png(arr)
            t["png_encode"] += time.perf_counter() - t0

            # estimate the re-decode ddddocr does internally
            t0 = time.perf_counter()
            cv2.imdecode(np.frombuffer(vpng, np.uint8), cv2.IMREAD_GRAYSCALE)
            t["imdecode"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            raw = backend(vpng)
            t["ocr"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            text, conf = cb._decoder.decode(raw)
            t["beam"] += time.perf_counter() - t0
            from captchabeam.types import Candidate

            candidates.append(Candidate(variant.name, text, conf))

        t0 = time.perf_counter()
        cb.selector.select(candidates)
        t["select"] += time.perf_counter() - t0

    n = len(images)
    device = "GPU" if use_gpu else "CPU"
    order = ["decode_png", "preprocess", "png_encode", "imdecode", "ocr", "beam", "select"]
    total = sum(t[k] for k in order)
    print(f"\ndevice={device} images={n}  (ocr bucket includes ddddocr's internal imdecode)\n")
    print(f"{'bucket':<12} {'ms/img':>9} {'%':>7}")
    print("-" * 30)
    for k in order:
        ms = t[k] / n * 1000
        print(f"{k:<12} {ms:>8.2f} {t[k] / total * 100:>6.1f}%")
    print("-" * 30)
    print(f"{'TOTAL':<12} {total / n * 1000:>8.2f} {100.0:>6.1f}%")
    print(f"\nnote: png_encode + imdecode = removable roundtrip overhead "
          f"(~{(t['png_encode'] + t['imdecode']) / n * 1000:.1f} ms/img, of which imdecode "
          f"is inside the ocr bucket)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()
    images = sorted(str(p) for p in args.dir.glob("*.png"))[: args.limit]
    profile(images, args.gpu)


if __name__ == "__main__":
    main()

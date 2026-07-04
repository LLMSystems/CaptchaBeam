"""Benchmark accuracy and inference latency for the three tiers on real data.

Usage: python scripts/benchmark.py [--limit N] [--dirs ...]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from captchabeam import CaptchaBeam
from captchabeam.eval import load_datasets, score

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRS = [
    ROOT / "data" / "captcha_holdout_100",
    ROOT / "data" / "captcha_holdout_extra_100",
    ROOT / "data" / "captcha_holdout_extra2_100",
]

TIERS = [
    ("otsu native (1 variant)", dict(variants=1, decoder="native")),
    ("6 variants native", dict(variants=6, decoder="native")),
    ("18 variants beam", dict(variants=18, decoder="beam")),
]


def run(dirs, limit):
    items = load_datasets(dirs)
    if limit:
        items = items[:limit]
    print(f"samples={len(items)}\n")
    print(f"{'tier':<26} {'exact':>7} {'char':>7} {'len_ok':>7} {'ms/img':>9}")
    print("-" * 62)
    for label, kwargs in TIERS:
        cb = CaptchaBeam(**kwargs)
        cb.decode(items[0].path)  # warm up (lazy model + first-call JIT)
        pairs = []
        t0 = time.perf_counter()
        for item in items:
            pairs.append((cb.decode(item.path).text, item.label))
        elapsed = time.perf_counter() - t0
        m = score(pairs, target_length=5)
        print(
            f"{label:<26} {m.exact_rate:>6.1%} {m.char_rate:>6.1%} "
            f"{m.length_ok_rate:>6.1%} {elapsed / len(items) * 1000:>8.1f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", type=Path, default=DEFAULT_DIRS)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    run(args.dirs, args.limit)


if __name__ == "__main__":
    main()

"""Cached evaluation harness.

The expensive step is OCR + decoding every preprocessing variant. This harness
caches per-variant candidates keyed by ``path|size|mtime|decoder|variants`` so
selector and ablation experiments rerun without re-invoking the model — the same
optimization the reference project's eval scripts used.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from ..config import DecodeConfig
from ..engine import CaptchaBeam
from ..select import AgreementSelector
from ..types import Candidate
from .dataset import LabeledImage, load_datasets
from .metrics import Metrics, score


def _sample_key(path: Path, decoder: str, variant_count: int) -> str:
    stat = path.stat()
    return f"{path.resolve()}|{stat.st_size}|{int(stat.st_mtime)}|{decoder}|v{variant_count}"


class SampleStore:
    """Per-sample cached candidates: list of ``{dataset, filename, label, candidates}``."""

    def __init__(self, samples: list[dict]) -> None:
        self.samples = samples

    @property
    def variant_names(self) -> list[str]:
        if not self.samples:
            return []
        return [c["variant_name"] for c in self.samples[0]["candidates"]]


def build_samples(
    dirs: Iterable[Path],
    *,
    variant_count: int = 18,
    decoder: str = "beam",
    decode_config: DecodeConfig | None = None,
    cache_path: Path | None = None,
    refresh: bool = False,
    backend=None,
) -> SampleStore:
    """Run (or load cached) per-variant candidates for every labeled image."""
    config = decode_config or DecodeConfig()
    cache: dict[str, dict] = {}
    if cache_path and cache_path.exists() and not refresh:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    engine = CaptchaBeam(
        backend=backend, variants=variant_count, decoder=decoder, decode_config=config
    )
    labeled: list[LabeledImage] = load_datasets(dirs)
    samples: list[dict] = []
    changed = False

    for item in labeled:
        key = _sample_key(item.path, decoder, variant_count)
        cached = cache.get(key)
        if cached is None:
            candidates = engine.candidates(item.path)
            cached = {
                "dataset": item.dataset,
                "filename": item.path.name,
                "label": item.label,
                "candidates": [asdict(c) for c in candidates],
            }
            cache[key] = cached
            changed = True
        samples.append(cached)

    if cache_path and changed:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    return SampleStore(samples)


def evaluate(
    store: SampleStore,
    *,
    target_length: int | None = 5,
    variants: list[str] | None = None,
    selector=None,
) -> Metrics:
    """Re-run the selector over cached candidates and score the result."""
    allowed = set(variants) if variants is not None else None
    selector = selector or AgreementSelector(target_length)
    pairs: list[tuple[str, str]] = []
    for sample in store.samples:
        cands = [
            Candidate(**c)
            for c in sample["candidates"]
            if allowed is None or c["variant_name"] in allowed
        ]
        if not cands:
            raise ValueError("No candidates left after filtering variants")
        chosen = selector.select(cands)
        pairs.append((chosen.text, sample["label"]))
    return score(pairs, target_length)

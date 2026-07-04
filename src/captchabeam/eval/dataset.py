"""Labeled dataset loading.

Each dataset directory contains images plus a ``labels.csv`` with ``filename``
and ``label`` columns (the format produced by the reference project's labeling
script). Loading is tolerant of ``file``/``text`` column aliases and a UTF-8 BOM.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True, slots=True)
class LabeledImage:
    dataset: str
    path: Path
    label: str


def load_labels(sample_dir: Path) -> list[LabeledImage]:
    sample_dir = Path(sample_dir)
    labels_path = sample_dir / "labels.csv"
    if not labels_path.exists():
        return []

    items: list[LabeledImage] = []
    with labels_path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            filename = (row.get("filename") or row.get("file") or "").strip()
            label = (row.get("label") or row.get("text") or "").strip().upper()
            path = sample_dir / filename
            if filename and label and path.exists():
                items.append(LabeledImage(sample_dir.name, path, label))
    return items


def load_datasets(dirs: Iterable[Path]) -> list[LabeledImage]:
    items: list[LabeledImage] = []
    for directory in dirs:
        items.extend(load_labels(Path(directory)))
    return items

"""Leave-one-out variant ablation.

For each variant, drop it and re-score the cached candidates. Reveals which
variants contribute (or hurt) accuracy. This is how the reference project ordered
and pruned its 18 variants without overfitting.
"""
from __future__ import annotations

from dataclasses import dataclass

from .harness import SampleStore, evaluate
from .metrics import Metrics


@dataclass(frozen=True, slots=True)
class AblationRow:
    dropped: str
    metrics: Metrics
    delta_exact: int


def leave_one_out(
    store: SampleStore,
    *,
    target_length: int | None = 5,
    selector=None,
) -> tuple[Metrics, list[AblationRow]]:
    """Return ``(baseline_with_all_variants, rows_sorted_by_exact_desc)``."""
    variant_names = store.variant_names
    baseline = evaluate(
        store, target_length=target_length, variants=variant_names, selector=selector
    )

    rows: list[AblationRow] = []
    for variant in variant_names:
        kept = [name for name in variant_names if name != variant]
        metrics = evaluate(store, target_length=target_length, variants=kept, selector=selector)
        rows.append(AblationRow(variant, metrics, metrics.exact - baseline.exact))

    rows.sort(key=lambda r: (r.metrics.exact, r.metrics.char_rate), reverse=True)
    return baseline, rows

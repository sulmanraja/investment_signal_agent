"""Alignment scorer — computes a weighted composite signal from normalized sources.

Weights reflect the relative predictive importance of each signal
category for a medium-term (3-12 month) equity investment horizon.
"""

from signals.normalizer import normalize_signal
from signals.category_registry import load_registry


DEFAULT_WEIGHTS = {
    "sec_revenue_growth":  0.25,
    "sec_gross_margin":    0.10,
    "macro_rate":          0.15,
    "macro_yield_curve":   0.10,
    "news_sentiment":      0.15,
    "github_growth":       0.10,
    "google_trends":       0.15,
}


def score_alignment(
    raw_signals: dict[str, float],
    weights: dict[str, float] | None = None,
) -> dict:
    """Compute a weighted composite alignment score.

    Args:
        raw_signals: Dict mapping signal source key → raw value.
        weights: Optional override for the default weight dict.

    Returns:
        Dict with:
          - "normalized": {source: normalized_value}
          - "weighted": {source: weighted_contribution}
          - "composite": float in [-1, 1]
          - "label": str — "STRONG BULL" | "BULL" | "MIXED" | "BEAR" | "STRONG BEAR"
    """
    w = weights or DEFAULT_WEIGHTS
    normalized = {}
    weighted = {}

    for source, raw in raw_signals.items():
        norm = normalize_signal(source, raw)
        weight = w.get(source, 0.0)
        normalized[source] = norm
        weighted[source] = norm * weight

    composite = sum(weighted.values())

    if composite >= 0.6:
        label = "STRONG BULL"
    elif composite >= 0.2:
        label = "BULL"
    elif composite >= -0.2:
        label = "MIXED"
    elif composite >= -0.6:
        label = "BEAR"
    else:
        label = "STRONG BEAR"

    return {
        "normalized": normalized,
        "weighted": weighted,
        "composite": composite,
        "label": label,
    }


if __name__ == "__main__":
    print("=== alignment_scorer local test ===")
    signals = {
        "sec_revenue_growth":  4.27,
        "sec_gross_margin":    0.784,
        "macro_rate":          5.25,
        "macro_yield_curve":  -0.30,
        "news_sentiment":      0.45,
        "github_growth":      3400,
        "google_trends":        81,
    }
    result = score_alignment(signals)
    print(f"  Composite: {result['composite']:+.3f}  → {result['label']}")
    print("  Per-source contributions:")
    for src, val in result["weighted"].items():
        print(f"    {src:<25s} norm={result['normalized'][src]:+.3f}  weighted={val:+.4f}")

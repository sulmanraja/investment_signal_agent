"""Signal normalizer — two modes for the investment signal pipeline.

Legacy (used in standalone analysis):
  normalize_signal(source, raw_value) → float in [-1, 1]

7-Agent Architecture (primary path):
  normalize_to_100(sub_scores, weights) → float in [0, 100]
  where sub_scores keys match the five DataCollector/Retrieval outputs:
    sec_edgar, github, news, google_trends, fred_macro
  and each value is already on a 0-100 scale.
"""

from typing import Any

# Default weights for the five sub-score dimensions
DEFAULT_WEIGHTS: dict[str, float] = {
    "sec_edgar":     0.30,
    "github":        0.20,
    "news":          0.20,
    "google_trends": 0.15,
    "fred_macro":    0.15,
}


def normalize_to_100(
    sub_scores: dict[str, float],
    weights: dict[str, float] | None = None,
) -> float:
    """Weighted composite of 0-100 sub-scores → single 0-100 score.

    Args:
        sub_scores: Dict with keys from DEFAULT_WEIGHTS, values in [0, 100].
        weights: Optional weight override. Must sum to ~1.0.

    Returns:
        Composite score in [0.0, 100.0].
    """
    w = weights or DEFAULT_WEIGHTS
    total_weight = 0.0
    composite = 0.0
    for key, score in sub_scores.items():
        wt = w.get(key, 0.0)
        composite += score * wt
        total_weight += wt
    if total_weight == 0.0:
        return 0.0
    # Re-scale if weights don't sum to 1 (e.g. partial key set)
    return max(0.0, min(100.0, composite / total_weight * 1.0 if total_weight == 1.0 else composite))


def normalize_signal(source: str, raw_value: Any) -> float:
    """Normalize a raw signal value to [-1, 1] (legacy path).

    Args:
        source: Signal source key. Supported values:
            "sec_revenue_growth"   — YoY growth rate as float (e.g. 4.27 for 427%)
            "sec_gross_margin"     — Gross margin as float (e.g. 0.784)
            "macro_rate"           — Fed funds rate (e.g. 5.25)
            "macro_yield_curve"    — 10Y-2Y spread (e.g. -0.3)
            "news_sentiment"       — Float in [-1, 1]
            "github_growth"        — Repo count change (int)
            "google_trends"        — Trend index 0-100
        raw_value: The raw numeric value.

    Returns:
        Normalized float in [-1, 1].
    """
    normalizers = {
        "sec_revenue_growth": lambda v: max(-1.0, min(1.0, v / 2.0)),
        "sec_gross_margin":   lambda v: max(-1.0, min(1.0, (v - 0.5) / 0.3)),
        "macro_rate":         lambda v: max(-1.0, min(1.0, -v / 7.0)),
        "macro_yield_curve":  lambda v: max(-1.0, min(1.0, v)),
        "news_sentiment":     lambda v: max(-1.0, min(1.0, float(v))),
        "github_growth":      lambda v: max(-1.0, min(1.0, v / 5000.0)),
        "google_trends":      lambda v: max(-1.0, min(1.0, (v - 50) / 50.0)),
    }
    fn = normalizers.get(source)
    if fn is None:
        raise ValueError(
            f"Unknown signal source: '{source}'. "
            f"Supported: {list(normalizers.keys())}"
        )
    return fn(raw_value)


if __name__ == "__main__":
    print("=== normalizer local test ===")

    print("\n--- normalize_to_100 (7-agent architecture) ---")
    sub_scores = {
        "sec_edgar":     82.0,
        "github":        74.0,
        "news":          68.0,
        "google_trends": 81.0,
        "fred_macro":    42.0,
    }
    composite = normalize_to_100(sub_scores)
    print(f"  Composite: {composite:.1f}/100")

    print("\n--- normalize_signal (legacy [-1, 1]) ---")
    cases = [
        ("sec_revenue_growth", 4.27),
        ("sec_gross_margin",   0.784),
        ("macro_rate",         5.25),
        ("macro_yield_curve", -0.30),
        ("news_sentiment",     0.45),
        ("github_growth",     3400),
        ("google_trends",       81),
    ]
    for source, value in cases:
        norm = normalize_signal(source, value)
        print(f"  {source:<25s} raw={value:<8}  normalized={norm:+.3f}")

"""Signal normalizer — maps raw tool outputs to a uniform [-1, 1] float signal.

Each source signal is clipped and linearly scaled so they can be
combined in the alignment scorer.
"""

from typing import Any


def normalize_signal(source: str, raw_value: Any) -> float:
    """Normalize a raw signal value to [-1, 1].

    Args:
        source: Signal source key. Supported values:
            "sec_revenue_growth"   — YoY growth rate as a float (e.g. 4.27 for 427%)
            "sec_gross_margin"     — Gross margin as a float (e.g. 0.784)
            "macro_rate"           — Fed funds rate (e.g. 5.25)
            "macro_yield_curve"    — 10Y-2Y spread (e.g. -0.3)
            "news_sentiment"       — Float in [-1, 1] from sentiment model
            "github_growth"        — Repo count change as integer
            "google_trends"        — Trend index 0-100
        raw_value: The raw numeric value.

    Returns:
        Normalized float in [-1, 1].
    """
    normalizers = {
        # Revenue growth: 0% → 0, ≥200% → +1, ≤-50% → -1
        "sec_revenue_growth": lambda v: max(-1.0, min(1.0, v / 2.0)),
        # Gross margin: 50% → 0, ≥80% → +1, ≤20% → -1
        "sec_gross_margin": lambda v: max(-1.0, min(1.0, (v - 0.5) / 0.3)),
        # Rate: 0% → 0 (neutral), ≥7% → -1 (headwind), negative rates → +1
        "macro_rate": lambda v: max(-1.0, min(1.0, -v / 7.0)),
        # Yield curve: +1% → +1 (bull), -1% → -1 (recessionary)
        "macro_yield_curve": lambda v: max(-1.0, min(1.0, v)),
        # News sentiment already in [-1, 1]
        "news_sentiment": lambda v: max(-1.0, min(1.0, float(v))),
        # GitHub growth: 0 repos → 0, ≥5000 → +1
        "github_growth": lambda v: max(-1.0, min(1.0, v / 5000.0)),
        # Google Trends 0-100 → [−1, 1] relative to neutral=50
        "google_trends": lambda v: max(-1.0, min(1.0, (v - 50) / 50.0)),
    }

    fn = normalizers.get(source)
    if fn is None:
        raise ValueError(f"Unknown signal source: '{source}'. "
                         f"Supported: {list(normalizers.keys())}")
    return fn(raw_value)


if __name__ == "__main__":
    print("=== normalizer local test ===")
    cases = [
        ("sec_revenue_growth", 4.27),
        ("sec_gross_margin", 0.784),
        ("macro_rate", 5.25),
        ("macro_yield_curve", -0.30),
        ("news_sentiment", 0.45),
        ("github_growth", 3400),
        ("google_trends", 81),
    ]
    for source, value in cases:
        norm = normalize_signal(source, value)
        print(f"  {source:<25s} raw={value:<8}  normalized={norm:+.3f}")

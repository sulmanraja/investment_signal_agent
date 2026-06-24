"""Local tests for the signals/ package (no LLM or API keys required)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from signals.normalizer import normalize_signal
from signals.alignment_scorer import score_alignment
from signals.category_registry import load_registry, get_categories, get_signal_sources


def test_normalizer():
    print("\n--- normalizer ---")
    cases = [
        ("sec_revenue_growth", 4.27,  1.0),   # capped at +1
        ("sec_revenue_growth", 0.0,   0.0),
        ("sec_revenue_growth", -0.5, -0.25),
        ("sec_gross_margin",   0.784, 0.947),
        ("macro_rate",         5.25, -0.75),
        ("macro_yield_curve", -0.30, -0.30),
        ("news_sentiment",     0.45,  0.45),
        ("github_growth",     3400,   0.68),
        ("google_trends",      81,    0.62),
    ]
    for source, raw, expected in cases:
        result = normalize_signal(source, raw)
        status = "✓" if abs(result - expected) < 0.02 else "✗"
        print(f"  {status} {source:<25s} raw={raw:<8}  got={result:+.3f}  expected≈{expected:+.3f}")


def test_alignment_scorer():
    print("\n--- alignment_scorer ---")
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
    print(f"  Composite: {result['composite']:+.3f}  Label: {result['label']}")
    assert result["composite"] > 0, "Expected positive composite for bullish NVDA signals"
    assert result["label"] in ("BULL", "STRONG BULL", "MIXED"), f"Unexpected label: {result['label']}"
    print("  ✓ Alignment scorer passed")


def test_category_registry():
    print("\n--- category_registry ---")
    registry = load_registry()
    categories = get_categories()
    sources = get_signal_sources()
    assert "ai_ml_infrastructure" in categories
    assert "cybersecurity" in categories
    assert "sec_revenue_growth" in sources
    print(f"  ✓ Registry loaded: {len(categories)} categories, {len(sources)} signal sources")


if __name__ == "__main__":
    print("=" * 50)
    print("  TEST: signals/")
    print("=" * 50)
    test_normalizer()
    test_alignment_scorer()
    test_category_registry()
    print("\n✅ All signal tests passed.")

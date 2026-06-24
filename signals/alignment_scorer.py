"""Alignment scorer — two modes for the investment signal pipeline.

7-Agent Architecture (primary path):
  classify_alignment(sub_scores) → AlignmentResult
  ALIGNED if alignment_spread ≤ 20 pts AND no hard contradiction.
  CONTRADICTORY if alignment_spread > 20 pts OR hard contradiction detected.

  Hard-contradiction patterns (per claud.md spec):
    Pattern 1: sec_edgar ≥ 70 AND news ≤ 30
    Pattern 2: github ≥ 70 AND google_trends ≥ 70 AND fred_macro ≤ 30

Legacy (standalone analysis):
  score_alignment(raw_signals, weights) → dict with composite in [-1, 1]
"""

from dataclasses import dataclass

from signals.normalizer import normalize_signal, normalize_to_100, DEFAULT_WEIGHTS
from signals.category_registry import load_registry

SPREAD_THRESHOLD = 20.0


@dataclass
class AlignmentResult:
    """Output of classify_alignment()."""
    classification: str          # "ALIGNED" or "CONTRADICTORY"
    alignment_spread: float      # max(sub_scores) - min(sub_scores)
    hard_contradiction: bool
    composite_score: float       # weighted 0-100 composite
    rationale: str


def _detect_hard_contradiction(sub_scores: dict[str, float]) -> tuple[bool, str]:
    """Return (is_contradiction, description)."""
    sec = sub_scores.get("sec_edgar", 50.0)
    news = sub_scores.get("news", 50.0)
    github = sub_scores.get("github", 50.0)
    trends = sub_scores.get("google_trends", 50.0)
    macro = sub_scores.get("fred_macro", 50.0)

    if sec >= 70.0 and news <= 30.0:
        return True, (
            f"Pattern 1: SEC filing strength (sec={sec:.0f}) conflicts sharply "
            f"with negative news sentiment (news={news:.0f})"
        )
    if github >= 70.0 and trends >= 70.0 and macro <= 30.0:
        return True, (
            f"Pattern 2: Strong developer adoption (github={github:.0f}, "
            f"trends={trends:.0f}) contradicts adverse macro conditions (macro={macro:.0f})"
        )
    return False, ""


def classify_alignment(
    sub_scores: dict[str, float],
    weights: dict[str, float] | None = None,
) -> AlignmentResult:
    """Classify signal alignment for the 7-agent architecture.

    Args:
        sub_scores: Dict with keys sec_edgar / github / news / google_trends / fred_macro,
                    each in [0, 100].
        weights: Optional weight override (defaults to DEFAULT_WEIGHTS).

    Returns:
        AlignmentResult with classification ALIGNED or CONTRADICTORY.
    """
    values = list(sub_scores.values())
    spread = max(values) - min(values) if values else 0.0
    composite = normalize_to_100(sub_scores, weights)
    hard, hard_desc = _detect_hard_contradiction(sub_scores)

    if hard or spread > SPREAD_THRESHOLD:
        if hard:
            rationale = hard_desc
        else:
            top_k = sorted(sub_scores.items(), key=lambda kv: kv[1], reverse=True)
            rationale = (
                f"Spread={spread:.1f}pts exceeds threshold={SPREAD_THRESHOLD}. "
                f"Highest: {top_k[0][0]}={top_k[0][1]:.0f}, "
                f"Lowest: {top_k[-1][0]}={top_k[-1][1]:.0f}."
            )
        return AlignmentResult(
            classification="CONTRADICTORY",
            alignment_spread=spread,
            hard_contradiction=hard,
            composite_score=composite,
            rationale=rationale,
        )

    sorted_by_score = sorted(sub_scores.items(), key=lambda kv: kv[1], reverse=True)
    rationale = (
        f"Spread={spread:.1f}pts ≤ {SPREAD_THRESHOLD}. "
        f"Signals in {sorted_by_score[-1][1]:.0f}–{sorted_by_score[0][1]:.0f} range."
    )
    return AlignmentResult(
        classification="ALIGNED",
        alignment_spread=spread,
        hard_contradiction=False,
        composite_score=composite,
        rationale=rationale,
    )


# ── Legacy path ──────────────────────────────────────────────────────────────

_LEGACY_WEIGHTS = {
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
    """Compute a weighted composite alignment score (legacy, returns [-1, 1]).

    Args:
        raw_signals: Dict mapping signal source key → raw value.
        weights: Optional weight override.

    Returns:
        Dict: normalized, weighted, composite, label.
    """
    w = weights or _LEGACY_WEIGHTS
    normalized = {}
    weighted = {}
    for source, raw in raw_signals.items():
        norm = normalize_signal(source, raw)
        wt = w.get(source, 0.0)
        normalized[source] = norm
        weighted[source] = norm * wt

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

    print("\n--- classify_alignment (7-agent architecture) ---")
    aligned_scores = {
        "sec_edgar":     75.0,
        "github":        68.0,
        "news":          72.0,
        "google_trends": 81.0,
        "fred_macro":    60.0,
    }
    r = classify_alignment(aligned_scores)
    print(f"  {r.classification}  spread={r.alignment_spread:.1f}  composite={r.composite_score:.1f}")
    print(f"  {r.rationale}")

    print()
    contradictory_scores = {
        "sec_edgar":     85.0,  # strong
        "github":        70.0,
        "news":          22.0,  # triggers Pattern 1
        "google_trends": 60.0,
        "fred_macro":    45.0,
    }
    r2 = classify_alignment(contradictory_scores)
    print(f"  {r2.classification}  spread={r2.alignment_spread:.1f}  hard={r2.hard_contradiction}")
    print(f"  {r2.rationale}")

    print("\n--- score_alignment (legacy) ---")
    signals = {
        "sec_revenue_growth": 4.27,
        "sec_gross_margin":   0.784,
        "macro_rate":         5.25,
        "macro_yield_curve": -0.30,
        "news_sentiment":     0.45,
        "github_growth":     3400,
        "google_trends":       81,
    }
    result = score_alignment(signals)
    print(f"  Composite: {result['composite']:+.3f}  → {result['label']}")

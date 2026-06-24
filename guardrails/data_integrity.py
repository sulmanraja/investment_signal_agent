"""Data Integrity Layer — Step 5 guardrails.

Responsibilities:
  1. Mandatory source attribution on every sub-score and retrieved passage.
  2. Staleness flagging: data older than its freshness threshold is flagged.
  3. Explicit failed-call status: no silent omission of missing sources.
  4. Source availability computation for escalation decisions.

Staleness thresholds (per data update cadence):
  news        48 h   — intraday news has a half-life of hours
  sec_edgar   2160 h — quarterly filing cycle (90 days)
  github      168 h  — weekly trending repo cycles
  google_trends 168 h
  fred_macro  168 h  — FRED series update weekly/monthly; fetch is real-time
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Optional

from schemas.messages import SourceAttribution, StalenessFlag

# Staleness thresholds in hours (keyed to sub-score dimension names)
STALENESS_THRESHOLDS_HOURS: dict[str, float] = {
    "news":          48.0,
    "sec_edgar":     2160.0,   # 90 days
    "github":        168.0,    # 7 days
    "google_trends": 168.0,
    "fred_macro":    168.0,
}


def build_attribution(
    source_key: str,
    tool_name: str,
    query: str,
    record_count: int,
    succeeded: bool,
    retrieved_at: str,
) -> SourceAttribution:
    """Build a mandatory SourceAttribution record for a sub-score."""
    if not succeeded:
        status = "failed"
    elif record_count == 0:
        status = "empty"
    else:
        status = "ok"
    return SourceAttribution(
        source_key=source_key,
        tool_name=tool_name,
        query=query,
        retrieved_at=retrieved_at,
        record_count=record_count,
        status=status,
    )


def check_staleness(
    source_key: str,
    data_timestamp: Optional[str],
    retrieved_at: str,
) -> StalenessFlag:
    """Check whether source data is stale.

    Args:
        source_key:      The sub-score dimension key.
        data_timestamp:  ISO-8601 timestamp from the data itself (e.g. article
                         published_at, filing date). If None, we use retrieved_at.
        retrieved_at:    ISO-8601 UTC timestamp when the fetch completed.

    Returns:
        StalenessFlag with is_stale=True if age exceeds threshold.
    """
    threshold_h = STALENESS_THRESHOLDS_HOURS.get(source_key, 168.0)
    now = datetime.now(timezone.utc)

    reference_ts = data_timestamp or retrieved_at
    try:
        ts = datetime.fromisoformat(reference_ts.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_h = (now - ts).total_seconds() / 3600.0
        is_stale = age_h > threshold_h
        reason = (
            f"Data is {age_h:.1f}h old; threshold={threshold_h:.0f}h"
            if is_stale
            else f"Data age {age_h:.1f}h within {threshold_h:.0f}h threshold"
        )
    except (ValueError, TypeError):
        age_h = None
        is_stale = False
        reason = "Could not parse data timestamp — staleness not determinable"

    return StalenessFlag(
        source_key=source_key,
        is_stale=is_stale,
        data_age_hours=age_h,
        threshold_hours=threshold_h,
        reason=reason,
    )


def check_news_staleness(articles: list[dict], retrieved_at: str) -> StalenessFlag:
    """Check staleness using the most recent article's published_at."""
    dates = [
        a.get("published_at") or a.get("pubDate") or a.get("date")
        for a in articles
        if a.get("published_at") or a.get("pubDate") or a.get("date")
    ]
    most_recent = max(dates) if dates else None
    return check_staleness("news", most_recent, retrieved_at)


def check_edgar_staleness(results_by_ticker: dict, retrieved_at: str) -> StalenessFlag:
    """Check staleness using the most recent SEC filing date across tickers."""
    dates = []
    for result in results_by_ticker.values():
        for excerpt in result.get("sec_edgar_excerpts", []):
            d = excerpt.get("date")
            if d:
                dates.append(d)
    most_recent = max(dates) if dates else None
    return check_staleness("sec_edgar", most_recent, retrieved_at)


def check_source_availability(
    attributions: list[SourceAttribution],
) -> tuple[float, list[str]]:
    """Return (availability_rate, failed_source_keys).

    availability_rate = sources with status 'ok' / total sources attempted.
    """
    if not attributions:
        return 0.0, []
    ok_count = sum(1 for a in attributions if a.status == "ok")
    failed = [a.source_key for a in attributions if a.status == "failed"]
    return round(ok_count / len(attributions), 3), failed


if __name__ == "__main__":
    print("=== data_integrity local test ===")
    now_iso = datetime.now(timezone.utc).isoformat()
    old_iso = (datetime.now(timezone.utc) - timedelta(hours=72)).isoformat()

    attr = build_attribution("github", "github_tool.fetch_github_trends",
                             "AI infrastructure GPU", 10, True, now_iso)
    print(f"Attribution: {attr.source_key} → {attr.status} ({attr.record_count} records)")

    fresh = check_staleness("github", now_iso, now_iso)
    stale = check_staleness("news", old_iso, now_iso)
    print(f"Fresh: {fresh.source_key} stale={fresh.is_stale} ({fresh.reason})")
    print(f"Stale: {stale.source_key} stale={stale.is_stale} ({stale.reason})")

    attrs = [
        build_attribution("github", "github_tool", "AI", 8, True, now_iso),
        build_attribution("news", "newsdata_tool", "AI chips", 0, False, now_iso),
        build_attribution("fred_macro", "fred_macro_tool", "", 5, True, now_iso),
    ]
    rate, failed = check_source_availability(attrs)
    print(f"Availability: {rate:.0%}  Failed: {failed}")

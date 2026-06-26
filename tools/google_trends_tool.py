"""Google Trends tool (MCP integration).

Fetches relative search interest for keywords to surface retail-sentiment
signals and product/brand momentum.
"""

import random
import time

from pytrends.request import TrendReq
from typing import Optional

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)

_MAX_RETRIES = 4
_BASE_DELAY_S = 5.0  # first retry waits ~5s; doubles each attempt + jitter


def _is_rate_limited(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "too many requests" in msg or "rate limit" in msg


def fetch_google_trends(
    keywords: list[str],
    timeframe: str = "today 3-m",
    geo: str = "US",
) -> dict:
    """Fetch Google Trends interest-over-time for given keywords.

    Retries up to _MAX_RETRIES times with exponential backoff on 429 responses.

    Args:
        keywords: Up to 5 keywords, e.g. ["NVIDIA GPU", "AMD GPU"].
        timeframe: Trends timeframe string, e.g. "today 3-m", "today 12-m".
        geo: ISO country code, e.g. "US". Empty string for worldwide.

    Returns:
        Dict with:
          - "interest_over_time": DataFrame as dict (date → {kw: score}).
          - "related_queries": {kw: {"top": [...], "rising": [...]}}.
    """
    pt = TrendReq(
        hl="en-US",
        tz=360,
        timeout=(10, 25),
        requests_args={"headers": {"User-Agent": _USER_AGENT}},
    )

    iot = None
    for attempt in range(_MAX_RETRIES):
        try:
            pt.build_payload(keywords[:5], timeframe=timeframe, geo=geo)
            iot = pt.interest_over_time()
            break
        except Exception as exc:
            if attempt == _MAX_RETRIES - 1 or not _is_rate_limited(exc):
                raise
            delay = _BASE_DELAY_S * (2 ** attempt) + random.uniform(0, 3)
            print(f"  [Trends] 429 rate-limited — retrying in {delay:.1f}s "
                  f"(attempt {attempt + 1}/{_MAX_RETRIES})")
            time.sleep(delay)

    related = {}
    for kw in keywords[:5]:
        try:
            rq = pt.related_queries()
            related[kw] = {
                "top": rq[kw]["top"].to_dict("records") if rq[kw]["top"] is not None else [],
                "rising": rq[kw]["rising"].to_dict("records") if rq[kw]["rising"] is not None else [],
            }
        except Exception:
            related[kw] = {"top": [], "rising": []}

    return {
        "interest_over_time": iot.to_dict() if iot is not None and not iot.empty else {},
        "related_queries": related,
    }


if __name__ == "__main__":
    print("=== google_trends_tool local test ===")
    data = fetch_google_trends(["NVIDIA GPU", "AMD GPU"], timeframe="today 3-m")
    iot = data["interest_over_time"]
    if iot:
        dates = list(iot.get("NVIDIA GPU", {}).keys())
        print(f"  Returned {len(dates)} data points.")
        if dates:
            last_date = max(dates)
            print(f"  Latest ({last_date}):", {k: v.get(last_date) for k, v in iot.items() if k != "isPartial"})
    else:
        print("  No data returned.")

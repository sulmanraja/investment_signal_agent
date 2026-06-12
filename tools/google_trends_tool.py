"""Google Trends tool (MCP integration).

Fetches relative search interest for keywords to surface retail-sentiment
signals and product/brand momentum.
"""

from pytrends.request import TrendReq
from typing import Optional


def fetch_google_trends(
    keywords: list[str],
    timeframe: str = "today 3-m",
    geo: str = "US",
) -> dict:
    """Fetch Google Trends interest-over-time for given keywords.

    Args:
        keywords: Up to 5 keywords, e.g. ["NVIDIA GPU", "AMD GPU"].
        timeframe: Trends timeframe string, e.g. "today 3-m", "today 12-m".
        geo: ISO country code, e.g. "US". Empty string for worldwide.

    Returns:
        Dict with:
          - "interest_over_time": DataFrame as dict (date → {kw: score}).
          - "related_queries": {kw: {"top": [...], "rising": [...]}}.
    """
    pt = TrendReq(hl="en-US", tz=360)
    pt.build_payload(keywords[:5], timeframe=timeframe, geo=geo)

    iot = pt.interest_over_time()
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
        "interest_over_time": iot.to_dict() if not iot.empty else {},
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

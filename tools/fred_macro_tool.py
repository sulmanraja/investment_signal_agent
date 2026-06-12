"""FRED (Federal Reserve Economic Data) macro tool (MCP integration).

Fetches macroeconomic indicators that contextualize the investment
environment: interest rates, CPI, GDP, unemployment, etc.
"""

import os
import requests
from typing import Optional


FRED_API_KEY = os.getenv("FRED_API_KEY", "")
FRED_BASE = "https://api.stlouisfed.org/fred"

# Common series IDs for quick reference
COMMON_SERIES = {
    "fed_funds_rate":   "FEDFUNDS",
    "cpi_yoy":          "CPIAUCSL",
    "real_gdp_growth":  "A191RL1Q225SBEA",
    "unemployment":     "UNRATE",
    "10y_treasury":     "DGS10",
    "2y_treasury":      "DGS2",
    "yield_curve":      "T10Y2Y",
    "pce_inflation":    "PCEPI",
    "ism_manufacturing":"MANEMP",
    "credit_spread":    "BAMLH0A0HYM2",
}


def fetch_macro_indicators(
    series_ids: Optional[list[str]] = None,
    observation_start: str = "2023-01-01",
    limit: int = 12,
) -> dict[str, list[dict]]:
    """Fetch FRED data series.

    Args:
        series_ids: List of FRED series IDs. Defaults to the 10 common ones.
        observation_start: ISO date string for the start of the series.
        limit: Max observations per series (most recent N).

    Returns:
        Dict mapping series_id → list of {"date": str, "value": str}.
    """
    if not FRED_API_KEY:
        raise EnvironmentError("FRED_API_KEY not set in environment.")

    if series_ids is None:
        series_ids = list(COMMON_SERIES.values())

    results = {}
    for sid in series_ids:
        params = {
            "series_id": sid,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "observation_start": observation_start,
            "sort_order": "desc",
            "limit": limit,
        }
        resp = requests.get(f"{FRED_BASE}/series/observations", params=params)
        resp.raise_for_status()
        observations = resp.json().get("observations", [])
        results[sid] = [{"date": o["date"], "value": o["value"]} for o in observations]

    return results


def get_latest_snapshot() -> dict[str, str]:
    """Return the single most-recent value for each common macro series."""
    data = fetch_macro_indicators(limit=1)
    return {
        name: data[sid][0]["value"] if data.get(sid) else "N/A"
        for name, sid in COMMON_SERIES.items()
    }


if __name__ == "__main__":
    print("=== fred_macro_tool local test ===")
    try:
        snapshot = get_latest_snapshot()
        for name, value in snapshot.items():
            print(f"  {name:<25s} {value}")
    except EnvironmentError as e:
        print(f"  Skipped: {e}")

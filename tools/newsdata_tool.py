"""NewsData.io tool (MCP integration).

Fetches recent news articles for a ticker or topic to surface
sentiment and event-driven signals.
"""

import os
import requests
from typing import Optional


NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY", "")
NEWSDATA_URL = "https://newsdata.io/api/1/news"


def fetch_news(
    query: str,
    language: str = "en",
    category: Optional[str] = "business",
    limit: int = 10,
) -> list[dict]:
    """Fetch recent news articles matching the query.

    Args:
        query: Search string, e.g. "NVDA earnings" or "AI chip demand".
        language: ISO language code.
        category: News category filter (business, technology, etc.).
        limit: Max articles to return (API max is 10 per page).

    Returns:
        List of article dicts with keys: title, description, source, url,
        published_at, sentiment.
    """
    if not NEWSDATA_API_KEY:
        raise EnvironmentError("NEWSDATA_API_KEY not set in environment.")

    params = {
        "apikey": NEWSDATA_API_KEY,
        "q": query,
        "language": language,
        "size": min(limit, 10),
    }
    if category:
        params["category"] = category

    resp = requests.get(NEWSDATA_URL, params=params)
    resp.raise_for_status()
    articles = resp.json().get("results", [])

    return [
        {
            "title": a.get("title", ""),
            "description": a.get("description", ""),
            "source": a.get("source_id", ""),
            "url": a.get("link", ""),
            "published_at": a.get("pubDate", ""),
            "sentiment": a.get("sentiment", None),
        }
        for a in articles
    ]


if __name__ == "__main__":
    print("=== newsdata_tool local test ===")
    try:
        articles = fetch_news("NVDA earnings AI chips", limit=5)
        for a in articles:
            print(f"  [{a['published_at']}] {a['title'][:80]}")
    except EnvironmentError as e:
        print(f"  Skipped: {e}")

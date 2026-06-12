"""GitHub trending repositories tool (MCP integration).

Fetches trending repos by language/topic to surface developer-sentiment
signals that correlate with technology adoption curves.
"""

import os
import requests
from typing import Optional


GITHUB_API = "https://api.github.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    **({"Authorization": f"Bearer {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}),
}


def fetch_github_trends(
    topic: str,
    language: Optional[str] = None,
    min_stars: int = 100,
    limit: int = 10,
) -> list[dict]:
    """Search GitHub for trending repos related to a topic.

    Args:
        topic: Search term, e.g. "AI infrastructure" or "NVDA CUDA".
        language: Optional language filter, e.g. "Python".
        min_stars: Minimum star count to filter noise.
        limit: Max repos to return.

    Returns:
        List of dicts with keys: name, stars, forks, description, url.
    """
    query = f"{topic} stars:>{min_stars}"
    if language:
        query += f" language:{language}"

    params = {
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": limit,
    }

    resp = requests.get(f"{GITHUB_API}/search/repositories", headers=_HEADERS, params=params)
    resp.raise_for_status()

    items = resp.json().get("items", [])
    return [
        {
            "name": r["full_name"],
            "stars": r["stargazers_count"],
            "forks": r["forks_count"],
            "description": r.get("description", ""),
            "url": r["html_url"],
            "pushed_at": r.get("pushed_at", ""),
        }
        for r in items
    ]


if __name__ == "__main__":
    print("=== github_tool local test ===")
    repos = fetch_github_trends("CUDA GPU inference", language="Python", limit=5)
    for r in repos:
        print(f"  {r['name']:40s}  ⭐ {r['stars']:>6,}  {r['description'][:60]}")

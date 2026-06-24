"""Loader for category_registry.json."""

import json
from pathlib import Path

_REGISTRY_PATH = Path(__file__).parent / "category_registry.json"


def load_registry() -> dict:
    """Return the full category registry as a dict."""
    return json.loads(_REGISTRY_PATH.read_text())


def get_categories() -> dict:
    """Return the technology category definitions (id → config)."""
    return load_registry()["categories"]


def get_weights() -> dict:
    """Return the signal-source weights for the composite score."""
    return load_registry()["weights"]


def get_category_ids() -> list[str]:
    """Return the ordered list of category IDs."""
    return list(load_registry()["categories"].keys())


def get_signal_sources() -> list[str]:
    """Return the legacy signal source keys recognized by normalize_signal."""
    return [
        "sec_revenue_growth",
        "sec_gross_margin",
        "macro_rate",
        "macro_yield_curve",
        "news_sentiment",
        "github_growth",
        "google_trends",
    ]

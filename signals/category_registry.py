"""Loader for category_registry.json."""

import json
from pathlib import Path

_REGISTRY_PATH = Path(__file__).parent / "category_registry.json"


def load_registry() -> dict:
    """Return the full category registry as a dict."""
    return json.loads(_REGISTRY_PATH.read_text())


def get_categories() -> dict:
    return load_registry()["categories"]


def get_signal_sources() -> dict:
    return load_registry()["signal_sources"]

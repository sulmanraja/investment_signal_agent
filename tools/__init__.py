def __getattr__(name):
    _map = {
        "ingest_tickers":        ("tools.edgar_ingest",       "ingest_tickers"),
        "retrieve_context":      ("tools.edgar_retriever",    "retrieve_context"),
        "fetch_github_trends":   ("tools.github_tool",        "fetch_github_trends"),
        "fetch_google_trends":   ("tools.google_trends_tool", "fetch_google_trends"),
        "fetch_news":            ("tools.newsdata_tool",      "fetch_news"),
        "fetch_macro_indicators":("tools.fred_macro_tool",    "fetch_macro_indicators"),
    }
    if name in _map:
        import importlib
        mod = importlib.import_module(_map[name][0])
        return getattr(mod, _map[name][1])
    raise AttributeError(f"module 'tools' has no attribute {name!r}")

__all__ = [
    "ingest_tickers", "retrieve_context", "fetch_github_trends",
    "fetch_google_trends", "fetch_news", "fetch_macro_indicators",
]

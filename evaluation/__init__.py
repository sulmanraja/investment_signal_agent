def __getattr__(name):
    _map = {
        "RunMetrics":    ("evaluation.metrics", "RunMetrics"),
        "MetricsStore":  ("evaluation.metrics", "MetricsStore"),
    }
    if name in _map:
        import importlib
        mod = importlib.import_module(_map[name][0])
        return getattr(mod, _map[name][1])
    raise AttributeError(f"module 'evaluation' has no attribute {name!r}")


__all__ = ["RunMetrics", "MetricsStore"]

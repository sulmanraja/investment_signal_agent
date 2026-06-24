def __getattr__(name):
    _map = {
        "check_staleness":              ("guardrails.data_integrity", "check_staleness"),
        "build_attribution":            ("guardrails.data_integrity", "build_attribution"),
        "check_source_availability":    ("guardrails.data_integrity", "check_source_availability"),
        "EscalationEvaluator":          ("guardrails.escalation",    "EscalationEvaluator"),
        "prepend_disclaimer":           ("guardrails.output_audit",  "prepend_disclaimer"),
        "audit_citation_groundedness":  ("guardrails.output_audit",  "audit_citation_groundedness"),
        "enforce_confidence_stance":    ("guardrails.output_audit",  "enforce_confidence_stance"),
        "compute_confidence_level":     ("guardrails.output_audit",  "compute_confidence_level"),
    }
    if name in _map:
        import importlib
        mod = importlib.import_module(_map[name][0])
        return getattr(mod, _map[name][1])
    raise AttributeError(f"module 'guardrails' has no attribute {name!r}")


__all__ = [
    "check_staleness", "build_attribution", "check_source_availability",
    "EscalationEvaluator",
    "prepend_disclaimer", "audit_citation_groundedness",
    "enforce_confidence_stance", "compute_confidence_level",
]

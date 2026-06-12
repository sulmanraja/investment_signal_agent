def __getattr__(name):
    _map = {
        "build_ooda_chain":            ("chains.ooda_loop",       "build_ooda_chain"),
        "build_signal_alignment_chain":("chains.signal_alignment","build_signal_alignment_chain"),
        "build_standard_scoring_chain":("chains.standard_scoring","build_standard_scoring_chain"),
        "build_tot_stance_chain":      ("chains.tot_stance_chain","build_tot_stance_chain"),
        "build_synthesis_chain":       ("chains.synthesis_chain", "build_synthesis_chain"),
    }
    if name in _map:
        import importlib
        mod = importlib.import_module(_map[name][0])
        return getattr(mod, _map[name][1])
    raise AttributeError(f"module 'chains' has no attribute {name!r}")

__all__ = [
    "build_ooda_chain", "build_signal_alignment_chain",
    "build_standard_scoring_chain", "build_tot_stance_chain", "build_synthesis_chain",
]

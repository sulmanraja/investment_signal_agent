def __getattr__(name):
    _map = {
        "ThoughtNode":       ("tot.thought_node", "ThoughtNode"),
        "generate_branches": ("tot.branches",     "generate_branches"),
        "evaluate_node":     ("tot.evaluator",    "evaluate_node"),
        "prune_beam":        ("tot.pruner",        "prune_beam"),
        "run_beam_search":   ("tot.beam_search",   "run_beam_search"),
    }
    if name in _map:
        module_path, attr = _map[name]
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module 'tot' has no attribute {name!r}")

__all__ = ["ThoughtNode", "generate_branches", "evaluate_node", "prune_beam", "run_beam_search"]

def __getattr__(name):
    _map = {
        "OrchestratorAgent":      ("agents.orchestrator_agent",       "OrchestratorAgent"),
        "DataCollectorAgent":     ("agents.data_collector_agent",     "DataCollectorAgent"),
        "RetrievalAgent":         ("agents.retrieval_agent",          "RetrievalAgent"),
        "SignalAnalystAgent":     ("agents.signal_analyst_agent",     "SignalAnalystAgent"),
        "ThoughtGeneratorAgent":  ("agents.thought_generator_agent",  "ThoughtGeneratorAgent"),
        "CriticAgent":            ("agents.critic_agent",             "CriticAgent"),
        "SynthesisAgent":         ("agents.synthesis_agent",          "SynthesisAgent"),
    }
    if name in _map:
        import importlib
        mod = importlib.import_module(_map[name][0])
        return getattr(mod, _map[name][1])
    raise AttributeError(f"module 'agents' has no attribute {name!r}")


__all__ = [
    "OrchestratorAgent",
    "DataCollectorAgent",
    "RetrievalAgent",
    "SignalAnalystAgent",
    "ThoughtGeneratorAgent",
    "CriticAgent",
    "SynthesisAgent",
]

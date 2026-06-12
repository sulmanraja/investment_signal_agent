def __getattr__(name):
    _map = {
        "EdgarRagAgent":       ("agents.edgar_rag_agent",        "EdgarRagAgent"),
        "ThoughtGeneratorAgent":("agents.thought_generator_agent","ThoughtGeneratorAgent"),
        "CriticAgent":         ("agents.critic_agent",           "CriticAgent"),
        "GitHubTrendsAgent":   ("agents.github_trends_agent",    "GitHubTrendsAgent"),
        "NewsAgent":           ("agents.news_agent",             "NewsAgent"),
        "MacroAgent":          ("agents.macro_agent",            "MacroAgent"),
    }
    if name in _map:
        import importlib
        mod = importlib.import_module(_map[name][0])
        return getattr(mod, _map[name][1])
    raise AttributeError(f"module 'agents' has no attribute {name!r}")

__all__ = [
    "EdgarRagAgent", "ThoughtGeneratorAgent", "CriticAgent",
    "GitHubTrendsAgent", "NewsAgent", "MacroAgent",
]

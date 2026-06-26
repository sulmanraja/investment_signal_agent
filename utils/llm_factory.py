"""LLM factory — switches between Ollama (local) and Claude API (MCP / cloud).

Provider is selected via the AGENT_LLM_PROVIDER environment variable:
  ollama     (default) — local Ollama; requires `ollama serve` and the model pulled
  anthropic            — Claude API; requires ANTHROPIC_API_KEY in .env

Model is selected via AGENT_LLM:
  ollama mode    → default: granite3.3:8b
  anthropic mode → default: claude-haiku-4-5-20251001
"""

import os
from crewai import LLM

_PROVIDER = os.getenv("AGENT_LLM_PROVIDER", "ollama")

if _PROVIDER == "anthropic":
    _MODEL = os.getenv("AGENT_LLM", "claude-haiku-4-5-20251001")
else:
    _MODEL   = os.getenv("AGENT_LLM", "granite3.3:8b")
    _OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def make_llm(temperature: float) -> LLM:
    """Return a CrewAI LLM configured for the active provider."""
    if _PROVIDER == "anthropic":
        return LLM(
            model=f"anthropic/{_MODEL}",
            temperature=temperature,
        )
    return LLM(
        model=f"ollama/{_MODEL}",
        base_url=_OLLAMA_URL,
        temperature=temperature,
    )

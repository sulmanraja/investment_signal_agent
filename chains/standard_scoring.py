"""Standard Scoring chain (LCEL).

Produces a structured 0-100 investment signal score across five dimensions:
  - Fundamental (30 pts): Revenue, margins, guidance from SEC filings.
  - Macro (20 pts): Rate environment, yield curve, credit spreads.
  - Sentiment (20 pts): News tone and social/search trends.
  - Momentum (15 pts): GitHub and Google Trends adoption signals.
  - Technicals (15 pts): Qualitative price-action notes (no live prices).
"""

import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

LLM_MODEL = os.getenv("AGENT_LLM", "granite3.3:8b")

_SCORING_PROMPT = PromptTemplate.from_template(
    """You are a systematic investment scoring engine for {ticker}.

Signals:
{signals}

Score each dimension on a 0-to-max scale:
1. Fundamental  (0-30): Revenue growth, margins, guidance quality.
2. Macro        (0-20): Rate/inflation environment for this sector.
3. Sentiment    (0-20): News tone, retail interest.
4. Momentum     (0-15): GitHub/Google Trends adoption signals.
5. Technicals   (0-15): Qualitative price trend / relative strength.

For each dimension, provide: score, max, and a one-line rationale.
Then compute: Total Score = sum of all scores.
Signal: STRONG BUY (>80) | BUY (60-80) | HOLD (40-60) | SELL (20-40) | STRONG SELL (<20)

Format as a clean table followed by "Total: <score>/100  Signal: <label>".
"""
)


def build_standard_scoring_chain():
    """Build and return the standard scoring LCEL chain.

    Input: dict with keys "ticker" and "signals".
    Output: str with scoring table and final signal.
    """
    llm = ChatOllama(model=LLM_MODEL, temperature=0.1)
    return _SCORING_PROMPT | llm | StrOutputParser()


if __name__ == "__main__":
    print("=== standard_scoring local test ===")
    chain = build_standard_scoring_chain()
    result = chain.invoke({
        "ticker": "NVDA",
        "signals": (
            "SEC: Revenue $22.6B (+427% YoY), gross margin 78.4%, strong guidance.\n"
            "Macro: Fed funds 5.25%, yield curve -0.3 (inverted).\n"
            "News: Positive — Blackwell on schedule; Risk — export controls.\n"
            "GitHub: CUDA repos +3,400 in 90 days (strong momentum).\n"
            "Google Trends: 'NVIDIA GPU' +62% in 3-month window."
        ),
    })
    print(result)

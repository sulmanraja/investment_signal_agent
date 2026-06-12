"""OODA Loop chain (LCEL).

Observe → Orient → Decide → Act.
Aggregates raw signals from all tools, orients them relative to the
investment mandate, decides a stance, and produces an action summary.
"""

import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap

LLM_MODEL = os.getenv("AGENT_LLM", "granite3.3:8b")

_OODA_PROMPT = PromptTemplate.from_template(
    """You are an investment analyst running an OODA loop for {ticker}.

--- OBSERVE ---
{signals}

--- ORIENT ---
Relate the above signals to the company's fundamentals, competitive position,
and the macro environment. What do these signals collectively suggest?

--- DECIDE ---
What is the recommended investment stance? (Bullish / Bearish / Neutral / Contrarian)
What is the confidence level? (High / Medium / Low)

--- ACT ---
What specific action should a portfolio manager take? (e.g., "Add 2% position",
"Reduce exposure by 1%", "Hold, monitor next earnings", "Short with tight stop")

Respond with clearly labeled ORIENT, DECIDE, and ACT sections.
"""
)


def build_ooda_chain():
    """Build and return the OODA loop LCEL chain.

    Input: dict with keys "ticker" and "signals".
    Output: str with ORIENT / DECIDE / ACT sections.
    """
    llm = ChatOllama(model=LLM_MODEL, temperature=0.3)
    return _OODA_PROMPT | llm | StrOutputParser()


if __name__ == "__main__":
    print("=== ooda_loop local test ===")
    chain = build_ooda_chain()
    result = chain.invoke({
        "ticker": "NVDA",
        "signals": (
            "SEC: Data-center revenue $22.6B (+427% YoY). Gross margin 78.4%.\n"
            "News: Blackwell GPU shipments ahead of schedule.\n"
            "GitHub: 3,400 new CUDA repos in 90 days.\n"
            "Macro: Fed funds 5.25%, yield curve inverted -0.3.\n"
            "Google Trends: 'NVIDIA GPU' up 62% 3-month interest."
        ),
    })
    print(result)

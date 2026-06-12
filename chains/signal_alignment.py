"""Signal Alignment chain (LCEL).

Checks whether individual signals from different sources (SEC, macro, news,
GitHub, trends) are directionally aligned or divergent. Outputs an alignment
summary and a composite alignment score in [-1, 1].
"""

import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

LLM_MODEL = os.getenv("AGENT_LLM", "granite3.3:8b")

_ALIGNMENT_PROMPT = PromptTemplate.from_template(
    """You are a quantitative signal analyst assessing cross-signal alignment for {ticker}.

Signals provided:
{signals}

For each signal source, state its directional bias: BULLISH (+1), NEUTRAL (0), or BEARISH (-1).

Then compute the composite alignment score as the simple average.

Format your response as:
- SEC Filing:      <BULLISH|NEUTRAL|BEARISH> (rationale in ≤10 words)
- Macro:           <BULLISH|NEUTRAL|BEARISH> (rationale in ≤10 words)
- News Sentiment:  <BULLISH|NEUTRAL|BEARISH> (rationale in ≤10 words)
- GitHub Trends:   <BULLISH|NEUTRAL|BEARISH> (rationale in ≤10 words)
- Google Trends:   <BULLISH|NEUTRAL|BEARISH> (rationale in ≤10 words)

Composite Score: <float in [-1, 1]>
Alignment: <STRONG BULL | BULL | MIXED | BEAR | STRONG BEAR>
"""
)


def build_signal_alignment_chain():
    """Build and return the signal alignment LCEL chain.

    Input: dict with keys "ticker" and "signals".
    Output: str with per-source bias and composite score.
    """
    llm = ChatOllama(model=LLM_MODEL, temperature=0.1)
    return _ALIGNMENT_PROMPT | llm | StrOutputParser()


if __name__ == "__main__":
    print("=== signal_alignment local test ===")
    chain = build_signal_alignment_chain()
    result = chain.invoke({
        "ticker": "NVDA",
        "signals": (
            "SEC: Revenue +427% YoY, margins expanding.\n"
            "Macro: Rates elevated, yield curve inverted.\n"
            "News: Blackwell shipments ahead of schedule; export control risk.\n"
            "GitHub: CUDA repos surging.\n"
            "Google Trends: 'NVIDIA GPU' up 62% in 3 months."
        ),
    })
    print(result)

"""Synthesis chain (LCEL).

Final step in the pipeline. Takes the scored signals, alignment analysis,
ToT stance, and OODA action summary and synthesises them into a single
investment brief with a clear recommendation.
"""

import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

LLM_MODEL = os.getenv("AGENT_LLM", "granite3.3:8b")

_SYNTHESIS_PROMPT = PromptTemplate.from_template(
    """You are a senior portfolio manager writing a final investment brief.

Ticker: {ticker}

--- SIGNAL SCORE ---
{signal_score}

--- SIGNAL ALIGNMENT ---
{signal_alignment}

--- INVESTMENT STANCE (Tree-of-Thoughts) ---
{tot_stance}

--- OODA DECISION ---
{ooda_decision}

Synthesise all of the above into a concise investment brief with:
1. EXECUTIVE SUMMARY (2-3 sentences).
2. RECOMMENDED ACTION: <BUY | HOLD | SELL | SHORT> with position sizing guidance.
3. CONVICTION LEVEL: <HIGH | MEDIUM | LOW> with one-line justification.
4. KEY RISKS (bullet list, max 4 items).
5. MONITORING TRIGGERS (bullet list, max 3 items).
"""
)


def build_synthesis_chain():
    """Build and return the synthesis LCEL chain.

    Input: dict with keys:
      ticker, signal_score, signal_alignment, tot_stance, ooda_decision.
    Output: str — the final investment brief.
    """
    llm = ChatOllama(model=LLM_MODEL, temperature=0.2)
    return _SYNTHESIS_PROMPT | llm | StrOutputParser()


if __name__ == "__main__":
    print("=== synthesis_chain local test ===")
    chain = build_synthesis_chain()
    result = chain.invoke({
        "ticker": "NVDA",
        "signal_score": "Total: 82/100  Signal: BUY",
        "signal_alignment": "Composite Score: 0.6  Alignment: BULL",
        "tot_stance": (
            "Thesis: NVDA is the dominant AI accelerator supplier with a multi-year "
            "demand runway driven by hyperscaler GPU build-outs.\n"
            "Risks: Export controls, valuation stretch.\n"
            "Triggers: H100 export ban expansion, margin compression below 70%."
        ),
        "ooda_decision": "Add 2% position on pullbacks; stop-loss at -15% from entry.",
    })
    print(result)

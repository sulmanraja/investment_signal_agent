"""ToT Stance chain (LCEL).

Bridges the Tree-of-Thoughts beam search result into a final stance
narrative. Takes the best ThoughtNode from beam search and formats it
as a structured investment stance for downstream synthesis.
"""

import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from tot.beam_search import run_beam_search, BEAM_WIDTH

LLM_MODEL = os.getenv("AGENT_LLM", "granite3.3:8b")
MAX_DEPTH = int(os.getenv("TOT_MAX_DEPTH", "2"))

_STANCE_PROMPT = PromptTemplate.from_template(
    """You are a portfolio strategist formalising an investment stance.

Ticker: {ticker}
Best thesis branch ({branch}):
{content}

Rewrite this as a formal investment stance in three paragraphs:
1. Thesis statement (what and why).
2. Key risks that could invalidate this stance.
3. Monitoring triggers (what would change your view).
"""
)


def build_tot_stance_chain():
    """Return an LCEL chain that runs ToT beam search then formats the best node.

    Input: dict with keys "ticker" and "context".
    Output: str with formatted stance narrative.
    """
    llm = ChatOllama(model=LLM_MODEL, temperature=0.3)
    format_chain = _STANCE_PROMPT | llm | StrOutputParser()

    def run(inputs: dict) -> str:
        best = run_beam_search(
            inputs["ticker"],
            inputs["context"],
            max_depth=MAX_DEPTH,
            beam_width=BEAM_WIDTH,
        )
        return format_chain.invoke({
            "ticker": best.ticker,
            "branch": best.branch,
            "content": best.content,
        })

    return RunnableLambda(run)


if __name__ == "__main__":
    print("=== tot_stance_chain local test ===")
    chain = build_tot_stance_chain()
    result = chain.invoke({
        "ticker": "NVDA",
        "context": (
            "NVDA Q1 FY2025: data-center revenue $22.6B (+427% YoY). "
            "Blackwell GPU demand described as 'insane'. Fed funds rate 5.25%. "
            "Export control risk on H100 chips to China."
        ),
    })
    print(result)

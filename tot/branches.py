"""Branch generator — expands an investment context into A/B/C/D stances.

A = Bullish  |  B = Bearish  |  C = Neutral  |  D = Contrarian
"""

import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from tot.thought_node import ThoughtNode

LLM_MODEL = os.getenv("AGENT_LLM", "granite3.3:8b")

_BRANCH_PROMPT = PromptTemplate.from_template(
    """You are an investment strategist generating four distinct thesis stances.

Given this context for {ticker}:
{context}

Generate four investment stances (2-3 sentences each):
A) BULLISH: The strongest bull case.
B) BEARISH: The strongest bear case.
C) NEUTRAL: The wait-and-see / hold case.
D) CONTRARIAN: A non-consensus view with supporting logic.

Format your response exactly as:
A) <text>
B) <text>
C) <text>
D) <text>
"""
)

BRANCH_LABELS = {"A": "Bullish", "B": "Bearish", "C": "Neutral", "D": "Contrarian"}


def _parse_branches(raw: str, ticker: str, depth: int) -> list[ThoughtNode]:
    """Parse the LLM's A/B/C/D output into ThoughtNode objects."""
    nodes = []
    for label in ["A", "B", "C", "D"]:
        start = raw.find(f"{label})")
        if start == -1:
            continue
        next_labels = [f"{l})" for l in ["A", "B", "C", "D"] if l != label]
        end = len(raw)
        for nl in next_labels:
            idx = raw.find(nl, start + 3)
            if idx != -1 and idx < end:
                end = idx
        content = raw[start + 2:end].strip()
        nodes.append(ThoughtNode(ticker=ticker, branch=label, depth=depth, content=content))
    return nodes


def generate_branches(ticker: str, context: str, depth: int = 0) -> list[ThoughtNode]:
    """Generate the four root investment branches for a ticker.

    Args:
        ticker: Stock ticker symbol.
        context: Aggregated signal context string.
        depth: Tree depth for these nodes (0 for initial generation).

    Returns:
        List of four ThoughtNode objects (A/B/C/D).
    """
    llm = ChatOllama(model=LLM_MODEL, temperature=0.6)
    chain = _BRANCH_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({"ticker": ticker, "context": context})
    return _parse_branches(raw, ticker, depth)


if __name__ == "__main__":
    print("=== branches local test ===")
    ctx = (
        "NVDA Q1 FY2025: data-center revenue $22.6B (+427% YoY). "
        "Blackwell GPU demand described as 'insane'. "
        "Fed funds rate 5.25%. Yield curve inverted. Export controls risk on H100/A100."
    )
    nodes = generate_branches("NVDA", ctx)
    for n in nodes:
        print(f"\n{n.branch}) {BRANCH_LABELS[n.branch]}:")
        print(f"  {n.content}")

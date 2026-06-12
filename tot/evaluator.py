"""Evaluator — scores ThoughtNode instances using a structured rubric.

Rubric dimensions (each 0-2, total max 10):
  1. Evidence quality      — is the claim grounded in data/facts?
  2. Logical consistency   — is the argument internally coherent?
  3. Risk awareness        — are key risks acknowledged?
  4. Signal alignment      — does it align with the aggregated signals?
  5. Actionability         — does it lead to a clear investment decision?
"""

import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from tot.thought_node import ThoughtNode

LLM_MODEL = os.getenv("AGENT_LLM", "granite3.3:8b")

_EVAL_PROMPT = PromptTemplate.from_template(
    """You are a rigorous investment thesis evaluator.

Score the following investment stance on five dimensions (0-2 each, total max 10):
1. Evidence quality      — is the claim grounded in data/facts?
2. Logical consistency   — is the argument internally coherent?
3. Risk awareness        — are key risks acknowledged?
4. Signal alignment      — does it align with the provided context?
5. Actionability         — does it lead to a clear investment decision?

Context for {ticker}:
{context}

Stance ({branch}):
{content}

Respond with ONLY a JSON object in this exact format:
{{"evidence": <int>, "logic": <int>, "risk": <int>, "alignment": <int>, "actionability": <int>, "total": <int>, "rationale": "<one sentence>"}}
"""
)


def evaluate_node(node: ThoughtNode, context: str) -> ThoughtNode:
    """Score a ThoughtNode and set node.score in place.

    Args:
        node: The ThoughtNode to evaluate.
        context: The aggregated signal context used for alignment scoring.

    Returns:
        The same node with score and evaluation metadata set.
    """
    llm = ChatOllama(model=LLM_MODEL, temperature=0.1)
    chain = _EVAL_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({
        "ticker": node.ticker,
        "branch": node.branch,
        "content": node.content,
        "context": context,
    })

    import json, re
    try:
        # Extract JSON even if the model adds prose around it
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            scores = json.loads(match.group())
            node.score = float(scores.get("total", 5))
        else:
            node.score = 5.0
    except (json.JSONDecodeError, ValueError):
        node.score = 5.0

    return node


if __name__ == "__main__":
    from tot.branches import generate_branches
    print("=== evaluator local test ===")
    ctx = (
        "NVDA Q1 FY2025: data-center revenue $22.6B (+427% YoY). "
        "Blackwell demand described as 'insane'. Fed funds rate 5.25%."
    )
    nodes = generate_branches("NVDA", ctx)
    for node in nodes:
        evaluate_node(node, ctx)
        print(f"  {node.branch}) score={node.score:.1f}  {node.content[:60]}…")

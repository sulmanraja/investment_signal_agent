"""Branch generator — expands a signal context into the four canonical branches.

Four canonical investment stances (per claud.md spec):
  A — Capital-Led:          Large capital commitments → structural demand
  B — Adoption-Led:         Developer/enterprise adoption → sustained growth
  C — Risk-Adjusted:        Macro/regulatory/competitive headwinds dominate
  D — Evidence-Insufficient: Contradictory signals → escalate to manual review

Each node MUST include a non-trivial counter_evidence field.
"""

import os
import json
import re
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from tot.thought_node import ThoughtNode, BRANCH_LABELS

LLM_MODEL = os.getenv("AGENT_LLM", "granite3.3:8b")

_BRANCH_PROMPT = PromptTemplate.from_template(
    """You are a technology investment strategist generating four canonical thesis branches.

Technology category: {ticker}
Signal context:
{context}

Generate one thesis node per branch. Each must include a non-trivial counter-argument
that genuinely challenges the stance.

Respond with ONLY valid JSON:
{{
  "nodes": [
    {{
      "branch": "A",
      "content": "<Capital-Led: large capital commitments signal durable structural demand — 2-3 sentences>",
      "counter_evidence": "<specific evidence that would disprove Branch A>"
    }},
    {{
      "branch": "B",
      "content": "<Adoption-Led: developer/enterprise adoption predicts sustained growth — 2-3 sentences>",
      "counter_evidence": "<specific evidence that would disprove Branch B>"
    }},
    {{
      "branch": "C",
      "content": "<Risk-Adjusted: macro/regulatory/competitive headwinds dominate near-term — 2-3 sentences>",
      "counter_evidence": "<specific evidence that would disprove Branch C>"
    }},
    {{
      "branch": "D",
      "content": "<Evidence-Insufficient: signals are irreconcilable; manual review required — 2-3 sentences>",
      "counter_evidence": "<what data would resolve the ambiguity>"
    }}
  ]
}}
"""
)


def generate_branches(ticker: str, context: str, depth: int = 0) -> list[ThoughtNode]:
    """Generate the four canonical investment branches for a category.

    Args:
        ticker: Technology category ID or stock ticker.
        context: Aggregated signal context string.
        depth: Tree depth for these nodes (0 for initial generation).

    Returns:
        List of four ThoughtNode objects (A/B/C/D).
    """
    llm = ChatOllama(model=LLM_MODEL, temperature=0.6)
    chain = _BRANCH_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({"ticker": ticker, "context": context})

    nodes = []
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            for item in data.get("nodes", []):
                branch = item.get("branch", "D")
                if branch not in ("A", "B", "C", "D"):
                    continue
                nodes.append(ThoughtNode(
                    ticker=ticker,
                    branch=branch,
                    depth=depth,
                    content=item.get("content", ""),
                    counter_evidence=item.get("counter_evidence", ""),
                ))
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: ensure all four branches exist
    existing = {n.branch for n in nodes}
    for branch in ["A", "B", "C", "D"]:
        if branch not in existing:
            nodes.append(ThoughtNode(
                ticker=ticker, branch=branch, depth=depth,
                content=f"[Branch {branch} — generation failed]",
                counter_evidence="",
            ))

    return sorted(nodes, key=lambda n: n.branch)


if __name__ == "__main__":
    print("=== branches local test ===")
    ctx = (
        "AI/ML Infrastructure — SEC: NVDA data-center revenue +427% YoY. "
        "GitHub: CUDA repos +3,400 in 90 days. Macro: Fed funds 5.25%, "
        "yield curve inverted. Export controls on H100/A100 chips."
    )
    nodes = generate_branches("ai_ml_infrastructure", ctx)
    for n in nodes:
        print(f"\n{n.branch}) {BRANCH_LABELS[n.branch]}:")
        print(f"   Thesis:  {n.content[:120]}")
        print(f"   Counter: {n.counter_evidence[:100]}")

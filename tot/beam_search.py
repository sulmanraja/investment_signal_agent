"""Beam search over the Tree-of-Thoughts (LCEL chain).

Expands investment thesis branches depth-first, evaluates each node with
the scoring rubric, prunes to beam_width=2 at each level, and returns
the best surviving node as the recommended thesis.
"""

import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from tot.thought_node import ThoughtNode
from tot.branches import generate_branches
from tot.evaluator import evaluate_node
from tot.pruner import prune_beam, BEAM_WIDTH

MAX_DEPTH = int(os.getenv("TOT_MAX_DEPTH", "2"))
LLM_MODEL = os.getenv("AGENT_LLM", "granite3.3:8b")

_EXPAND_PROMPT = PromptTemplate.from_template(
    """You are an investment analyst deepening a thesis.

Ticker: {ticker}
Context: {context}
Current stance ({branch}): {content}

Elaborate and strengthen this stance in 2-3 sentences. Add one specific
supporting data point or risk factor not already mentioned.
"""
)


def _expand_node(node: ThoughtNode, context: str) -> ThoughtNode:
    """Expand a node by elaborating its content one depth deeper."""
    llm = ChatOllama(model=LLM_MODEL, temperature=0.5)
    chain = _EXPAND_PROMPT | llm | StrOutputParser()
    elaboration = chain.invoke({
        "ticker": node.ticker,
        "branch": node.branch,
        "content": node.content,
        "context": context,
    })
    child = ThoughtNode(
        ticker=node.ticker,
        branch=node.branch,
        depth=node.depth + 1,
        content=elaboration.strip(),
        parent=node,
    )
    node.children.append(child)
    return child


def run_beam_search(
    ticker: str,
    context: str,
    max_depth: int = MAX_DEPTH,
    beam_width: int = BEAM_WIDTH,
) -> ThoughtNode:
    """Run beam-search ToT and return the best leaf node.

    Args:
        ticker: Stock ticker symbol.
        context: Aggregated investment signal context.
        max_depth: How many elaboration levels to run beyond the root.
        beam_width: Number of nodes to keep per level.

    Returns:
        The highest-scoring leaf ThoughtNode after beam search.
    """
    print(f"[ToT] Generating root branches for {ticker} …")
    nodes = generate_branches(ticker, context, depth=0)

    for node in nodes:
        evaluate_node(node, context)
        print(f"  Root {node.branch}) score={node.score:.1f}")

    beam = prune_beam(nodes, beam_width)
    print(f"[ToT] Beam after depth 0: {[n.branch for n in beam]}")

    for depth in range(1, max_depth + 1):
        expanded = [_expand_node(n, context) for n in beam]
        for node in expanded:
            evaluate_node(node, context)
            print(f"  Depth {depth} {node.branch}) score={node.score:.1f}")
        beam = prune_beam(expanded, beam_width)
        print(f"[ToT] Beam after depth {depth}: {[n.branch for n in beam]}")

    best = max(beam, key=lambda n: n.score or 0.0)
    print(f"[ToT] Best node: {best}")
    return best


# LCEL-compatible runnable wrapper
def build_tot_chain():
    """Return an LCEL-compatible runnable that accepts {'ticker', 'context'}."""
    from langchain_core.runnables import RunnableLambda
    return RunnableLambda(lambda inputs: run_beam_search(inputs["ticker"], inputs["context"]))


if __name__ == "__main__":
    print("=== beam_search local test ===")
    ctx = (
        "NVDA Q1 FY2025: data-center revenue $22.6B (+427% YoY). "
        "Blackwell demand described as 'insane'. Fed funds rate 5.25%. "
        "Export control risk on H100 chips to China."
    )
    best = run_beam_search("NVDA", ctx, max_depth=1, beam_width=2)
    print(f"\nRecommended stance: {best.branch}")
    print(best.content)

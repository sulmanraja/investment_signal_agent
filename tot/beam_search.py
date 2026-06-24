"""Beam search over the Tree-of-Thoughts.

Termination conditions (per claud.md spec):
  1. clear_winner: top survivor scores ≥ 85 AND leads second-best by ≥ 15 pts.
  2. single_survivor: only one node remains after pruning.
  3. max_depth: depth limit reached (MAX_DEPTH = 2).
  4. branch_d_auto: zero survivors after floor/beam → Branch D auto-promoted.

Child expansion includes counter_evidence so the Critic gate applies at depth 1+.
"""

import os
import json
import re
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from tot.thought_node import ThoughtNode, BRANCH_LABELS
from tot.branches import generate_branches
from tot.evaluator import evaluate_node
from tot.pruner import prune_beam, BEAM_WIDTH, FLOOR_SCORE

MAX_DEPTH = int(os.getenv("TOT_MAX_DEPTH", "2"))
CLEAR_WINNER_THRESHOLD = 85.0
CLEAR_WINNER_MARGIN = 15.0
LLM_MODEL = os.getenv("AGENT_LLM", "granite3.3:8b")

_EXPAND_PROMPT = PromptTemplate.from_template(
    """You are an investment analyst deepening a thesis for technology category '{ticker}'.

Signal context:
{context}

Parent thesis (Branch {branch} — {branch_label}, depth {depth}):
{content}

Your task: elaborate and strengthen this stance in 2-3 sentences, adding at least
one specific data point or risk factor not mentioned above. Also provide a
non-trivial counter-argument that genuinely challenges this refined stance.

Respond with ONLY valid JSON:
{{"content": "<refined thesis — 2-3 sentences>", "counter_evidence": "<specific counter-argument>"}}
"""
)


def _expand_node(node: ThoughtNode, context: str) -> ThoughtNode:
    """Expand a node into a child by elaborating its content one depth deeper."""
    llm = ChatOllama(model=LLM_MODEL, temperature=0.5)
    chain = _EXPAND_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({
        "ticker": node.ticker,
        "branch": node.branch,
        "branch_label": BRANCH_LABELS.get(node.branch, node.branch),
        "depth": node.depth,
        "content": node.content,
        "context": context,
    })

    content = node.content + " [refined]"
    counter_evidence = ""
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            content = data.get("content", content).strip()
            counter_evidence = data.get("counter_evidence", "").strip()
    except (json.JSONDecodeError, ValueError):
        pass

    child = ThoughtNode(
        ticker=node.ticker,
        branch=node.branch,
        depth=node.depth + 1,
        content=content,
        counter_evidence=counter_evidence,
        parent=node,
    )
    node.children.append(child)
    return child


def _check_termination(
    survivors: list[ThoughtNode],
    depth: int,
    branch_d_promoted: bool,
    max_depth: int,
) -> tuple[bool, str]:
    """Check all termination conditions. Returns (should_stop, reason)."""
    if branch_d_promoted:
        return True, "branch_d_auto"
    if not survivors:
        return True, "no_survivors"
    if len(survivors) == 1:
        return True, "single_survivor"
    if depth >= max_depth:
        return True, "max_depth"
    top_score = survivors[0].score or 0.0
    second_score = survivors[1].score or 0.0 if len(survivors) > 1 else 0.0
    if top_score >= CLEAR_WINNER_THRESHOLD and (top_score - second_score) >= CLEAR_WINNER_MARGIN:
        return True, "clear_winner"
    return False, ""


def run_beam_search(
    ticker: str,
    context: str,
    max_depth: int = MAX_DEPTH,
    beam_width: int = BEAM_WIDTH,
) -> tuple[ThoughtNode, list[ThoughtNode], str]:
    """Run beam-search ToT and return the winner.

    Args:
        ticker: Technology category ID.
        context: Aggregated investment signal context.
        max_depth: Depth limit (usually 2).
        beam_width: Max survivors per level.

    Returns:
        (winner, pruned_all, termination_reason)
        - winner: The highest-scoring surviving ThoughtNode.
        - pruned_all: All pruned nodes across all depths (for Critic logs).
        - termination_reason: One of clear_winner / single_survivor /
                              max_depth / branch_d_auto / no_survivors.
    """
    print(f"[ToT] Generating root branches for '{ticker}' …")
    nodes = generate_branches(ticker, context, depth=0)

    for node in nodes:
        evaluate_node(node, context)
        score_str = f"{node.score:.1f}" if node.score is not None else "?"
        print(f"  Root {node.branch}({node.branch_label()}) score={score_str}")

    survivors, pruned_all, d_promoted = prune_beam(nodes, beam_width, FLOOR_SCORE)
    print(f"[ToT] Beam d=0: survivors={[n.branch for n in survivors]}  d_promoted={d_promoted}")

    stop, reason = _check_termination(survivors, 0, d_promoted, max_depth)
    if stop:
        winner = survivors[0] if survivors else max(nodes, key=lambda n: n.score or 0.0)
        print(f"[ToT] Early termination at d=0 — {reason}")
        return winner, pruned_all, reason

    for depth in range(1, max_depth + 1):
        expanded = [_expand_node(n, context) for n in survivors]
        for node in expanded:
            evaluate_node(node, context)
            score_str = f"{node.score:.1f}" if node.score is not None else "?"
            print(f"  Depth {depth} {node.branch}({node.branch_label()}) score={score_str}")

        survivors, newly_pruned, d_promoted = prune_beam(expanded, beam_width, FLOOR_SCORE)
        pruned_all = pruned_all + newly_pruned
        print(f"[ToT] Beam d={depth}: survivors={[n.branch for n in survivors]}  d_promoted={d_promoted}")

        stop, reason = _check_termination(survivors, depth, d_promoted, max_depth)
        if stop:
            winner = survivors[0] if survivors else max(expanded, key=lambda n: n.score or 0.0)
            print(f"[ToT] Termination at d={depth} — {reason}")
            return winner, pruned_all, reason

    winner = max(survivors, key=lambda n: n.score or 0.0)
    return winner, pruned_all, "max_depth"


def build_tot_chain():
    """Return an LCEL-compatible runnable that accepts {'ticker', 'context'}."""
    from langchain_core.runnables import RunnableLambda
    return RunnableLambda(
        lambda inputs: run_beam_search(inputs["ticker"], inputs["context"])
    )


if __name__ == "__main__":
    print("=== beam_search local test ===")
    ctx = (
        "AI/ML Infrastructure — SEC: NVDA data-center revenue $22.6B (+427% YoY). "
        "GitHub: CUDA repos +3,400 in 90 days. News sentiment: 0.72. "
        "Macro: Fed funds 5.25%, yield curve inverted. "
        "Export controls on H100/A100 chips to China restrict TAM."
    )
    winner, pruned, reason = run_beam_search("ai_ml_infrastructure", ctx, max_depth=1)
    print(f"\n[Winner] Branch {winner.branch} ({winner.branch_label()}) — {reason}")
    print(f"  Score: {winner.score:.1f}/100")
    print(f"  Thesis: {winner.content[:160]}")
    print(f"\n[Pruned] {len(pruned)} node(s) eliminated:")
    for n in pruned:
        print(f"  {n.branch} d={n.depth} score={n.score}")

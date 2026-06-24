"""Evaluator — scores ThoughtNode instances using the five-criterion rubric.

Five criteria (0-20 each, 100 max — per claud.md spec):
  1. Evidence Alignment      — claim grounded in signal data
  2. Internal Consistency    — argument is coherent
  3. Macro Compatibility     — accounts for macro/regulatory environment
  4. Actionability           — leads to a clear Buy/Hold/Reduce decision
  5. Confidence Calibration  — certainty appropriate given evidence

Counter-evidence gate: if counter_evidence is trivial/empty, Evidence Alignment
is forced to 0 before the LLM scores the other four dimensions.
"""

import os
import json
import re
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from tot.thought_node import ThoughtNode

LLM_MODEL = os.getenv("AGENT_LLM", "granite3.3:8b")

_EVAL_PROMPT = PromptTemplate.from_template(
    """You are an independent investment thesis evaluator. Score the thesis below
on five criteria (0-20 each, 100 max). Do not reference any other branch.

Signal context for {ticker}:
{context}

Branch {branch} — {branch_label}:
Thesis: {content}
Counter-evidence: {counter_evidence}

Score each criterion:
1. Evidence Alignment      (0-20): Grounded in the signal data above?
2. Internal Consistency    (0-20): Coherent without internal contradictions?
3. Macro Compatibility     (0-20): Accounts for macro/regulatory environment?
4. Actionability           (0-20): Leads to a clear Buy/Hold/Reduce decision?
5. Confidence Calibration  (0-20): Certainty appropriate given the evidence?

Respond with ONLY valid JSON:
{{"evidence_alignment":<int>,"internal_consistency":<int>,"macro_compatibility":<int>,"actionability":<int>,"confidence_calibration":<int>,"total":<sum>,"key_weakness":"<one sentence>"}}
"""
)

_TRIVIAL_MARKERS = ["", "n/a", "none", "no counter", "__trivial__", "unable to parse"]


def _is_trivial(text: str) -> bool:
    clean = text.strip().lower()
    return len(clean) < 20 or any(clean.startswith(m) for m in _TRIVIAL_MARKERS)


def evaluate_node(node: ThoughtNode, context: str) -> ThoughtNode:
    """Score a ThoughtNode against the five-criterion rubric.

    Counter-evidence gate: Evidence Alignment is forced to 0 if trivial.
    """
    from tot.thought_node import BRANCH_LABELS
    evidence_zeroed = _is_trivial(node.counter_evidence)

    llm = ChatOllama(model=LLM_MODEL, temperature=0.1)
    chain = _EVAL_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({
        "ticker": node.ticker,
        "branch": node.branch,
        "branch_label": BRANCH_LABELS.get(node.branch, node.branch),
        "content": node.content,
        "counter_evidence": node.counter_evidence if not evidence_zeroed else "(MISSING — gated)",
        "context": context,
    })

    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            ev = 0 if evidence_zeroed else int(data.get("evidence_alignment", 0))
            node.score = (
                ev
                + int(data.get("internal_consistency", 0))
                + int(data.get("macro_compatibility", 0))
                + int(data.get("actionability", 0))
                + int(data.get("confidence_calibration", 0))
            )
        else:
            node.score = 0.0 if evidence_zeroed else 40.0
    except (json.JSONDecodeError, ValueError):
        node.score = 0.0 if evidence_zeroed else 40.0

    return node


if __name__ == "__main__":
    from tot.branches import generate_branches
    print("=== evaluator local test ===")
    ctx = (
        "AI/ML Infrastructure — SEC: NVDA data-center +427% YoY. "
        "GitHub: CUDA repos +3,400 in 90d. "
        "Macro: Fed funds 5.25%, yield curve inverted."
    )
    nodes = generate_branches("ai_ml_infrastructure", ctx)
    for node in nodes:
        evaluate_node(node, ctx)
        print(f"  Branch {node.branch} ({node.branch_label()})  score={node.score:.0f}/100  {node.content[:60]}…")

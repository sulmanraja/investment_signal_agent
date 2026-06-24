"""Thought Generator Agent (CrewAI — ToT branch generator).

Invoked ONLY for CONTRADICTORY categories. Produces the four canonical
investment-thesis branch hypotheses. Must NOT see or be influenced by
Critic scores during generation (separation enforced by the Orchestrator).

Four canonical branches (per claud.md spec):
  A — Capital-Led:         Large capital commitments from leaders indicate
                            durable structural demand (12-18 month horizon).
  B — Adoption-Led:        Developer and enterprise adoption patterns predict
                            sustained category growth regardless of near-term spend.
  C — Risk-Adjusted:       Macro, regulatory, or competitive headwinds dominate
                            the near-term outlook; caution warranted.
  D — Evidence-Insufficient: Contradictory signals cannot be reconciled; requires
                            escalation to manual expert review.

Each node MUST include a non-trivial counter_evidence field — the Orchestrator
validates this before sending nodes to the Critic.
"""

import asyncio
import os
import json
import re
from crewai import Agent, Task, Crew, LLM

from schemas.messages import (
    GenerationRequest,
    ThoughtNodeSchema,
    ThoughtNodesReport,
    BranchType,
)

LLM_MODEL = os.getenv("AGENT_LLM", "granite3.3:8b")

_BRANCH_DESCRIPTIONS = {
    BranchType.CAPITAL_LED:            "Capital-Led: large capital commitments signal durable structural demand",
    BranchType.ADOPTION_LED:           "Adoption-Led: developer/enterprise adoption predicts sustained growth",
    BranchType.RISK_ADJUSTED:          "Risk-Adjusted: macro/regulatory/competitive headwinds dominate near-term",
    BranchType.EVIDENCE_INSUFFICIENT:  "Evidence-Insufficient: contradictory signals require manual expert review",
}

_ROOT_GENERATION_PROMPT = """You are an investment thesis strategist analysing a CONTRADICTORY technology category.

Category: {category_label}
Signal Context (no Critic scores — do not reference any prior evaluations):
{context}

Generate exactly four investment thesis nodes, one per canonical branch.
For each branch write 2-3 sentences of thesis, then a non-trivial counter-argument
that could genuinely invalidate the stance.

Respond in valid JSON only, with this exact structure:
{{
  "nodes": [
    {{
      "branch": "A",
      "content": "<Capital-Led thesis — 2-3 sentences>",
      "counter_evidence": "<Non-trivial counter-argument that could disprove Branch A>"
    }},
    {{
      "branch": "B",
      "content": "<Adoption-Led thesis — 2-3 sentences>",
      "counter_evidence": "<Non-trivial counter-argument that could disprove Branch B>"
    }},
    {{
      "branch": "C",
      "content": "<Risk-Adjusted thesis — 2-3 sentences>",
      "counter_evidence": "<Non-trivial counter-argument that could disprove Branch C>"
    }},
    {{
      "branch": "D",
      "content": "<Evidence-Insufficient stance — 2-3 sentences explaining why signals are irreconcilable>",
      "counter_evidence": "<What would resolve the ambiguity and allow a definitive stance>"
    }}
  ]
}}
"""

_CHILD_REFINEMENT_PROMPT = """You are an investment thesis strategist deepening a surviving branch.

Category: {category_label}
Signal Context:
{context}

Parent branch ({branch_label}):
{parent_content}

Elaborate and strengthen this thesis by:
1. Adding one specific data point or mechanism not already mentioned.
2. Sharpening the actionability (what exactly should an engineering leader do?).
3. Updating the counter_evidence to reflect any new information.

Respond in valid JSON only:
{{
  "content": "<refined thesis — 2-3 sentences>",
  "counter_evidence": "<updated non-trivial counter-argument>"
}}
"""


def _parse_root_nodes(raw: str, request: GenerationRequest) -> list[ThoughtNodeSchema]:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group())
        nodes = []
        branch_map = {"A": BranchType.CAPITAL_LED, "B": BranchType.ADOPTION_LED,
                      "C": BranchType.RISK_ADJUSTED, "D": BranchType.EVIDENCE_INSUFFICIENT}
        for n in data.get("nodes", []):
            branch = branch_map.get(n.get("branch", "D"), BranchType.EVIDENCE_INSUFFICIENT)
            nodes.append(ThoughtNodeSchema(
                branch=branch,
                depth=0,
                content=n.get("content", ""),
                counter_evidence=n.get("counter_evidence", ""),
            ))
        return nodes
    except json.JSONDecodeError:
        return []


def _parse_child_node(raw: str, parent: ThoughtNodeSchema) -> ThoughtNodeSchema | None:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
        return ThoughtNodeSchema(
            branch=parent.branch,
            depth=parent.depth + 1,
            content=data.get("content", ""),
            counter_evidence=data.get("counter_evidence", ""),
            parent_branch=parent.branch,
        )
    except json.JSONDecodeError:
        return None


class ThoughtGeneratorAgent:
    """CrewAI-backed thought generator producing canonical ToT branch nodes."""

    name = "ThoughtGeneratorAgent"

    def __init__(self):
        llm = LLM(model=f"ollama/{LLM_MODEL}", temperature=0.7)
        self._agent = Agent(
            role="Investment Thesis Branch Generator",
            goal=(
                "Generate four canonical investment thesis branches for a CONTRADICTORY "
                "technology category. Each branch must include a non-trivial counter-argument. "
                "Never reference Critic scores or prior evaluations."
            ),
            backstory=(
                "You are a senior portfolio strategist who specialises in scenario analysis "
                "for technology investments. You are rigorous about surfacing counter-arguments "
                "before committing to any investment thesis."
            ),
            llm=llm,
            verbose=False,
            allow_delegation=False,
        )

    async def run(self, request: GenerationRequest) -> ThoughtNodesReport:
        if request.depth == 0:
            return await self._generate_root_nodes(request)
        return await self._generate_child_nodes(request)

    async def _generate_root_nodes(self, request: GenerationRequest) -> ThoughtNodesReport:
        prompt = _ROOT_GENERATION_PROMPT.format(
            category_label=request.category_label,
            context=request.context,
        )
        task = Task(
            description=prompt,
            agent=self._agent,
            expected_output="JSON object with 'nodes' array containing four branch objects.",
        )
        crew = Crew(agents=[self._agent], tasks=[task], verbose=False)
        raw = str(await crew.kickoff_async())
        nodes = _parse_root_nodes(raw, request)

        # Fallback: if parsing fails, create stub nodes
        if not nodes:
            nodes = [
                ThoughtNodeSchema(
                    branch=b, depth=0,
                    content=f"[Parse failure — raw output truncated] {raw[:200]}",
                    counter_evidence="Unable to parse — requires manual review.",
                )
                for b in [BranchType.CAPITAL_LED, BranchType.ADOPTION_LED,
                           BranchType.RISK_ADJUSTED, BranchType.EVIDENCE_INSUFFICIENT]
            ]

        print(f"  [ThoughtGenerator] Generated {len(nodes)} root nodes for '{request.category_label}'")
        return ThoughtNodesReport(
            run_id=request.run_id,
            category_id=request.category_id,
            depth=0,
            nodes=nodes,
        )

    async def _generate_child_nodes(self, request: GenerationRequest) -> ThoughtNodesReport:
        llm = LLM(model=f"ollama/{LLM_MODEL}", temperature=0.5)
        branch_map = {"A": BranchType.CAPITAL_LED, "B": BranchType.ADOPTION_LED,
                      "C": BranchType.RISK_ADJUSTED, "D": BranchType.EVIDENCE_INSUFFICIENT}
        child_nodes = []
        for branch_label in request.survivor_branches:
            branch = branch_map.get(branch_label, BranchType.EVIDENCE_INSUFFICIENT)
            prompt = _CHILD_REFINEMENT_PROMPT.format(
                category_label=request.category_label,
                context=request.context,
                branch_label=_BRANCH_DESCRIPTIONS[branch],
                parent_content="[parent content from prior round]",
            )
            task = Task(
                description=prompt,
                agent=self._agent,
                expected_output="JSON with 'content' and 'counter_evidence' fields.",
            )
            crew = Crew(agents=[self._agent], tasks=[task], verbose=False)
            raw = str(await crew.kickoff_async())
            parent_stub = ThoughtNodeSchema(
                branch=branch, depth=request.depth - 1,
                content="", counter_evidence="",
            )
            child = _parse_child_node(raw, parent_stub)
            if child:
                child_nodes.append(child)

        print(f"  [ThoughtGenerator] Generated {len(child_nodes)} child nodes (depth={request.depth})")
        return ThoughtNodesReport(
            run_id=request.run_id,
            category_id=request.category_id,
            depth=request.depth,
            nodes=child_nodes,
        )


if __name__ == "__main__":
    print("=== thought_generator_agent local test ===")
    req = GenerationRequest(
        run_id="test-001",
        category_id="ai_ml_infrastructure",
        category_label="AI / ML Infrastructure",
        context=(
            "SEC: NVDA data-center revenue +427% YoY. AMD AI revenue +80% YoY.\n"
            "GitHub: CUDA repos +3,400 in 90 days (strong adoption).\n"
            "News: Mixed — Blackwell on schedule; US export controls on H100/A100.\n"
            "FRED: Fed funds 5.25%, yield curve inverted -0.3.\n"
            "Google Trends: 'NVIDIA GPU' interest +62% in 3 months."
        ),
        depth=0,
    )
    agent = ThoughtGeneratorAgent()
    report = asyncio.run(agent.run(req))
    for node in report.nodes:
        print(f"\n  Branch {node.branch.value}: {node.content[:100]}…")
        print(f"  Counter: {node.counter_evidence[:80]}…")

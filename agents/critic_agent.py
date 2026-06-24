"""Critic Agent (CrewAI — ToT node evaluator).

Invoked ONLY for CONTRADICTORY categories. Scores each thought node
independently using the five-criterion rubric. Never receives sibling
node scores — one Critic invocation per node (never batched).

Five-criterion rubric (per claud.md spec), 0-20 each (100 max):
  1. Evidence Alignment      — claim grounded in available signal data
  2. Internal Consistency    — argument is coherent and self-consistent
  3. Macro Compatibility     — accounts for macro/regulatory environment
  4. Actionability           — leads to clear Buy/Hold/Reduce decision
  5. Confidence Calibration  — certainty level is appropriate given evidence

Score-variance monitoring: if all root scores fall within a 10-point band,
a re-evaluation pass is triggered with an augmented prompt identifying the
single most significant weakness per node.

Counter-evidence gate: nodes with empty or vague counter_evidence have
their Evidence Alignment score zeroed before Critic evaluation (enforced
by the Orchestrator, not the Critic itself).
"""

import asyncio
import os
import json
import re
from crewai import Agent, Task, Crew, LLM

from schemas.messages import (
    EvaluationRequest,
    NodeScoreReport,
    ScoreReport,
    BranchType,
)

LLM_MODEL = os.getenv("AGENT_LLM", "granite3.3:8b")

VARIANCE_BAND = 10  # trigger re-evaluation if all scores within this band

_EVAL_PROMPT = """You are a rigorous investment thesis critic evaluating a SINGLE thought node.
Do NOT consider any other branch's argument. Evaluate only what is presented here.

Category: {category_label}
Signal Context:
{context}

Branch ({branch_label}):
Thesis: {content}
Counter-evidence provided by author: {counter_evidence}

Score this node on five criteria (0-20 each, 100 max):
1. Evidence Alignment      (0-20): Is the thesis grounded in the signal data above?
2. Internal Consistency    (0-20): Is the argument internally coherent without contradictions?
3. Macro Compatibility     (0-20): Does it account for the macro/regulatory environment?
4. Actionability           (0-20): Does it clearly lead to a Buy, Hold, or Reduce decision?
5. Confidence Calibration  (0-20): Is the stated certainty appropriate given the evidence?

Respond in valid JSON only:
{{
  "evidence_alignment": <int 0-20>,
  "internal_consistency": <int 0-20>,
  "macro_compatibility": <int 0-20>,
  "actionability": <int 0-20>,
  "confidence_calibration": <int 0-20>,
  "total": <sum>,
  "key_weakness": "<single most significant flaw in this thesis — one sentence>"
}}
"""

_REEVAL_PROMPT = """You are re-evaluating an investment thesis node after detecting low score variance.
All nodes scored within a narrow band — you must now identify and apply a meaningful penalty
for the single most significant weakness before re-scoring.

Category: {category_label}
Signal Context:
{context}

Branch ({branch_label}):
Thesis: {content}
Counter-evidence: {counter_evidence}
Initial key weakness identified: {prior_weakness}

Re-score with a penalty applied for the identified weakness. The re-score total
MUST differ from the initial score by at least 5 points where the weakness is real.

Respond in valid JSON only (same structure as before):
{{
  "evidence_alignment": <int 0-20>,
  "internal_consistency": <int 0-20>,
  "macro_compatibility": <int 0-20>,
  "actionability": <int 0-20>,
  "confidence_calibration": <int 0-20>,
  "total": <sum>,
  "key_weakness": "<updated single most significant flaw>"
}}
"""

_BRANCH_LABELS = {
    BranchType.CAPITAL_LED:           "A — Capital-Led",
    BranchType.ADOPTION_LED:          "B — Adoption-Led",
    BranchType.RISK_ADJUSTED:         "C — Risk-Adjusted",
    BranchType.EVIDENCE_INSUFFICIENT: "D — Evidence-Insufficient",
}


def _parse_scores(raw: str, branch: BranchType, depth: int,
                  counter_evidence_zeroed: bool = False) -> NodeScoreReport | None:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
        ev = 0 if counter_evidence_zeroed else int(data.get("evidence_alignment", 0))
        return NodeScoreReport(
            branch=branch,
            depth=depth,
            evidence_alignment=ev,
            internal_consistency=int(data.get("internal_consistency", 0)),
            macro_compatibility=int(data.get("macro_compatibility", 0)),
            actionability=int(data.get("actionability", 0)),
            confidence_calibration=int(data.get("confidence_calibration", 0)),
            total=ev + int(data.get("internal_consistency", 0)) +
                  int(data.get("macro_compatibility", 0)) +
                  int(data.get("actionability", 0)) +
                  int(data.get("confidence_calibration", 0)),
            key_weakness=data.get("key_weakness", ""),
            counter_evidence_zeroed=counter_evidence_zeroed,
        )
    except (json.JSONDecodeError, ValueError):
        return None


class CriticAgent:
    """CrewAI-backed critic that scores each ToT node independently."""

    name = "CriticAgent"

    def __init__(self):
        llm = LLM(model=f"ollama/{LLM_MODEL}", temperature=0.1)
        self._agent = Agent(
            role="Investment Thesis Critic",
            goal=(
                "Score a single investment thesis branch on five criteria (0-20 each, 100 max). "
                "Never reference sibling branches. Be precise and penalise weak evidence."
            ),
            backstory=(
                "You are a risk manager and independent research analyst who stress-tests "
                "investment theses one at a time. You are not influenced by other analysts' "
                "views — you form your own assessment purely from the evidence presented."
            ),
            llm=llm,
            verbose=False,
            allow_delegation=False,
        )

    async def score_node(
        self,
        request: EvaluationRequest,
        category_label: str,
        prior_weakness: str = "",
    ) -> NodeScoreReport | None:
        """Score a single node. One call = one node. Never batched."""
        node = request.node
        branch_label = _BRANCH_LABELS.get(node.branch, str(node.branch))

        template = _REEVAL_PROMPT if request.re_evaluation else _EVAL_PROMPT
        prompt = template.format(
            category_label=category_label,
            context=request.context,
            branch_label=branch_label,
            content=node.content,
            counter_evidence=node.counter_evidence,
            prior_weakness=prior_weakness,
        )

        task = Task(
            description=prompt,
            agent=self._agent,
            expected_output="JSON object with five criterion scores and a key_weakness.",
        )
        crew = Crew(agents=[self._agent], tasks=[task], verbose=False)
        raw = str(await crew.kickoff_async())

        result = _parse_scores(
            raw, node.branch, node.depth,
            counter_evidence_zeroed=False,
        )
        if result:
            print(f"  [Critic] {branch_label}  total={result.total}  "
                  f"weakness='{result.key_weakness[:60]}'")
        return result

    async def score_all_nodes(
        self,
        requests: list[EvaluationRequest],
        category_label: str,
    ) -> ScoreReport:
        """Score a list of nodes one at a time. Triggers re-evaluation if needed."""
        assert len({r.run_id for r in requests}) == 1, "All requests must share run_id"
        assert len({r.category_id for r in requests}) == 1, "All requests must share category_id"

        initial_scores: list[NodeScoreReport] = []
        weaknesses: dict[str, str] = {}

        for req in requests:
            score = await self.score_node(req, category_label)
            if score:
                initial_scores.append(score)
                weaknesses[req.node.branch.value] = score.key_weakness

        # Variance monitoring: re-evaluate if all totals within VARIANCE_BAND
        variance_triggered = False
        if initial_scores:
            totals = [s.total for s in initial_scores]
            if max(totals) - min(totals) <= VARIANCE_BAND:
                print(f"  [Critic] ⚠ Score variance ≤{VARIANCE_BAND} pts — triggering re-evaluation")
                variance_triggered = True
                re_scores: list[NodeScoreReport] = []
                for req, prior in zip(requests, initial_scores):
                    reeval_req = EvaluationRequest(
                        run_id=req.run_id,
                        category_id=req.category_id,
                        node=req.node,
                        context=req.context,
                        re_evaluation=True,
                    )
                    score = await self.score_node(
                        reeval_req, category_label,
                        prior_weakness=weaknesses.get(req.node.branch.value, ""),
                    )
                    re_scores.append(score or prior)
                initial_scores = re_scores

        return ScoreReport(
            run_id=requests[0].run_id,
            category_id=requests[0].category_id,
            node_scores=initial_scores,
            variance_triggered=variance_triggered,
        )


if __name__ == "__main__":
    print("=== critic_agent local test ===")
    from schemas.messages import ThoughtNodeSchema
    node = ThoughtNodeSchema(
        branch=BranchType.CAPITAL_LED,
        depth=0,
        content=(
            "NVDA and AMD capital commitments from hyperscalers indicate durable AI "
            "infrastructure demand through 2026. Data-center revenue trajectories "
            "support a sustained BUY stance for the semiconductor sector."
        ),
        counter_evidence=(
            "US export controls on H100/A100 chips could materially reduce TAM, "
            "and AMD's AI revenue base remains small relative to NVDA."
        ),
    )
    req = EvaluationRequest(
        run_id="test-001",
        category_id="ai_ml_infrastructure",
        node=node,
        context="SEC: NVDA +427% YoY. GitHub CUDA repos surging. Fed funds 5.25%.",
        re_evaluation=False,
    )
    agent = CriticAgent()
    score = asyncio.run(agent.score_node(req, "AI / ML Infrastructure"))
    if score:
        print(f"\n  Total: {score.total}/100")
        print(f"  Weakness: {score.key_weakness}")

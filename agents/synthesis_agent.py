"""Synthesis Agent.

Consumes the final aggregated SynthesisPackage produced by the Orchestrator
and generates the Technology Investment Horizon Report narrative (Markdown).

Constraints (per claud.md spec):
  - Makes NO further tool calls — operates purely on the provided context.
  - Receives the complete package: all category results, ToT reasoning traces,
    pruning logs, emerging candidates, and portfolio deltas.
  - Is structurally isolated from all other agents — no shared state.
"""

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from crewai import Agent, Task, Crew, LLM

from schemas.messages import SynthesisPackage, InvestmentStance, AlignmentClassification

LLM_MODEL = os.getenv("AGENT_LLM", "granite3.3:8b")
REPORTS_DIR = Path(__file__).parent.parent / "output" / "reports"

_SYNTHESIS_PROMPT = """You are a senior technology investment strategist writing a formal report.
You have been given the complete analysis package. Make no additional tool calls —
write purely from the evidence provided below.

Report Date: {cycle_date}

=== CATEGORY RESULTS ===
{category_summary}

=== EMERGING CANDIDATES ===
{emerging_section}

=== PORTFOLIO CHANGES VS PRIOR CYCLE ===
{delta_section}

=== REASONING TRACES (ToT branches that were evaluated) ===
{reasoning_traces}

Write a Technology Investment Horizon Report for engineering leaders with:

1. EXECUTIVE SUMMARY (3-4 sentences covering the overall technology investment climate)

2. CATEGORY STANCES (one subsection per category):
   - Stance: BUY / HOLD / REDUCE  [confidence: HIGH/MEDIUM/LOW]
   - Rationale: 2-3 sentences grounded in the signal evidence
   - Key risk: one sentence
   - [If MANUAL_REVIEW: flag explicitly with the reason]

3. EMERGING OPPORTUNITIES (2-3 sentences per candidate)

4. PORTFOLIO ACTIONS (concrete, actionable steps for an engineering leader)

5. MONITORING TRIGGERS (what would change these stances — 3-5 bullets)

Use crisp, professional language. Avoid hedging more than once per stance.
"""


def _format_category_summary(pkg: SynthesisPackage) -> str:
    lines = []
    for r in pkg.category_results:
        stance_str = r.stance.value
        if r.manual_review_flag:
            stance_str = "MANUAL_REVIEW ⚠"
        lines.append(
            f"  {r.category_label}: {stance_str}  "
            f"(score={r.final_score:.0f}/100  "
            f"path={'ToT' if r.classification == AlignmentClassification.CONTRADICTORY else 'Direct'})"
        )
        if r.reasoning_trace:
            lines.append(f"    Winning branch: {r.winning_branch.value if r.winning_branch else 'N/A'}")
            lines.append(f"    Trace: {r.reasoning_trace[:200]}")
    return "\n".join(lines)


def _format_delta_section(pkg: SynthesisPackage) -> str:
    if not pkg.portfolio_deltas:
        return "  No prior cycle — this is the inaugural report."
    return "\n".join(
        f"  {cat_id}: {delta}" for cat_id, delta in pkg.portfolio_deltas.items()
    )


def _format_emerging(pkg: SynthesisPackage) -> str:
    if not pkg.emerging_candidates:
        return "  No emerging candidates identified this cycle."
    lines = [f"  Candidates: {', '.join(pkg.emerging_candidates)}"]
    if pkg.emerging_rationale:
        lines.append(f"  Rationale: {pkg.emerging_rationale}")
    return "\n".join(lines)


def _format_reasoning_traces(pkg: SynthesisPackage) -> str:
    traces = []
    for r in pkg.category_results:
        if r.classification == AlignmentClassification.CONTRADICTORY and r.pruned_nodes:
            traces.append(f"  {r.category_label} (pruned nodes):")
            for n in r.pruned_nodes:
                traces.append(
                    f"    Branch {n.branch.value}: total={n.total}/100  "
                    f"weakness='{n.key_weakness[:80]}'"
                )
    return "\n".join(traces) if traces else "  All categories resolved via direct scoring (ALIGNED)."


class SynthesisAgent:
    """Produces the final Investment Horizon Report from the aggregated package."""

    name = "SynthesisAgent"

    def __init__(self):
        llm = LLM(model=f"ollama/{LLM_MODEL}", temperature=0.2)
        self._agent = Agent(
            role="Technology Investment Report Writer",
            goal=(
                "Write a crisp, evidence-grounded Technology Investment Horizon Report "
                "from the provided analysis package. No tool calls. No fabrication."
            ),
            backstory=(
                "You are a senior investment strategist producing reports for engineering "
                "leaders at large technology companies. You distil complex multi-source "
                "analyses into actionable, well-structured investment guidance."
            ),
            llm=llm,
            verbose=False,
            allow_delegation=False,
        )

    async def run(self, pkg: SynthesisPackage) -> tuple[str, bool, list[str]]:
        """Generate the report and apply all output-time guardrails.

        Returns:
            (final_report_text, citation_audit_passed, unsupported_claims)
        """
        from guardrails.output_audit import run_output_guardrails

        prompt = _SYNTHESIS_PROMPT.format(
            cycle_date=pkg.cycle_date,
            category_summary=_format_category_summary(pkg),
            emerging_section=_format_emerging(pkg),
            delta_section=_format_delta_section(pkg),
            reasoning_traces=_format_reasoning_traces(pkg),
        )
        task = Task(
            description=prompt,
            agent=self._agent,
            expected_output="Complete Markdown Technology Investment Horizon Report.",
        )
        crew = Crew(agents=[self._agent], tasks=[task], verbose=False)
        draft = str(await crew.kickoff_async())

        # Apply: confidence-stance consistency, citation audit, scope disclaimer
        final_text, audit_passed, unsupported = run_output_guardrails(draft, pkg)

        self._write_report(pkg, final_text)
        return final_text, audit_passed, unsupported

    def _write_report(self, pkg: SynthesisPackage, text: str) -> Path:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = REPORTS_DIR / f"horizon_report_{ts}.md"
        header = (
            f"# Technology Investment Horizon Report\n"
            f"*Run ID: {pkg.run_id} | Cycle: {pkg.cycle_date}*\n\n---\n\n"
        )
        path.write_text(header + text)
        print(f"[SynthesisAgent] Report written → {path}")
        return path


if __name__ == "__main__":
    print("=== synthesis_agent local test ===")
    from schemas.messages import CategoryResult, AlignmentClassification
    pkg = SynthesisPackage(
        run_id="test-001",
        cycle_date="2026-06-15",
        category_results=[
            CategoryResult(
                category_id="ai_ml_infrastructure",
                category_label="AI / ML Infrastructure",
                classification=AlignmentClassification.ALIGNED,
                stance=InvestmentStance.BUY,
                final_score=78.0,
            ),
            CategoryResult(
                category_id="cloud_edge",
                category_label="Cloud & Edge Computing",
                classification=AlignmentClassification.CONTRADICTORY,
                stance=InvestmentStance.HOLD,
                final_score=54.0,
                winning_branch=None,
                reasoning_trace="Capital-Led branch won (score=71); Risk-Adjusted pruned (score=38).",
            ),
        ],
        emerging_candidates=["Neuromorphic Computing", "Quantum Networking"],
        emerging_rationale="Both show early GitHub adoption momentum without SEC capex confirmation yet.",
        portfolio_deltas={"ai_ml_infrastructure": "HOLD→BUY", "cloud_edge": "BUY→HOLD"},
    )
    agent = SynthesisAgent()
    report = asyncio.run(agent.run(pkg))
    print(f"\nReport preview:\n{report[:400]}…")

"""Orchestrator Agent.

Coordination only — no domain-level reasoning. Responsibilities:
  1. Load the category registry and signal weights.
  2. For each category: dispatch TaskAssignments to DataCollectorAgent and
     RetrievalAgent in parallel (asyncio.gather).
  3. Validate counter_evidence before Critic evaluation (hard gate).
  4. Route to ALIGNED (deterministic weighted scoring) or CONTRADICTORY (ToT).
  5. For CONTRADICTORY: run the full ToT beam search loop.
  6. After all categories: run the emerging-candidate ToT pass.
  7. Assemble the SynthesisPackage and dispatch to SynthesisAgent.
  8. Persist the run to long-term memory.

Per claud.md spec:
  - Must NOT perform domain reasoning (no LLM calls in this agent).
  - Owns all routing, state management, and persistence.
  - ALIGNED path: deterministic weighted average only.
  - ToT path: Generator → validate counter_evidence → Critic (per node) → prune → repeat.
"""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Optional

from signals.category_registry import load_registry
from schemas.messages import (
    TaskAssignment,
    SubScoreReport,
    RetrievalReport,
    AlignmentRequest,
    AlignmentVerdict,
    AlignmentClassification,
    GenerationRequest,
    ThoughtNodeSchema,
    ThoughtNodesReport,
    EvaluationRequest,
    NodeScoreReport,
    ScoreReport,
    SynthesisPackage,
    CategoryResult,
    InvestmentStance,
    BranchType,
)

# ToT beam search constants (per claud.md spec)
BEAM_WIDTH = 2
FLOOR_SCORE = 40          # hard floor — nodes below this are pruned
CLEAR_WINNER_THRESHOLD = 85
CLEAR_WINNER_MARGIN = 15
MAX_DEPTH = 2             # root (0) + one child refinement pass (1)

# Signal weights for ALIGNED deterministic scoring
DEFAULT_SIGNAL_WEIGHTS = {
    "sec_edgar":     0.30,
    "github":        0.20,
    "news":          0.20,
    "google_trends": 0.15,
    "fred_macro":    0.15,
}

TRIVIAL_COUNTER_EVIDENCE_MARKERS = [
    "", "n/a", "none", "no counter", "no counter evidence",
    "unable to parse", "[parse failure",
]


# ── Helper: counter_evidence gate ─────────────────────────────────────────────

def _is_trivial_counter_evidence(text: str) -> bool:
    """Return True if counter_evidence is empty or vague (fails the gate)."""
    clean = text.strip().lower()
    if len(clean) < 20:
        return True
    return any(clean.startswith(m) for m in TRIVIAL_COUNTER_EVIDENCE_MARKERS)


# ── Helper: deterministic weighted scoring for ALIGNED ────────────────────────

def _weighted_score(sub_scores: dict[str, float], weights: dict[str, float]) -> float:
    total_weight = sum(weights.get(k, 0) for k in sub_scores)
    if total_weight == 0:
        return 50.0
    return round(
        sum(sub_scores[k] * weights.get(k, 0) for k in sub_scores) / total_weight,
        1,
    )


def _score_to_stance(score: float) -> InvestmentStance:
    if score >= 65:
        return InvestmentStance.BUY
    if score >= 40:
        return InvestmentStance.HOLD
    return InvestmentStance.REDUCE


# ── Helper: ToT beam-search pruning ──────────────────────────────────────────

def _prune_beam(
    scores: list[NodeScoreReport],
) -> tuple[list[NodeScoreReport], list[NodeScoreReport], bool]:
    """Apply beam pruning rules.

    Returns:
        (survivors, pruned, branch_d_promoted)
    """
    # Hard floor
    above_floor = [s for s in scores if s.total >= FLOOR_SCORE]
    pruned = [s for s in scores if s.total < FLOOR_SCORE]

    # If nothing survives the floor, auto-promote Branch D
    if not above_floor:
        branch_d = next(
            (s for s in scores if s.branch == BranchType.EVIDENCE_INSUFFICIENT), None
        )
        if branch_d:
            return [branch_d], pruned, True
        return [], pruned, True

    # Keep top BEAM_WIDTH by score
    survivors = sorted(above_floor, key=lambda s: s.total, reverse=True)[:BEAM_WIDTH]
    pruned += [s for s in above_floor if s not in survivors]
    return survivors, pruned, False


def _check_termination(
    survivors: list[NodeScoreReport],
    depth: int,
) -> tuple[bool, Optional[str]]:
    """Check early-termination conditions.

    Returns (should_terminate, reason).
    """
    if not survivors:
        return True, "no_survivors"
    if len(survivors) == 1:
        return True, "single_survivor"
    if depth >= MAX_DEPTH:
        return True, "max_depth"
    top = survivors[0]
    if len(survivors) >= 2:
        margin = top.total - survivors[1].total
        if top.total >= CLEAR_WINNER_THRESHOLD and margin >= CLEAR_WINNER_MARGIN:
            return True, "clear_winner"
    return False, None


# ── Main Orchestrator ─────────────────────────────────────────────────────────

class OrchestratorAgent:
    """Coordinates all agents for a full investment signal analysis run."""

    def __init__(self):
        from agents.data_collector_agent import DataCollectorAgent
        from agents.retrieval_agent import RetrievalAgent
        from agents.signal_analyst_agent import SignalAnalystAgent
        from agents.thought_generator_agent import ThoughtGeneratorAgent
        from agents.critic_agent import CriticAgent
        from agents.synthesis_agent import SynthesisAgent

        self._data_collector = DataCollectorAgent()
        self._retrieval = RetrievalAgent()
        self._signal_analyst = SignalAnalystAgent()
        self._thought_generator = ThoughtGeneratorAgent()
        self._critic = CriticAgent()
        self._synthesis = SynthesisAgent()

        self._registry = load_registry()
        self._run_id = str(uuid.uuid4())[:8]
        self._cycle_date = datetime.now(timezone.utc).date().isoformat()
        self._sub_scores_cache: dict[str, SubScoreReport] = {}
        self._reeval_trigger_count = 0

    # ── Step 1: Collect signals for one category ──────────────────────────────

    async def _collect_category(
        self, category_id: str, category: dict
    ) -> tuple[SubScoreReport, RetrievalReport]:
        task = TaskAssignment(
            run_id=self._run_id,
            category_id=category_id,
            category_label=category["label"],
            tickers=category.get("tickers", []),
            query_terms=category.get("query_terms", []),
            cycle_date=self._cycle_date,
        )
        print(f"\n[Orchestrator] Collecting '{category['label']}' — running DataCollector + RetrievalAgent in parallel")
        sub_scores, retrieval = await asyncio.gather(
            self._data_collector.run(task),
            self._retrieval.run_async(task),
        )
        self._sub_scores_cache[category_id] = sub_scores
        return sub_scores, retrieval

    # ── Step 2: ALIGNED path (deterministic) ─────────────────────────────────

    def _score_aligned(
        self,
        verdict: AlignmentVerdict,
        sub_scores: SubScoreReport,
        retrieval: RetrievalReport,
        weights: dict[str, float],
    ) -> CategoryResult:
        scores = {
            "sec_edgar":     retrieval.sec_score,
            "github":        sub_scores.github_score,
            "news":          sub_scores.news_score,
            "google_trends": sub_scores.google_trends_score,
            "fred_macro":    sub_scores.fred_macro_score,
        }
        final_score = _weighted_score(scores, weights)
        stance = _score_to_stance(final_score)
        print(f"  [Orchestrator] ALIGNED → score={final_score:.1f}  stance={stance.value}")
        return CategoryResult(
            category_id=verdict.category_id,
            category_label=self._registry["categories"][verdict.category_id]["label"],
            classification=AlignmentClassification.ALIGNED,
            stance=stance,
            final_score=final_score,
        )

    # ── Step 3: CONTRADICTORY path (ToT beam search) ─────────────────────────

    def _validate_counter_evidence(self, nodes: list[ThoughtNodeSchema]) -> list[ThoughtNodeSchema]:
        """Gate: zero the counter_evidence field for trivial entries (Critic will see the flag)."""
        validated = []
        for node in nodes:
            if _is_trivial_counter_evidence(node.counter_evidence):
                print(f"  [Orchestrator] ⚠ Counter-evidence gate failed for Branch {node.branch.value} — Evidence Alignment will be zeroed")
                # Replace with a sentinel so Critic knows to zero Evidence Alignment
                node = ThoughtNodeSchema(
                    branch=node.branch, depth=node.depth,
                    content=node.content,
                    counter_evidence="__TRIVIAL__",
                    parent_branch=node.parent_branch,
                )
            validated.append(node)
        return validated

    async def _run_critic_pass(
        self,
        nodes: list[ThoughtNodeSchema],
        context: str,
        category_id: str,
        category_label: str,
    ) -> ScoreReport:
        eval_requests = [
            EvaluationRequest(
                run_id=self._run_id,
                category_id=category_id,
                node=node,
                context=context,
                re_evaluation=False,
            )
            for node in nodes
        ]
        # Score each node independently — one call per node
        score_report = await self._critic.score_all_nodes(eval_requests, category_label)
        if score_report.variance_triggered:
            self._reeval_trigger_count += 1

        # Apply counter_evidence zero to nodes that failed the gate
        zeroed_scores = []
        for score, node in zip(score_report.node_scores, nodes):
            if node.counter_evidence == "__TRIVIAL__":
                zeroed = NodeScoreReport(
                    branch=score.branch, depth=score.depth,
                    evidence_alignment=0,
                    internal_consistency=score.internal_consistency,
                    macro_compatibility=score.macro_compatibility,
                    actionability=score.actionability,
                    confidence_calibration=score.confidence_calibration,
                    total=score.internal_consistency + score.macro_compatibility +
                          score.actionability + score.confidence_calibration,
                    key_weakness=score.key_weakness,
                    counter_evidence_zeroed=True,
                )
                zeroed_scores.append(zeroed)
            else:
                zeroed_scores.append(score)

        score_report.node_scores = zeroed_scores
        return score_report

    async def _run_tot(
        self,
        category_id: str,
        category_label: str,
        context: str,
        weights: dict[str, float],
    ) -> CategoryResult:
        print(f"  [Orchestrator] CONTRADICTORY → entering ToT beam search")
        all_pruned: list[NodeScoreReport] = []

        # --- Depth 0: Generate 4 root nodes ---
        gen_req = GenerationRequest(
            run_id=self._run_id, category_id=category_id,
            category_label=category_label, context=context, depth=0,
        )
        nodes_report: ThoughtNodesReport = await self._thought_generator.run(gen_req)
        validated_nodes = self._validate_counter_evidence(nodes_report.nodes)

        score_report = await self._run_critic_pass(validated_nodes, context, category_id, category_label)
        survivors, pruned, branch_d_auto = _prune_beam(score_report.node_scores)
        all_pruned.extend(pruned)

        terminate, reason = _check_termination(survivors, depth=0)
        if branch_d_auto:
            print(f"  [Orchestrator] Branch D auto-promoted (no nodes passed floor={FLOOR_SCORE})")
            return self._build_result(
                category_id, category_label, InvestmentStance.MANUAL_REVIEW,
                50.0, BranchType.EVIDENCE_INSUFFICIENT, "Branch D auto-promoted",
                all_pruned, manual_review=True,
            )

        if terminate:
            print(f"  [Orchestrator] ToT terminated at depth=0 ({reason})")
            winner = survivors[0]
            return self._build_result_from_winner(
                category_id, category_label, winner, validated_nodes, all_pruned, weights,
            )

        # --- Depth 1: Refine survivors ---
        survivor_branches = [s.branch.value for s in survivors]
        child_gen_req = GenerationRequest(
            run_id=self._run_id, category_id=category_id,
            category_label=category_label, context=context,
            depth=1, survivor_branches=survivor_branches,
        )
        child_report = await self._thought_generator.run(child_gen_req)
        child_nodes = self._validate_counter_evidence(child_report.nodes)

        child_scores = await self._run_critic_pass(child_nodes, context, category_id, category_label)
        child_survivors, child_pruned, child_d_auto = _prune_beam(child_scores.node_scores)
        all_pruned.extend(child_pruned)

        if child_d_auto or not child_survivors:
            return self._build_result(
                category_id, category_label, InvestmentStance.MANUAL_REVIEW,
                50.0, BranchType.EVIDENCE_INSUFFICIENT,
                "No survivors after depth-1 pruning", all_pruned, manual_review=True,
            )

        winner = child_survivors[0]
        all_pruned.extend(child_survivors[1:])
        return self._build_result_from_winner(
            category_id, category_label, winner, child_nodes, all_pruned, weights,
        )

    def _build_result_from_winner(
        self,
        category_id: str,
        category_label: str,
        winner: NodeScoreReport,
        nodes: list[ThoughtNodeSchema],
        all_pruned: list[NodeScoreReport],
        weights: dict[str, float],
    ) -> CategoryResult:
        # Branch D winning → manual review flag
        if winner.branch == BranchType.EVIDENCE_INSUFFICIENT:
            print(f"  [Orchestrator] ⚠ Branch D won — flagging for manual review")
            return self._build_result(
                category_id, category_label, InvestmentStance.MANUAL_REVIEW,
                winner.total, winner.branch,
                f"Branch D selected (score={winner.total})", all_pruned, manual_review=True,
            )

        # Map branch to stance
        stance_map = {
            BranchType.CAPITAL_LED:   InvestmentStance.BUY,
            BranchType.ADOPTION_LED:  InvestmentStance.BUY,
            BranchType.RISK_ADJUSTED: InvestmentStance.REDUCE,
        }
        stance = stance_map.get(winner.branch, InvestmentStance.HOLD)
        # Moderate score: winner.total is 0-100 critic score, not the signal score
        final_score = winner.total
        winning_node = next((n for n in nodes if n.branch == winner.branch), None)
        trace = winning_node.content[:300] if winning_node else ""

        print(f"  [Orchestrator] ToT winner: Branch {winner.branch.value}  score={winner.total}  stance={stance.value}")
        return self._build_result(
            category_id, category_label, stance, final_score,
            winner.branch, trace, all_pruned,
        )

    def _build_result(
        self,
        category_id: str,
        category_label: str,
        stance: InvestmentStance,
        final_score: float,
        winning_branch: Optional[BranchType],
        reasoning_trace: str,
        pruned_nodes: list[NodeScoreReport],
        manual_review: bool = False,
    ) -> CategoryResult:
        return CategoryResult(
            category_id=category_id,
            category_label=category_label,
            classification=AlignmentClassification.CONTRADICTORY,
            stance=stance,
            final_score=final_score,
            winning_branch=winning_branch,
            reasoning_trace=reasoning_trace,
            manual_review_flag=manual_review,
            pruned_nodes=pruned_nodes,
        )

    # ── Step 4: Emerging-candidate ToT pass ───────────────────────────────────

    async def _run_emerging_pass(
        self, all_results: list[CategoryResult], context_by_category: dict[str, str]
    ) -> tuple[list[str], str]:
        """Single-level ToT pass over anomalous positive-signal candidates."""
        candidates = [
            r.category_id for r in all_results
            if r.stance == InvestmentStance.HOLD and r.final_score >= 55
        ]
        if not candidates:
            return [], ""

        combined_context = "\n\n".join(
            f"=== {cid} ===\n{context_by_category.get(cid, '')}"
            for cid in candidates[:3]
        )
        gen_req = GenerationRequest(
            run_id=self._run_id,
            category_id="emerging",
            category_label="Emerging Candidates",
            context=combined_context,
            depth=0,
            survivor_branches=[],
        )
        nodes_report = await self._thought_generator.run(gen_req)
        if not nodes_report.nodes:
            return candidates[:2], "Insufficient signal to refine."

        validated = self._validate_counter_evidence(nodes_report.nodes[:3])
        score_report = await self._run_critic_pass(
            validated, combined_context, "emerging", "Emerging Candidates"
        )
        if not score_report.node_scores:
            return candidates[:2], ""

        best = max(score_report.node_scores, key=lambda s: s.total)
        best_node = next((n for n in validated if n.branch == best.branch), None)
        rationale = best_node.content[:400] if best_node else ""
        return candidates[:3], rationale

    # ── Main run ──────────────────────────────────────────────────────────────

    async def run(self, category_ids: Optional[list[str]] = None) -> str:
        tech_categories = self._registry.get("categories", {})
        if category_ids:
            tech_categories = {k: v for k, v in tech_categories.items() if k in category_ids}

        import time
        self._run_start = time.monotonic()

        all_results: list[CategoryResult] = []
        context_by_category: dict[str, str] = {}

        for cat_id, category in tech_categories.items():
            weights = category.get("signal_weights", DEFAULT_SIGNAL_WEIGHTS)

            # Parallel data + retrieval collection
            sub_scores, retrieval = await self._collect_category(cat_id, category)

            # Build signal context — use LLM-generated rationales from DataCollectorAgent,
            # then append authoritative SEC filing passages from RetrievalAgent.
            context = sub_scores.signal_context or (
                f"=== {category['label']} — Signal Scores ===\n"
                f"GitHub={sub_scores.github_score:.0f}  "
                f"News={sub_scores.news_score:.0f}  "
                f"Trends={sub_scores.google_trends_score:.0f}  "
                f"Macro={sub_scores.fred_macro_score:.0f}  "
                f"SEC_EDGAR={retrieval.sec_score:.0f}"
            )
            if retrieval.passages:
                passage_lines = "\n".join(
                    f"  [{p.ticker} | {p.date} | {p.query_type}] {p.content[:200]}"
                    for p in retrieval.passages[:5]
                )
                context += f"\n\n=== SEC EDGAR Filing Passages (RetrievalAgent) ===\n{passage_lines}"
            context_by_category[cat_id] = context

            # Signal analyst classification
            alignment_req = self._signal_analyst.build_request(sub_scores, retrieval)
            verdict = self._signal_analyst.run(alignment_req)

            if verdict.classification == AlignmentClassification.ALIGNED:
                result = self._score_aligned(verdict, sub_scores, retrieval, weights)
            else:
                result = await self._run_tot(
                    cat_id, category["label"], context, weights
                )

            all_results.append(result)

        # Emerging-candidate pass
        emerging, emerging_rationale = await self._run_emerging_pass(all_results, context_by_category)

        # Assemble synthesis package
        pkg = SynthesisPackage(
            run_id=self._run_id,
            cycle_date=self._cycle_date,
            category_results=all_results,
            emerging_candidates=emerging,
            emerging_rationale=emerging_rationale,
            reeval_trigger_count=self._reeval_trigger_count,
        )

        # ── Output-time guardrails ────────────────────────────────────────────
        from guardrails.escalation import EscalationEvaluator, format_escalation_notice
        from guardrails.data_integrity import check_source_availability
        from evaluation.metrics import MetricsStore, compute_run_metrics
        import time

        run_end = time.monotonic()
        latency_s = run_end - getattr(self, "_run_start", run_end)

        all_attributions = [
            attr
            for cat_id in context_by_category
            for sub in [getattr(self, "_sub_scores_cache", {}).get(cat_id)]
            if sub is not None
            for attr in sub.source_attributions
        ]
        source_availability, _ = check_source_availability(all_attributions) if all_attributions else (1.0, [])
        staleness_count = sum(
            1
            for sub in getattr(self, "_sub_scores_cache", {}).values()
            for flag in sub.staleness_flags
            if flag.is_stale
        )

        trailing_accuracy = MetricsStore().get_trailing_forecast_accuracy()
        escalation = EscalationEvaluator().evaluate(
            pkg, all_attributions, source_availability, trailing_accuracy
        )
        pkg.escalation = escalation

        # Level 3 halt — do not call SynthesisAgent
        if escalation.halt_pipeline:
            halt_notice = format_escalation_notice(escalation)
            print(f"[Orchestrator] LEVEL 3 HALT — pipeline stopped.{halt_notice}")
            self._persist(all_results)
            return halt_notice

        # Synthesis agent produces the final report (includes output audit)
        report, audit_passed, unsupported = await self._synthesis.run(pkg)
        pkg.citation_audit_passed = audit_passed
        pkg.unsupported_claims = unsupported

        # Re-evaluate escalation now that citation audit result is known
        if not audit_passed:
            escalation = EscalationEvaluator().evaluate(
                pkg, all_attributions, source_availability, trailing_accuracy
            )
            pkg.escalation = escalation
            if escalation.halt_pipeline:
                report += format_escalation_notice(escalation)

        # ── Evaluation metrics ────────────────────────────────────────────────
        metrics = compute_run_metrics(
            pkg, source_availability, staleness_count, latency_s,
            self._run_id, self._cycle_date,
        )
        MetricsStore().record(metrics)
        print(
            f"[Orchestrator] Run complete — escalation={escalation.level.value}  "
            f"latency={latency_s:.1f}s  availability={source_availability:.0%}"
        )

        # Persist to long-term memory
        self._persist(all_results)
        return report

    def _persist(self, results: list[CategoryResult]) -> None:
        from memory.long_term import LongTermMemory
        mem = LongTermMemory()
        for r in results:
            mem.store_run(
                ticker=r.category_id,
                signal_score=r.final_score,
                stance=r.stance.value,
                recommendation=r.stance.value,
                summary=r.reasoning_trace or r.stance.value,
            )


def run_orchestrator(category_ids: Optional[list[str]] = None) -> str:
    return asyncio.run(OrchestratorAgent().run(category_ids))


if __name__ == "__main__":
    print("=== orchestrator_agent local test (single category, requires Ollama + FAISS) ===")
    report = run_orchestrator(category_ids=["ai_ml_infrastructure"])
    print(f"\nReport preview:\n{report[:600]}…")

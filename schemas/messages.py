"""Structured JSON schemas for all inter-agent messages.

All agent-to-Orchestrator and Orchestrator-to-agent messages use these
Pydantic models to enforce type contracts instead of free-text summaries.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class AlignmentClassification(str, Enum):
    ALIGNED = "ALIGNED"
    CONTRADICTORY = "CONTRADICTORY"


class InvestmentStance(str, Enum):
    BUY = "BUY"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    MANUAL_REVIEW = "MANUAL_REVIEW"


class BranchType(str, Enum):
    CAPITAL_LED = "A"           # Large capital commitments from leaders
    ADOPTION_LED = "B"          # Developer/enterprise adoption momentum
    RISK_ADJUSTED = "C"         # Macro/regulatory/competitive headwinds dominate
    EVIDENCE_INSUFFICIENT = "D" # Contradictory — requires manual review


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class EscalationLevel(str, Enum):
    NONE = "NONE"
    LEVEL_1_FLAG = "LEVEL_1_FLAG"   # in-brief flag, no delivery delay
    LEVEL_2_HOLD = "LEVEL_2_HOLD"   # hold pending reviewer approval
    LEVEL_3_HALT = "LEVEL_3_HALT"   # full pipeline halt


# ── Safety / Reliability models ───────────────────────────────────────────────

class SourceAttribution(BaseModel):
    """Mandatory source attribution for every sub-score (data integrity layer)."""
    source_key: str = Field(..., description="e.g. 'github', 'news', 'sec_edgar'")
    tool_name: str = Field(..., description="Python tool/agent that produced this data")
    query: str = Field(..., description="Actual query or search string used")
    retrieved_at: str = Field(..., description="ISO-8601 UTC timestamp of fetch")
    record_count: int = Field(..., description="Number of records returned (0 = failed)")
    status: str = Field(..., description="'ok' | 'empty' | 'failed'")


class StalenessFlag(BaseModel):
    """Staleness flag raised when underlying data is older than the threshold."""
    source_key: str
    is_stale: bool
    data_age_hours: Optional[float] = None
    threshold_hours: float
    reason: str


class RunEscalation(BaseModel):
    """Three-tier human escalation result for a pipeline run."""
    level: EscalationLevel = EscalationLevel.NONE
    triggers: list[str] = Field(default_factory=list,
                                description="Human-readable list of conditions that fired")
    requires_reviewer: bool = False
    halt_pipeline: bool = False
    resolved_at: Optional[str] = None


# ── 1. Task Assignment: Orchestrator → Data Collector / Retrieval Agent ────────

class TaskAssignment(BaseModel):
    run_id: str = Field(..., description="Unique identifier for this analysis run")
    category_id: str = Field(..., description="Technology category key from registry")
    category_label: str = Field(..., description="Human-readable category name")
    tickers: list[str] = Field(..., description="Representative tickers for SEC retrieval")
    query_terms: list[str] = Field(..., description="Search terms for data collection")
    cycle_date: str = Field(..., description="ISO date of the analysis cycle")


# ── 2. Sub-Score Report: Data Collector → Orchestrator ────────────────────────

class SubScoreReport(BaseModel):
    run_id: str
    category_id: str
    github_score: float = Field(..., ge=0, le=100, description="GitHub trends sub-score (0-100)")
    news_score: float = Field(..., ge=0, le=100, description="News sentiment sub-score (0-100)")
    google_trends_score: float = Field(..., ge=0, le=100, description="Google Trends sub-score (0-100)")
    fred_macro_score: float = Field(..., ge=0, le=100, description="FRED macro sub-score (0-100)")
    sec_edgar_score: float = Field(
        0.0, ge=0, le=100,
        description=(
            "SEC EDGAR RAG sub-score (0-100) from edgar_retriever coverage × quality. "
            "Provided by DataCollectorAgent; Orchestrator prefers RetrievalAgent's "
            "sec_score when available (higher fidelity)."
        )
    )
    signal_context: str = Field(
        "",
        description=(
            "LLM-generated per-source narrative rationales concatenated into a single "
            "context string. Used as the primary context input for the Thought Generator "
            "and Critic agents. Empty if LLM scoring was skipped."
        )
    )
    # ── Data integrity fields (Step 5 guardrails) ─────────────────────────────
    source_attributions: list[SourceAttribution] = Field(
        default_factory=list,
        description="Mandatory attribution for every sub-score produced"
    )
    staleness_flags: list[StalenessFlag] = Field(
        default_factory=list,
        description="Raised when source data is older than its freshness threshold"
    )
    failed_sources: list[str] = Field(
        default_factory=list,
        description="Sources that returned no data — listed explicitly, not silently omitted"
    )
    raw_evidence: dict = Field(default_factory=dict, description="Raw data snapshots for audit")


# ── 3. Retrieval Report: Retrieval Agent → Orchestrator ───────────────────────

class RetrievalPassage(BaseModel):
    ticker: str
    date: str
    accession: str
    query_type: str = Field(..., description="capital_commitment | platform_prioritization | forward_guidance")
    content: str
    similarity_score: float = Field(..., ge=0, le=1)
    keyword_confirmed: bool


class RetrievalReport(BaseModel):
    run_id: str
    category_id: str
    sec_score: float = Field(..., ge=0, le=100, description="SEC filing sub-score (0-100)")
    passages: list[RetrievalPassage] = Field(default_factory=list)
    passages_below_threshold: int = Field(0, description="Count of passages filtered by similarity < 0.72")


# ── 4. Alignment Request: Orchestrator → Signal Analyst ───────────────────────

class AlignmentRequest(BaseModel):
    run_id: str
    category_id: str
    sub_scores: dict[str, float] = Field(
        ...,
        description="Keyed by source: sec_edgar, github, news, google_trends, fred_macro"
    )


# ── 5. Alignment Verdict: Signal Analyst → Orchestrator ───────────────────────

class AlignmentVerdict(BaseModel):
    run_id: str
    category_id: str
    classification: AlignmentClassification
    alignment_spread: float = Field(..., description="max(sub_scores) - min(sub_scores)")
    hard_contradiction: bool = Field(False, description="True if strong capital + major deprecation detected")
    rationale: str = Field(..., description="One sentence explaining the classification")


# ── 6. Generation Request: Orchestrator → Thought Generator ───────────────────

class GenerationRequest(BaseModel):
    run_id: str
    category_id: str
    category_label: str
    context: str = Field(..., description="Aggregated signal context — no Critic scores included")
    depth: int = Field(0, description="0 = root nodes, 1 = child refinements")
    survivor_branches: list[str] = Field(
        default_factory=list,
        description="Branch types to refine at depth > 0 (e.g. ['A', 'B'])"
    )


# ── 7. Thought Nodes: Thought Generator → Orchestrator ────────────────────────

class ThoughtNodeSchema(BaseModel):
    branch: BranchType
    depth: int
    content: str = Field(..., description="Investment thesis text for this branch")
    counter_evidence: str = Field(
        ...,
        description="Non-trivial counter-argument that could invalidate this stance"
    )
    parent_branch: Optional[BranchType] = None


class ThoughtNodesReport(BaseModel):
    run_id: str
    category_id: str
    depth: int
    nodes: list[ThoughtNodeSchema]


# ── 8. Evaluation Request: Orchestrator → Critic ──────────────────────────────

class EvaluationRequest(BaseModel):
    run_id: str
    category_id: str
    node: ThoughtNodeSchema = Field(..., description="Single node to evaluate — never batched")
    context: str = Field(..., description="Signal context — no sibling scores included")
    re_evaluation: bool = Field(False, description="True if variance triggered a re-evaluation pass")


# ── 9. Score Report: Critic → Orchestrator ────────────────────────────────────

class NodeScoreReport(BaseModel):
    branch: BranchType
    depth: int
    evidence_alignment: int = Field(..., ge=0, le=20)
    internal_consistency: int = Field(..., ge=0, le=20)
    macro_compatibility: int = Field(..., ge=0, le=20)
    actionability: int = Field(..., ge=0, le=20)
    confidence_calibration: int = Field(..., ge=0, le=20)
    total: int = Field(..., ge=0, le=100)
    key_weakness: str = Field(..., description="Single most significant weakness identified")
    counter_evidence_zeroed: bool = Field(
        False,
        description="True if Evidence Alignment was zeroed due to trivial counter_evidence"
    )


class ScoreReport(BaseModel):
    run_id: str
    category_id: str
    node_scores: list[NodeScoreReport]
    variance_triggered: bool = Field(
        False,
        description="True if all scores fell within a 10-point band on initial pass"
    )


# ── 10. Synthesis Package: Orchestrator → Synthesis Agent ─────────────────────

class CategoryResult(BaseModel):
    category_id: str
    category_label: str
    classification: AlignmentClassification
    stance: InvestmentStance
    final_score: float = Field(..., ge=0, le=100)
    winning_branch: Optional[BranchType] = None
    reasoning_trace: Optional[str] = None
    manual_review_flag: bool = False
    pruned_nodes: list[NodeScoreReport] = Field(default_factory=list)
    # ── Output-time guardrails (Step 5) ───────────────────────────────────────
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM
    in_brief_flags: list[str] = Field(
        default_factory=list,
        description="Level 1 human flags included in the report without delivery delay"
    )


class SynthesisPackage(BaseModel):
    run_id: str
    cycle_date: str
    category_results: list[CategoryResult]
    emerging_candidates: list[str] = Field(default_factory=list, description="2-3 surfaced emerging tech categories")
    emerging_rationale: Optional[str] = None
    portfolio_deltas: dict[str, str] = Field(
        default_factory=dict,
        description="category_id → change vs prior cycle (e.g. 'BUY→HOLD')"
    )
    prior_cycle_date: Optional[str] = None
    # ── Pipeline-level guardrails (Step 5) ────────────────────────────────────
    escalation: RunEscalation = Field(
        default_factory=RunEscalation,
        description="Three-tier escalation verdict for the full run"
    )
    citation_audit_passed: bool = True
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Claims identified by citation audit without verifiable source backing"
    )
    reeval_trigger_count: int = Field(
        default=0, description="Number of Critic variance re-evaluations triggered this run"
    )

"""Three-Tier Escalation Model — Step 5 human-in-the-loop design.

Level 1 — IN-BRIEF FLAG (no delivery delay):
  · Any category resolved via Branch D (MANUAL_REVIEW stance)
  · Any sub-score with a staleness flag
  · Any stance with LOW confidence

Level 2 — MANDATORY HOLD (pending reviewer approval):
  · Any source dimension availability < 70% across the run
  · Two or more Critic variance re-evaluation triggers in one run
  · Trailing Forecast Accuracy below threshold (requires metrics history)

Level 3 — FULL PIPELINE HALT:
  · Citation groundedness audit failure (unsupported claims detected)
  · Two or more data sources fully unavailable (status='failed')
  · Unrecoverable schema validation failure

At MVP scale the reviewer is the engineering leader themselves via explicit
feedback. A dedicated technical owner role is identified as the production
target for Level 2 and Level 3 escalations.
"""

from __future__ import annotations

from schemas.messages import (
    SynthesisPackage,
    EscalationLevel,
    RunEscalation,
    InvestmentStance,
    ConfidenceLevel,
    SourceAttribution,
)

# Thresholds
SOURCE_AVAILABILITY_THRESHOLD = 0.70   # below this → Level 2
MAX_REEVAL_TRIGGERS_PER_RUN = 2        # at or above this → Level 2
FAILED_SOURCES_HALT_THRESHOLD = 2      # at or above this → Level 3
FORECAST_ACCURACY_THRESHOLD = 0.55     # trailing accuracy below this → Level 2


class EscalationEvaluator:
    """Evaluates a completed run against all three escalation tiers.

    Call after synthesis but before delivering the report.
    """

    def evaluate(
        self,
        pkg: SynthesisPackage,
        source_attributions: list[SourceAttribution],
        source_availability: float,
        trailing_forecast_accuracy: float | None = None,
    ) -> RunEscalation:
        """Return the highest applicable escalation level for the run.

        Args:
            pkg:                      Completed SynthesisPackage.
            source_attributions:      All attributions collected across all categories.
            source_availability:      Overall source availability rate (0.0–1.0).
            trailing_forecast_accuracy: Optional historical accuracy (None → skip check).

        Returns:
            RunEscalation with level, triggers, requires_reviewer, halt_pipeline.
        """
        l3 = self._check_level_3(pkg, source_attributions)
        if l3:
            return RunEscalation(
                level=EscalationLevel.LEVEL_3_HALT,
                triggers=l3,
                requires_reviewer=True,
                halt_pipeline=True,
            )

        l2 = self._check_level_2(pkg, source_availability, trailing_forecast_accuracy)
        if l2:
            return RunEscalation(
                level=EscalationLevel.LEVEL_2_HOLD,
                triggers=l2,
                requires_reviewer=True,
                halt_pipeline=False,
            )

        l1 = self._check_level_1(pkg)
        if l1:
            return RunEscalation(
                level=EscalationLevel.LEVEL_1_FLAG,
                triggers=l1,
                requires_reviewer=False,
                halt_pipeline=False,
            )

        return RunEscalation(
            level=EscalationLevel.NONE,
            triggers=[],
            requires_reviewer=False,
            halt_pipeline=False,
        )

    # ── Level 3 — Pipeline Halt ───────────────────────────────────────────────

    def _check_level_3(
        self,
        pkg: SynthesisPackage,
        attributions: list[SourceAttribution],
    ) -> list[str]:
        triggers = []

        if not pkg.citation_audit_passed:
            claims = "; ".join(pkg.unsupported_claims[:3])
            triggers.append(
                f"Citation groundedness audit failed — unsupported claim(s) detected: {claims}"
            )

        failed_sources = [a.source_key for a in attributions if a.status == "failed"]
        unique_failed = list(set(failed_sources))
        if len(unique_failed) >= FAILED_SOURCES_HALT_THRESHOLD:
            triggers.append(
                f"Two or more data sources fully unavailable: {unique_failed}"
            )

        return triggers

    # ── Level 2 — Mandatory Hold ──────────────────────────────────────────────

    def _check_level_2(
        self,
        pkg: SynthesisPackage,
        source_availability: float,
        trailing_accuracy: float | None,
    ) -> list[str]:
        triggers = []

        if source_availability < SOURCE_AVAILABILITY_THRESHOLD:
            triggers.append(
                f"Source availability {source_availability:.0%} is below the "
                f"{SOURCE_AVAILABILITY_THRESHOLD:.0%} threshold — data coverage insufficient"
            )

        if pkg.reeval_trigger_count >= MAX_REEVAL_TRIGGERS_PER_RUN:
            triggers.append(
                f"Critic score-variance re-evaluation triggered {pkg.reeval_trigger_count}× "
                f"(≥ {MAX_REEVAL_TRIGGERS_PER_RUN}) — signal quality is low across multiple categories"
            )

        if (
            trailing_accuracy is not None
            and trailing_accuracy < FORECAST_ACCURACY_THRESHOLD
        ):
            triggers.append(
                f"Trailing Forecast Accuracy {trailing_accuracy:.0%} is below the "
                f"{FORECAST_ACCURACY_THRESHOLD:.0%} threshold — model confidence degraded"
            )

        return triggers

    # ── Level 1 — In-Brief Flags ──────────────────────────────────────────────

    def _check_level_1(self, pkg: SynthesisPackage) -> list[str]:
        flags = []

        manual_review_cats = [
            r.category_label for r in pkg.category_results
            if r.stance == InvestmentStance.MANUAL_REVIEW
        ]
        if manual_review_cats:
            flags.append(
                f"Branch D resolution required for: {', '.join(manual_review_cats)} — "
                "signals irreconcilable; human judgment needed"
            )

        stale_cats = [
            r.category_label for r in pkg.category_results
            if any(flag.strip() for flag in r.in_brief_flags if "stale" in flag.lower())
        ]
        if stale_cats:
            flags.append(
                f"Stale sub-score detected in: {', '.join(stale_cats)} — "
                "data may not reflect current conditions"
            )

        low_confidence = [
            f"{r.category_label} ({r.stance.value})"
            for r in pkg.category_results
            if r.confidence_level == ConfidenceLevel.LOW
        ]
        if low_confidence:
            flags.append(
                f"LOW confidence stance in: {', '.join(low_confidence)} — "
                "validate before acting"
            )

        return flags


def format_escalation_notice(escalation: RunEscalation) -> str:
    """Return a human-readable escalation notice for inclusion in the report."""
    if escalation.level == EscalationLevel.NONE:
        return ""
    level_labels = {
        EscalationLevel.LEVEL_1_FLAG: "LEVEL 1 — IN-BRIEF FLAGS",
        EscalationLevel.LEVEL_2_HOLD: "LEVEL 2 — MANDATORY HOLD (reviewer approval required)",
        EscalationLevel.LEVEL_3_HALT: "LEVEL 3 — PIPELINE HALTED (do not act on this report)",
    }
    label = level_labels.get(escalation.level, escalation.level.value)
    trigger_lines = "\n".join(f"  • {t}" for t in escalation.triggers)
    return (
        f"\n\n---\n\n> **⚠ ESCALATION NOTICE — {label}**\n>\n"
        + "\n".join(f"> {line}" for line in trigger_lines.splitlines())
        + "\n"
    )


if __name__ == "__main__":
    from schemas.messages import (
        SynthesisPackage, CategoryResult, AlignmentClassification,
        InvestmentStance, ConfidenceLevel, SourceAttribution,
    )
    print("=== escalation local test ===")
    pkg = SynthesisPackage(
        run_id="test-001", cycle_date="2026-06-16",
        category_results=[
            CategoryResult(
                category_id="ai_ml", category_label="AI / ML Infrastructure",
                classification=AlignmentClassification.CONTRADICTORY,
                stance=InvestmentStance.MANUAL_REVIEW,
                final_score=50.0, manual_review_flag=True,
                confidence_level=ConfidenceLevel.LOW,
                in_brief_flags=["Branch D promoted", "stale news sub-score"],
            ),
        ],
        citation_audit_passed=True,
        reeval_trigger_count=0,
    )
    attrs = [
        SourceAttribution(source_key="github", tool_name="github_tool", query="AI",
                          retrieved_at="2026-06-16T10:00:00+00:00", record_count=8, status="ok"),
        SourceAttribution(source_key="news", tool_name="newsdata_tool", query="AI chips",
                          retrieved_at="2026-06-16T10:00:00+00:00", record_count=0, status="failed"),
    ]
    evaluator = EscalationEvaluator()
    result = evaluator.evaluate(pkg, attrs, source_availability=0.75)
    print(f"Level: {result.level.value}")
    for t in result.triggers:
        print(f"  · {t}")
    print(format_escalation_notice(result))

"""Signal Analyst Agent.

Computes the alignment spread across all five normalized sub-scores for a
technology category and classifies it as ALIGNED or CONTRADICTORY.

Classification rules (per claud.md spec):
  ALIGNED      — spread ≤ 20 points (all sources broadly agree)
  CONTRADICTORY — spread > 20 points OR a hard contradiction is detected

Hard contradiction patterns:
  - Strong capital commitment signal (SEC ≥ 70) + major deprecation/risk news (News ≤ 30)
  - Strong adoption signal (GitHub ≥ 70, Trends ≥ 70) + negative macro (FRED ≤ 30)

ALIGNED categories are scored deterministically by the Orchestrator.
CONTRADICTORY categories enter the ToT beam search pipeline.
"""

from schemas.messages import (
    AlignmentRequest,
    AlignmentVerdict,
    AlignmentClassification,
    SubScoreReport,
    RetrievalReport,
)

SPREAD_THRESHOLD = 20.0


def _detect_hard_contradiction(scores: dict[str, float]) -> bool:
    """Check for patterns where sources contradict each other directionally."""
    sec = scores.get("sec_edgar", 50.0)
    news = scores.get("news", 50.0)
    github = scores.get("github", 50.0)
    trends = scores.get("google_trends", 50.0)
    macro = scores.get("fred_macro", 50.0)

    # Pattern 1: Strong capital commitment but major deprecation/risk news
    if sec >= 70 and news <= 30:
        return True

    # Pattern 2: Strong adoption momentum but severely negative macro
    if github >= 70 and trends >= 70 and macro <= 30:
        return True

    return False


class SignalAnalystAgent:
    """Classifies a category as ALIGNED or CONTRADICTORY from its sub-scores."""

    name = "SignalAnalystAgent"

    def run(self, request: AlignmentRequest) -> AlignmentVerdict:
        scores = request.sub_scores
        values = list(scores.values())

        if not values:
            return AlignmentVerdict(
                run_id=request.run_id,
                category_id=request.category_id,
                classification=AlignmentClassification.CONTRADICTORY,
                alignment_spread=0.0,
                hard_contradiction=False,
                rationale="No sub-scores provided — defaulting to CONTRADICTORY.",
            )

        spread = max(values) - min(values)
        hard = _detect_hard_contradiction(scores)
        is_contradictory = spread > SPREAD_THRESHOLD or hard

        classification = (
            AlignmentClassification.CONTRADICTORY
            if is_contradictory
            else AlignmentClassification.ALIGNED
        )

        # Build rationale
        high_src = max(scores, key=scores.get)
        low_src = min(scores, key=scores.get)
        if hard:
            rationale = (
                f"Hard contradiction: {high_src}={scores[high_src]:.0f} conflicts "
                f"directionally with {low_src}={scores[low_src]:.0f}."
            )
        elif is_contradictory:
            rationale = (
                f"Spread={spread:.1f} pts exceeds threshold ({SPREAD_THRESHOLD}): "
                f"{high_src}={scores[high_src]:.0f} vs {low_src}={scores[low_src]:.0f}."
            )
        else:
            avg = sum(values) / len(values)
            rationale = (
                f"All five sources within {spread:.1f} pts of each other "
                f"(avg={avg:.0f}). Proceeding with deterministic scoring."
            )

        print(f"  [SignalAnalyst] {classification.value}  spread={spread:.1f}  "
              f"hard={hard}  — {rationale}")

        return AlignmentVerdict(
            run_id=request.run_id,
            category_id=request.category_id,
            classification=classification,
            alignment_spread=round(spread, 1),
            hard_contradiction=hard,
            rationale=rationale,
        )

    @staticmethod
    def build_request(
        sub_score_report: SubScoreReport,
        retrieval_report: RetrievalReport,
    ) -> AlignmentRequest:
        """Convenience method to build an AlignmentRequest from both agent reports."""
        return AlignmentRequest(
            run_id=sub_score_report.run_id,
            category_id=sub_score_report.category_id,
            sub_scores={
                "sec_edgar":     retrieval_report.sec_score,
                "github":        sub_score_report.github_score,
                "news":          sub_score_report.news_score,
                "google_trends": sub_score_report.google_trends_score,
                "fred_macro":    sub_score_report.fred_macro_score,
            },
        )


if __name__ == "__main__":
    print("=== signal_analyst_agent local test ===")
    agent = SignalAnalystAgent()

    # Case 1: ALIGNED
    aligned_req = AlignmentRequest(
        run_id="test", category_id="ai_ml",
        sub_scores={"sec_edgar": 75, "github": 80, "news": 72, "google_trends": 68, "fred_macro": 62},
    )
    v1 = agent.run(aligned_req)
    print(f"  Case 1: {v1.classification.value}  spread={v1.alignment_spread}")

    # Case 2: CONTRADICTORY (spread)
    contra_req = AlignmentRequest(
        run_id="test", category_id="cloud",
        sub_scores={"sec_edgar": 82, "github": 30, "news": 45, "google_trends": 35, "fred_macro": 28},
    )
    v2 = agent.run(contra_req)
    print(f"  Case 2: {v2.classification.value}  spread={v2.alignment_spread}  hard={v2.hard_contradiction}")

    # Case 3: Hard contradiction
    hard_req = AlignmentRequest(
        run_id="test", category_id="semis",
        sub_scores={"sec_edgar": 78, "github": 60, "news": 25, "google_trends": 55, "fred_macro": 50},
    )
    v3 = agent.run(hard_req)
    print(f"  Case 3: {v3.classification.value}  hard={v3.hard_contradiction}")

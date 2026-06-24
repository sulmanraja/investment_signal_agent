"""Evaluation Metrics — Step 5.

Three metric families tracked per run:

Output Quality (measured structurally per run; accuracy tracked over time):
  forecast_accuracy          — % stances later confirmed correct (requires feedback)
  citation_groundedness      — did the citation audit pass? (1.0 or 0.0)
  confidence_calibration     — % of HIGH-confidence stances confirmed correct (feedback)
  engineering_leader_agree   — % of stances rated 'agree' in feedback (feedback)

ToT Reasoning Process Quality:
  branch_d_promotion_rate    — % categories that ended as Branch D / MANUAL_REVIEW
  counter_evidence_complete  — % ToT nodes with non-trivial counter_evidence
  critic_score_variance      — mean std-dev across Critic scores in the run

Operational Reliability:
  source_availability        — % sources returning data across all categories
  staleness_flag_rate        — % sub-scores with at least one staleness flag
  end_to_end_latency_s       — total run time in seconds
  reeval_trigger_count       — number of variance re-evaluations fired

Metrics are persisted to memory/metrics_store.json and read back to support
the Level 2 trailing-accuracy escalation check.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

METRICS_PATH = Path(__file__).parent.parent / "memory" / "metrics_store.json"


@dataclass
class RunMetrics:
    run_id: str
    cycle_date: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # ToT reasoning process quality
    branch_d_promotion_rate: float = 0.0
    counter_evidence_completeness: float = 0.0
    critic_score_variance: float = 0.0
    reeval_trigger_count: int = 0

    # Operational reliability
    source_availability: float = 1.0
    staleness_flag_count: int = 0
    end_to_end_latency_s: float = 0.0
    category_count: int = 0

    # Escalation
    escalation_level: str = "NONE"
    escalation_triggers: list[str] = field(default_factory=list)

    # Citation audit
    citation_audit_passed: bool = True

    # Feedback-derived (populated retroactively via feedback mechanism)
    forecast_accuracy: Optional[float] = None
    confidence_calibration: Optional[float] = None
    engineering_leader_agree: Optional[float] = None


class MetricsStore:
    """Appends RunMetrics to a JSON-lines file and reads history for analysis."""

    def __init__(self, path: Path = METRICS_PATH):
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, metrics: RunMetrics) -> None:
        """Append this run's metrics to the store."""
        with open(self._path, "a") as f:
            f.write(json.dumps(asdict(metrics)) + "\n")

    def load_all(self) -> list[dict]:
        """Return all recorded metrics as a list of dicts, oldest first."""
        if not self._path.exists():
            return []
        records = []
        for line in self._path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return records

    def get_trailing_forecast_accuracy(self, n: int = 10) -> Optional[float]:
        """Return the mean forecast_accuracy over the last n completed runs.

        Returns None if no runs with feedback exist yet.
        """
        all_records = self.load_all()
        accuracy_values = [
            r["forecast_accuracy"]
            for r in all_records[-n:]
            if r.get("forecast_accuracy") is not None
        ]
        if not accuracy_values:
            return None
        return round(sum(accuracy_values) / len(accuracy_values), 3)

    def get_trailing_branch_d_rate(self, n: int = 10) -> float:
        """Return the mean branch_d_promotion_rate over the last n runs."""
        all_records = self.load_all()
        values = [r["branch_d_promotion_rate"] for r in all_records[-n:]]
        return round(sum(values) / len(values), 3) if values else 0.0

    def summary(self) -> dict:
        """Return a compact summary of the metrics store."""
        records = self.load_all()
        if not records:
            return {"total_runs": 0}
        latest = records[-1]
        return {
            "total_runs": len(records),
            "latest_run_id": latest.get("run_id"),
            "latest_cycle": latest.get("cycle_date"),
            "trailing_branch_d_rate": self.get_trailing_branch_d_rate(),
            "trailing_forecast_accuracy": self.get_trailing_forecast_accuracy(),
            "avg_latency_s": round(
                sum(r.get("end_to_end_latency_s", 0) for r in records) / len(records), 1
            ),
        }


def compute_run_metrics(
    pkg,              # SynthesisPackage
    source_availability: float,
    staleness_flag_count: int,
    latency_s: float,
    run_id: str,
    cycle_date: str,
) -> RunMetrics:
    """Derive RunMetrics from a completed SynthesisPackage.

    This is called by the Orchestrator after the full pipeline completes.
    """
    from schemas.messages import InvestmentStance, BranchType

    results = pkg.category_results
    n = max(len(results), 1)

    # Branch D promotion rate
    branch_d_count = sum(
        1 for r in results
        if r.stance == InvestmentStance.MANUAL_REVIEW
        or r.winning_branch == BranchType.EVIDENCE_INSUFFICIENT
    )
    branch_d_rate = round(branch_d_count / n, 3)

    # Counter-evidence completeness (from pruned_nodes across all categories)
    all_nodes = [node for r in results for node in (r.pruned_nodes or [])]
    zeroed = sum(1 for n_score in all_nodes if n_score.counter_evidence_zeroed)
    total_nodes = max(len(all_nodes), 1)
    completeness = round(1.0 - zeroed / total_nodes, 3)

    # Critic score variance: std-dev of totals across all scored nodes
    all_totals = [n_score.total for r in results for n_score in (r.pruned_nodes or [])]
    if len(all_totals) >= 2:
        mean = sum(all_totals) / len(all_totals)
        variance = sum((x - mean) ** 2 for x in all_totals) / len(all_totals)
        score_variance = round(math.sqrt(variance), 2)
    else:
        score_variance = 0.0

    return RunMetrics(
        run_id=run_id,
        cycle_date=cycle_date,
        branch_d_promotion_rate=branch_d_rate,
        counter_evidence_completeness=completeness,
        critic_score_variance=score_variance,
        reeval_trigger_count=pkg.reeval_trigger_count,
        source_availability=source_availability,
        staleness_flag_count=staleness_flag_count,
        end_to_end_latency_s=round(latency_s, 1),
        category_count=len(results),
        escalation_level=pkg.escalation.level.value,
        escalation_triggers=list(pkg.escalation.triggers),
        citation_audit_passed=pkg.citation_audit_passed,
    )


if __name__ == "__main__":
    print("=== metrics local test ===")
    store = MetricsStore(Path("/tmp/test_metrics_store.json"))
    m = RunMetrics(
        run_id="test-001", cycle_date="2026-06-16",
        branch_d_promotion_rate=0.2,
        counter_evidence_completeness=0.85,
        critic_score_variance=12.5,
        reeval_trigger_count=1,
        source_availability=0.9,
        staleness_flag_count=1,
        end_to_end_latency_s=143.2,
        category_count=5,
        escalation_level="LEVEL_1_FLAG",
    )
    store.record(m)
    print(f"Recorded: {m.run_id}")
    print(f"Summary: {store.summary()}")

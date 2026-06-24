"""Investment Signal Agent — main entry point.

7-Agent Architecture (primary path):
  Delegates to OrchestratorAgent which coordinates all seven agents and
  produces a Technology Investment Horizon Report for engineering leaders.

  Agents:
    1. Orchestrator         — task routing, weighted scoring, ToT coordination
    2. Data Collector       — parallel async collection (GitHub, news, trends, macro)
    3. Retrieval Agent      — SEC EDGAR semantic search (FAISS, ≥0.72 threshold)
    4. Signal Analyst       — ALIGNED / CONTRADICTORY classification
    5. Thought Generator    — 4 canonical ToT branches with counter_evidence
    6. Critic               — 5-criterion rubric scoring (0-20 each, 100 max)
    7. Synthesis Agent      — writes the final Markdown report

Usage:
  python main.py                         # run all five categories
  python main.py ai_ml_infrastructure    # single category
  python main.py ai_ml_infrastructure semiconductors  # subset of categories
"""

import argparse
import sys

from agents.orchestrator_agent import run_orchestrator
from signals.category_registry import get_category_ids


def main() -> None:
    valid_ids = get_category_ids()

    parser = argparse.ArgumentParser(
        description="Technology Investment Horizon Report — 7-Agent Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Category IDs:\n"
            + "\n".join(f"  {cid}" for cid in valid_ids)
        ),
    )
    parser.add_argument(
        "categories",
        nargs="*",
        metavar="CATEGORY_ID",
        help=(
            "One or more category IDs to analyse (default: all five). "
            f"Valid IDs: {', '.join(valid_ids)}"
        ),
    )
    args = parser.parse_args()

    requested = args.categories or None  # None → Orchestrator runs all

    if requested:
        unknown = [c for c in requested if c not in valid_ids]
        if unknown:
            print(f"[error] Unknown category IDs: {unknown}")
            print(f"        Valid IDs: {valid_ids}")
            sys.exit(1)

    report_path = run_orchestrator(category_ids=requested)
    print(f"\nReport written → {report_path}")


if __name__ == "__main__":
    main()

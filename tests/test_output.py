"""Local tests for the output/ package (no LLM or API keys required)."""

import sys
import os
import tempfile
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import output.investment_brief as brief_module


def test_generate_brief():
    print("\n--- investment_brief ---")
    original_dir = brief_module.REPORTS_DIR

    with tempfile.TemporaryDirectory() as tmpdir:
        brief_module.REPORTS_DIR = Path(tmpdir)

        sample = (
            "## Executive Summary\nNVDA shows record AI chip demand.\n\n"
            "## Recommended Action\nBUY — 2% position.\n\n"
            "## Conviction Level\nHIGH\n\n"
            "## Key Risks\n- Export controls\n- Valuation\n\n"
            "## Monitoring Triggers\n- Gross margin < 70%\n"
        )
        path = brief_module.generate_brief(
            "NVDA", sample, signal_score=82.0, alignment_label="BULL"
        )
        assert path.exists(), "Brief file was not created"
        content = path.read_text()
        assert "NVDA" in content
        assert "82.0/100" in content
        assert "BULL" in content
        print(f"  ✓ Brief written to: {path.name}")

        reports = brief_module.list_reports("NVDA")
        assert len(reports) == 1
        print(f"  ✓ list_reports returned {len(reports)} file(s)")

    brief_module.REPORTS_DIR = original_dir


if __name__ == "__main__":
    print("=" * 50)
    print("  TEST: output/")
    print("=" * 50)
    test_generate_brief()
    print("\n✅ All output tests passed.")

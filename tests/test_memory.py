"""Local tests for the memory/ package (no LLM or API keys required)."""

import sys
import os
import tempfile
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory


def test_short_term():
    print("\n--- short_term memory ---")
    mem = ShortTermMemory()
    assert len(mem) == 0

    mem.add("What is NVDA revenue growth?", "427% YoY to $22.6B.")
    mem.add("What are the key risks?", "Export controls, valuation.")
    assert len(mem) == 4  # 2 human + 2 AI messages
    print(f"  ✓ Stored {len(mem)} messages")

    transcript = mem.as_string()
    assert "Human:" in transcript and "AI:" in transcript
    print("  ✓ as_string() works")

    mem.clear()
    assert len(mem) == 0
    print("  ✓ Clear works")


def test_long_term():
    print("\n--- long_term memory ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        store = Path(tmpdir) / "test_store.json"
        mem = LongTermMemory(store_path=store)
        assert len(mem) == 0

        rec = mem.store_run(
            ticker="NVDA",
            signal_score=82.0,
            stance="Bullish",
            recommendation="BUY",
            summary="Record AI chip demand.",
        )
        assert rec["ticker"] == "NVDA"
        assert rec["id"] == 1
        print(f"  ✓ Stored run #{rec['id']} for {rec['ticker']}")

        mem2 = LongTermMemory(store_path=store)
        assert len(mem2) == 1
        print("  ✓ Persistence across reload works")

        mem2.store_run("MSFT", 71.0, "Neutral", "HOLD", "Cloud growth solid.")
        assert len(mem2.get_by_ticker("NVDA")) == 1
        assert len(mem2.get_by_ticker("MSFT")) == 1
        print("  ✓ Ticker filtering works")

        assert len(mem2.get_latest(5)) == 2
        print("  ✓ get_latest works")


if __name__ == "__main__":
    print("=" * 50)
    print("  TEST: memory/")
    print("=" * 50)
    test_short_term()
    test_long_term()
    print("\n✅ All memory tests passed.")

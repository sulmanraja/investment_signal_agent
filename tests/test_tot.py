"""Local tests for the tot/ package (no LLM required for node/pruner tests)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tot.thought_node import ThoughtNode
from tot.pruner import prune_beam


def test_thought_node():
    print("\n--- thought_node ---")
    root = ThoughtNode(ticker="NVDA", branch="A", depth=0, content="Bullish: AI demand.", score=8.5)
    child = ThoughtNode(ticker="NVDA", branch="A", depth=1, content="Data-center +427% YoY.", score=9.0, parent=root)
    root.children.append(child)

    assert root.is_evaluated()
    assert len(root.children) == 1

    lineage = child.lineage()
    assert len(lineage) == 2
    assert lineage[0] is root
    assert lineage[1] is child
    print(f"  ✓ Lineage depth: {len(lineage)}")

    unscored = ThoughtNode(ticker="NVDA", branch="B", depth=0, content="Bearish.")
    assert not unscored.is_evaluated()
    print("  ✓ Unscored node detection works")


def test_pruner():
    print("\n--- pruner ---")
    # Scores on 0-100 scale; floor is 40.0
    nodes = [
        ThoughtNode("NVDA", "A", 0, "Bullish",     score=85.0),
        ThoughtNode("NVDA", "B", 0, "Bearish",     score=62.0),
        ThoughtNode("NVDA", "C", 0, "Neutral",     score=71.0),
        ThoughtNode("NVDA", "D", 0, "Contrarian",  score=58.0),
    ]
    survivors, pruned, d_promoted = prune_beam(nodes, beam_width=2)
    assert len(survivors) == 2
    assert not d_promoted
    assert survivors[0].branch == "A"  # highest score
    assert survivors[1].branch == "C"  # second highest
    print(f"  ✓ Beam width=2 kept: {[n.branch for n in survivors]}")

    # With unscored nodes (score=None → treated as 0, falls below floor)
    mixed = [
        ThoughtNode("NVDA", "A", 0, "Bullish"),
        ThoughtNode("NVDA", "B", 0, "Bearish", score=90.0),
    ]
    survivors2, _, _ = prune_beam(mixed, beam_width=1)
    assert survivors2[0].branch == "B"
    print("  ✓ None scores treated as 0 in pruning")


if __name__ == "__main__":
    print("=" * 50)
    print("  TEST: tot/ (no LLM)")
    print("=" * 50)
    test_thought_node()
    test_pruner()
    print("\n✅ All ToT structural tests passed.")
    print("\nNote: test branches/evaluator/beam_search require Ollama running locally.")
    print("  Run: python -m tot.branches   (requires Ollama)")
    print("  Run: python -m tot.beam_search (requires Ollama)")

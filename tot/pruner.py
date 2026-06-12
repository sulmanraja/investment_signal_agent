"""Pruner — keeps the top-K nodes per beam search step.

Default beam width = 2: only the two highest-scoring branches survive
to the next expansion depth.
"""

from tot.thought_node import ThoughtNode

BEAM_WIDTH = 2


def prune_beam(nodes: list[ThoughtNode], beam_width: int = BEAM_WIDTH) -> list[ThoughtNode]:
    """Return the top `beam_width` nodes sorted by score descending.

    Nodes without a score (None) are treated as score = 0.

    Args:
        nodes: List of ThoughtNode instances to prune.
        beam_width: Maximum nodes to keep.

    Returns:
        Pruned list of at most `beam_width` nodes.
    """
    scored = sorted(nodes, key=lambda n: n.score or 0.0, reverse=True)
    return scored[:beam_width]


if __name__ == "__main__":
    from tot.thought_node import ThoughtNode
    print("=== pruner local test ===")
    nodes = [
        ThoughtNode("NVDA", "A", 0, "Bullish stance", score=8.5),
        ThoughtNode("NVDA", "B", 0, "Bearish stance", score=6.2),
        ThoughtNode("NVDA", "C", 0, "Neutral stance", score=7.1),
        ThoughtNode("NVDA", "D", 0, "Contrarian stance", score=5.8),
    ]
    kept = prune_beam(nodes, beam_width=2)
    print(f"Kept {len(kept)} of {len(nodes)} nodes:")
    for n in kept:
        print(f"  {n}")

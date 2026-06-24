"""Pruner — hard floor + beam-width filtering for ToT nodes.

Rules (per claud.md spec):
  1. Hard floor at 40/100: any node below 40 is eliminated regardless.
  2. Beam width = 2: of the surviving nodes, keep at most the top 2 by score.
  3. Branch D auto-promote: if zero nodes survive after the floor + beam pass,
     auto-promote the Branch D (Evidence-Insufficient) node from the input list
     rather than returning an empty beam.

Returns (survivors, pruned, branch_d_promoted) so callers can detect the
auto-promote case and apply termination logic.
"""

from tot.thought_node import ThoughtNode

BEAM_WIDTH = 2
FLOOR_SCORE = 40.0


def prune_beam(
    nodes: list[ThoughtNode],
    beam_width: int = BEAM_WIDTH,
    floor_score: float = FLOOR_SCORE,
) -> tuple[list[ThoughtNode], list[ThoughtNode], bool]:
    """Apply hard floor + beam-width pruning.

    Args:
        nodes: All evaluated ThoughtNode instances at this depth.
        beam_width: Maximum survivors to keep.
        floor_score: Minimum score to survive (exclusive lower bound).

    Returns:
        (survivors, pruned_nodes, branch_d_promoted)
        - survivors: nodes that passed the floor and fit the beam.
        - pruned_nodes: nodes eliminated (below floor or out of beam).
        - branch_d_promoted: True if no survivors and Branch D was auto-promoted.
    """
    sorted_nodes = sorted(nodes, key=lambda n: n.score or 0.0, reverse=True)

    above_floor = [n for n in sorted_nodes if (n.score or 0.0) >= floor_score]
    below_floor = [n for n in sorted_nodes if (n.score or 0.0) < floor_score]

    survivors = above_floor[:beam_width]
    out_of_beam = above_floor[beam_width:]
    pruned = below_floor + out_of_beam

    if survivors:
        return survivors, pruned, False

    # Auto-promote Branch D
    branch_d = next((n for n in sorted_nodes if n.branch == "D"), None)
    if branch_d:
        return [branch_d], [n for n in sorted_nodes if n is not branch_d], True

    # Last resort: return empty (caller must handle)
    return [], sorted_nodes, True


if __name__ == "__main__":
    print("=== pruner local test ===")
    nodes = [
        ThoughtNode("ai_ml", "A", 0, "Capital-Led thesis", score=72.0),
        ThoughtNode("ai_ml", "B", 0, "Adoption-Led thesis", score=58.0),
        ThoughtNode("ai_ml", "C", 0, "Risk-Adjusted thesis", score=35.0),  # below floor
        ThoughtNode("ai_ml", "D", 0, "Evidence-Insufficient", score=28.0),  # below floor
    ]
    survivors, pruned, d_promoted = prune_beam(nodes)
    print(f"Survivors ({len(survivors)}): {[f'{n.branch}={n.score}' for n in survivors]}")
    print(f"Pruned   ({len(pruned)}):    {[f'{n.branch}={n.score}' for n in pruned]}")
    print(f"Branch D promoted: {d_promoted}")

    print("\n--- edge case: all below floor → Branch D auto-promote ---")
    all_low = [
        ThoughtNode("ai_ml", "A", 0, "Bullish", score=20.0),
        ThoughtNode("ai_ml", "B", 0, "Growth", score=18.0),
        ThoughtNode("ai_ml", "C", 0, "Cautious", score=15.0),
        ThoughtNode("ai_ml", "D", 0, "Ambiguous", score=22.0),
    ]
    s, p, promoted = prune_beam(all_low)
    print(f"Survivors: {[n.branch for n in s]}  promoted={promoted}")

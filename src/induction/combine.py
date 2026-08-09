"""Induction + transduction pass@2 combination (BARC's "combine induction and
transduction" recipe, offline).

Induction yields test predictions from programs that reproduce ALL train pairs
(high precision but not perfect on ARC-AGI-2). Neural yields ranked candidates
from TTFT+DFS+voting. The pass@2-optimal policy DIVERSIFIES the two slots: give
induction one slot when it has a verified answer, neural the other — so we score
whenever EITHER source is right, not just when they agree.
"""
from __future__ import annotations

from collections import Counter
from typing import List, Optional
from arc.grids import Grid


def _key(g: Grid):
    return tuple(map(tuple, g))


def rank_induction(verified_test_preds: List[Grid]) -> List[Grid]:
    """Distinct verified predictions ranked by program-consensus (frequency)."""
    cnt = Counter(_key(g) for g in verified_test_preds)
    return [list(map(list, k)) for k, _ in cnt.most_common()]


def combine_pass2(neural_ranked: List[Grid], induction_verified: List[Grid],
                  min_induction_consensus: int = 1) -> List[Grid]:
    """Return 2 predictions [attempt_1, attempt_2].

    - attempt_1 = top induction-verified prediction if it has >= min_induction_consensus
      supporting programs, else the top neural candidate.
    - attempt_2 = the top neural candidate distinct from attempt_1; fall backs keep
      both slots filled and distinct where possible.
    """
    ind_ranked = rank_induction(induction_verified)
    ind_counts = Counter(_key(g) for g in induction_verified)

    picks: List[Grid] = []

    # slot 1: induction if it has enough consensus, else neural
    if ind_ranked and ind_counts[_key(ind_ranked[0])] >= min_induction_consensus:
        picks.append(ind_ranked[0])
    elif neural_ranked:
        picks.append(neural_ranked[0])

    # slot 2: best neural distinct from slot 1
    for g in neural_ranked:
        if not picks or g != picks[0]:
            picks.append(g)
            break

    # fallbacks: second induction, then duplicate, to always return 2
    if len(picks) < 2:
        for g in ind_ranked:
            if not picks or g != picks[0]:
                picks.append(g); break
    while len(picks) < 2:
        picks.append(picks[0] if picks else [[0]])
    return picks[:2]

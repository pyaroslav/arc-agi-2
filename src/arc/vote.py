"""Augment-and-vote selection: re-score candidates with m shared augmentations
and pick the top-2 for the pass@2 submission.

score(s) = generation_count(s) + (1/m) * sum_j logP_j(s)

where logP_j is the model's mean per-token log-prob of candidate ``s`` under the
j-th augmentation (NVARC's geometric-mean ensemble of per-augmentation
likelihoods, plus the DFS/sample vote count).
"""
from __future__ import annotations

from typing import List, Dict, Tuple
import numpy as np

from .model import ModelBundle, sequence_logprob
from .grids import Grid
from .data import Augmentation, sample_augmentations
from .decode import _aug_context, _key


def score_candidates(bundle: ModelBundle, train_pairs: List[dict], test_input: Grid,
                     candidates: Dict[Tuple, Tuple[Grid, int]],
                     score_augs: List[Augmentation]) -> List[Tuple[float, int, Grid]]:
    """Return [(score, count, grid)] sorted descending."""
    scored = []
    m = max(1, len(score_augs))
    for k, (grid, count) in candidates.items():
        lp_sum = 0.0
        for aug in score_augs:
            ctx = _aug_context(train_pairs, aug)
            lp_sum += sequence_logprob(bundle, ctx, aug.apply_grid(test_input), aug.apply_grid(grid))
        score = count + lp_sum / m
        scored.append((score, count, grid))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored


def select_pass2(bundle: ModelBundle, train_pairs: List[dict], test_input: Grid,
                 candidates: Dict[Tuple, Tuple[Grid, int]], num_augmentations: int,
                 seed: int = 0, fallback: Grid = None) -> List[Grid]:
    """Pick 2 distinct predictions. Falls back to a default grid if <2 candidates."""
    if candidates:
        rng = np.random.default_rng(seed)
        n_demos = len(train_pairs)
        score_augs = sample_augmentations(num_augmentations, rng, n_demos=n_demos,
                                          include_identity=True)
        scored = score_candidates(bundle, train_pairs, test_input, candidates, score_augs)
        preds = [g for _, _, g in scored[:2]]
    else:
        preds = []
    if not fallback:
        fallback = test_input  # last-resort guess
    while len(preds) < 2:
        # second guess defaults to fallback if distinct, else a copy
        preds.append(fallback if fallback not in preds else (preds[0] if preds else fallback))
    return preds[:2]

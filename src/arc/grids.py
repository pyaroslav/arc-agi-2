"""Grid primitives + invertible geometric (D4) and color transforms.

A *grid* is a rectangular ``list[list[int]]`` with values in 0..9 (ARC colors).
All transforms here are **invertible** so that a prediction made in an augmented
frame can be mapped back to the original frame (the core trick behind
augmentation-and-vote inference, NVARC/ARChitects).
"""
from __future__ import annotations

from typing import List
import numpy as np

Grid = List[List[int]]
NUM_COLORS = 10  # ARC uses colors 0..9


# --------------------------------------------------------------------------- #
# Basic helpers
# --------------------------------------------------------------------------- #
def to_np(grid: Grid) -> np.ndarray:
    return np.asarray(grid, dtype=np.int64)


def to_list(arr: np.ndarray) -> Grid:
    return [[int(c) for c in row] for row in arr.tolist()]


def is_valid_grid(grid) -> bool:
    """Rectangular, non-empty, all cells int in 0..9."""
    if not isinstance(grid, list) or len(grid) == 0:
        return False
    if not all(isinstance(r, list) and len(r) > 0 for r in grid):
        return False
    w = len(grid[0])
    if any(len(r) != w for r in grid):
        return False
    return all(isinstance(c, int) and 0 <= c < NUM_COLORS for r in grid for c in r)


def grids_equal(a, b) -> bool:
    return a == b


# --------------------------------------------------------------------------- #
# D4 dihedral group (8 elements) — see docstring in grids.apply_d4
# --------------------------------------------------------------------------- #
# Each element is parameterized as (flip, rot): optionally transpose (reflect
# across the main diagonal), then rotate counter-clockwise by ``rot`` * 90deg.
# (flip in {0,1}) x (rot in {0,1,2,3}) enumerates all 8 symmetries exactly once.
D4_ELEMENTS = [(flip, rot) for flip in (0, 1) for rot in (0, 1, 2, 3)]


def apply_d4(grid: Grid, flip: int, rot: int) -> Grid:
    a = to_np(grid)
    if flip:
        a = a.T
    a = np.rot90(a, k=rot % 4)
    return to_list(a)


def invert_d4(grid: Grid, flip: int, rot: int) -> Grid:
    """Inverse of :func:`apply_d4` for the same (flip, rot).

    apply does  rot90(T?(g), rot); invert undoes rotation first, then transpose.
    This composition is the exact inverse for every (flip, rot) (verified in tests).
    """
    a = to_np(grid)
    a = np.rot90(a, k=(4 - rot % 4) % 4)
    if flip:
        a = a.T
    return to_list(a)


# --------------------------------------------------------------------------- #
# Color permutations
# --------------------------------------------------------------------------- #
def random_color_perm(rng: np.random.Generator, fix_background: bool = False) -> List[int]:
    """Return a permutation ``perm`` of colors where cell value c -> perm[c].

    If ``fix_background`` keep color 0 mapped to 0 (ARC background convention),
    permuting only 1..9. Otherwise permute all 10 colors.
    """
    if fix_background:
        rest = list(range(1, NUM_COLORS))
        rng.shuffle(rest)
        return [0] + rest
    perm = list(range(NUM_COLORS))
    rng.shuffle(perm)
    return perm


def invert_perm(perm: List[int]) -> List[int]:
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


def apply_color(grid: Grid, perm: List[int]) -> Grid:
    return [[perm[c] for c in row] for row in grid]


def invert_color(grid: Grid, perm: List[int]) -> Grid:
    return apply_color(grid, invert_perm(perm))

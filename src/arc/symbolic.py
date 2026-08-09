"""Compact symbolic ARC solver (roadmap #5 probe).

A small library of parameter-learning rules. `synthesize(train_pairs)` returns
every program whose transform reproduces ALL train outputs exactly; `predict`
applies it to a test input. Used to (a) measure how much net-new coverage a cheap
DSL adds on ARC-AGI-2 (the benchmark was built to defeat brute-force DSL, so we
EXPECT little — this quantifies it) and (b) feed verified candidates / soft
tiebreaks into the neural vote.

Each rule is `(name, fit(train_pairs) -> params|None, apply(grid, params) -> grid|None)`.
Programs are single rules plus a few depth-2 compositions (D4 ∘ colormap).
"""
from __future__ import annotations

from typing import List, Optional, Callable, Tuple
import numpy as np

from .grids import to_np, to_list, Grid, D4_ELEMENTS, apply_d4

Pair = dict  # {"input": Grid, "output": Grid}


# --------------------------------------------------------------------------- #
# individual transforms (grid -> grid or None)
# --------------------------------------------------------------------------- #
def _d4(grid: Grid, fr) -> Grid:
    return apply_d4(grid, fr[0], fr[1])


def _learn_colormap(pairs: List[Pair]) -> Optional[dict]:
    """A single global color->color map consistent across all train pairs
    (shapes must already match input==output)."""
    m = {}
    for p in pairs:
        a, b = to_np(p["input"]), to_np(p["output"])
        if a.shape != b.shape:
            return None
        for x, y in zip(a.flatten(), b.flatten()):
            x, y = int(x), int(y)
            if x in m and m[x] != y:
                return None
            m[x] = y
    return m


def _apply_colormap(grid: Grid, m: dict) -> Grid:
    a = to_np(grid)
    out = np.vectorize(lambda c: m.get(int(c), int(c)))(a)
    return to_list(out)


def _tiling_factors(pairs: List[Pair]) -> Optional[Tuple[int, int]]:
    fr = None
    for p in pairs:
        a, b = np.shape(p["input"]), np.shape(p["output"])
        if a[0] == 0 or a[1] == 0 or b[0] % a[0] or b[1] % a[1]:
            return None
        f = (b[0] // a[0], b[1] // a[1])
        if fr is None:
            fr = f
        elif fr != f:
            return None
    return fr if fr and fr != (1, 1) else None


def _apply_tile(grid: Grid, fr) -> Grid:
    return to_list(np.tile(to_np(grid), fr))


def _apply_scale(grid: Grid, fr) -> Grid:
    a = to_np(grid)
    return to_list(np.kron(a, np.ones(fr, dtype=a.dtype)))


def _crop_bbox(grid: Grid, bg) -> Optional[Grid]:
    a = to_np(grid)
    mask = a != bg
    if not mask.any():
        return None
    r0, r1 = np.where(mask.any(1))[0][[0, -1]]
    c0, c1 = np.where(mask.any(0))[0][[0, -1]]
    return to_list(a[r0:r1 + 1, c0:c1 + 1])


# --------------------------------------------------------------------------- #
# rule registry: each returns list of (apply_fn) programs consistent with train
# --------------------------------------------------------------------------- #
def _consistent(pairs, fn) -> bool:
    for p in pairs:
        try:
            out = fn(p["input"])
        except Exception:
            return False
        if out is None or out != p["output"]:
            return False
    return True


def synthesize(train_pairs: List[Pair]) -> List[Callable[[Grid], Grid]]:
    """Return programs (callables) reproducing all train pairs exactly."""
    progs: List[Callable] = []

    # 1) D4 geometric transforms
    for fr in D4_ELEMENTS:
        fn = (lambda g, fr=fr: _d4(g, fr))
        if _consistent(train_pairs, fn):
            progs.append(fn)

    # 2) global color map (same shape)
    m = _learn_colormap(train_pairs)
    if m is not None:
        fn = (lambda g, m=m: _apply_colormap(g, m))
        if _consistent(train_pairs, fn):
            progs.append(fn)

    # 3) depth-2: D4 then color map  (learn map AFTER applying the transform)
    for fr in D4_ELEMENTS:
        if fr == (0, 0):
            continue
        tpairs = [{"input": _d4(p["input"], fr), "output": p["output"]} for p in train_pairs]
        m2 = _learn_colormap(tpairs)
        if m2 is not None:
            fn = (lambda g, fr=fr, m2=m2: _apply_colormap(_d4(g, fr), m2))
            if _consistent(train_pairs, fn):
                progs.append(fn)

    # 4) tiling / scaling
    fr = _tiling_factors(train_pairs)
    if fr:
        for ap in (_apply_tile, _apply_scale):
            fn = (lambda g, fr=fr, ap=ap: ap(g, fr))
            if _consistent(train_pairs, fn):
                progs.append(fn)

    # 5) crop to bounding box (try bg = 0 and most-common color)
    for bgsel in ("zero", "mode"):
        def fn(g, bgsel=bgsel):
            a = to_np(g)
            bg = 0 if bgsel == "zero" else int(np.bincount(a.flatten()).argmax())
            return _crop_bbox(g, bg)
        if _consistent(train_pairs, fn):
            progs.append(fn)

    return progs


def predict(train_pairs: List[Pair], test_input: Grid) -> List[Grid]:
    """All distinct test predictions from train-consistent programs (precision-first)."""
    outs, seen = [], set()
    for prog in synthesize(train_pairs):
        try:
            o = prog(test_input)
        except Exception:
            continue
        if o is None:
            continue
        k = tuple(map(tuple, o))
        if k not in seen:
            seen.add(k)
            outs.append(o)
    return outs

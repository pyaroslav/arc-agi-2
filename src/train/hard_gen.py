"""Generate HARDER synthetic ARC tasks the base model hasn't mastered.

Two streams:
  1) COMPOSITIONAL: chain 1-3 parametric primitives (geometry, color-map, tiling,
     scaling, mirror, gravity, crop, border, recolor-by-size) into a single fixed
     rule, then render k input->output demo pairs. Composition is what ARC-AGI-2
     tests and what single-transform synthetic data underteaches.
  2) HARDEN: take an existing task and add INPUT-side complexity (edge padding,
     color splitting, separators) while keeping the transformation valid — the
     2025 winner's "augment in a different manner than rotation/colour" idea.

Output: ARC-style {"pairs":[{input,output}...]} JSON, consumable by sft_data.py.
"""
from __future__ import annotations

import numpy as np
from typing import List, Callable

BG = 0
MAXD = 30


# --------------------------------------------------------------------------- #
# input-grid generators
# --------------------------------------------------------------------------- #
def rand_scatter(rng, h, w, ncol=4, density=0.3):
    g = np.zeros((h, w), int)
    cols = rng.choice(range(1, 10), size=min(ncol, 9), replace=False)
    mask = rng.random((h, w)) < density
    g[mask] = rng.choice(cols, size=int(mask.sum()))
    return g


def rand_rects(rng, h, w, n=3):
    g = np.zeros((h, w), int)
    for _ in range(n):
        c = int(rng.integers(1, 10))
        r0, r1 = sorted(rng.integers(0, h, 2)); c0, c1 = sorted(rng.integers(0, w, 2))
        g[r0:r1 + 1, c0:c1 + 1] = c
    return g


def rand_input(rng):
    h, w = int(rng.integers(3, 12)), int(rng.integers(3, 12))
    return (rand_scatter if rng.random() < 0.5 else rand_rects)(rng, h, w)


# --------------------------------------------------------------------------- #
# parametric primitives: sampler returns params; apply(grid, params) -> grid
# --------------------------------------------------------------------------- #
def _present(g):
    return [c for c in range(1, 10) if (g == c).any()]


PRIMS: List[tuple] = []


def prim(name):
    def deco(fn):
        PRIMS.append((name, fn))
        return fn
    return deco


@prim("d4")
def _d4(rng):
    flip, rot = int(rng.integers(0, 2)), int(rng.integers(1, 4))
    return lambda g: (np.rot90(g.T if flip else g, rot))


@prim("colormap")
def _colormap(rng):
    perm = rng.permutation(10)
    return lambda g: perm[g]


@prim("tile")
def _tile(rng):
    fh, fw = int(rng.integers(1, 4)), int(rng.integers(1, 4))
    if fh == 1 and fw == 1:
        fw = 2
    return lambda g: (np.tile(g, (fh, fw)) if g.shape[0]*fh <= MAXD and g.shape[1]*fw <= MAXD else g)


@prim("scale")
def _scale(rng):
    f = int(rng.integers(2, 4))
    return lambda g: (np.kron(g, np.ones((f, f), int)) if g.shape[0]*f <= MAXD and g.shape[1]*f <= MAXD else g)


@prim("mirror")
def _mirror(rng):
    axis = int(rng.integers(0, 2))
    return lambda g: (np.concatenate([g, np.flip(g, axis)], axis) if g.shape[axis]*2 <= MAXD else g)


@prim("gravity")
def _gravity(rng):
    down = rng.random() < 0.5
    def f(g):
        out = np.zeros_like(g)
        for j in range(g.shape[1]):
            col = g[:, j]; nz = col[col != BG]
            if down:
                out[g.shape[0]-len(nz):, j] = nz
            else:
                out[:len(nz), j] = nz
        return out
    return f


@prim("border")
def _border(rng):
    c = int(rng.integers(1, 10))
    return lambda g: (np.pad(g, 1, constant_values=c) if max(g.shape)+2 <= MAXD else g)


@prim("recolor_size")
def _recolor_size(rng):
    # recolor each connected mono component by a color keyed on its size parity
    from scipy.ndimage import label
    a_col, b_col = int(rng.integers(1, 10)), int(rng.integers(1, 10))
    def f(g):
        out = g.copy()
        for c in _present(g):
            lab, n = label(g == c)
            for k in range(1, n + 1):
                sz = (lab == k).sum()
                out[lab == k] = a_col if sz % 2 == 0 else b_col
        return out
    return f


def sample_program(rng, depth):
    fns = [PRIMS[i][1](rng) for i in rng.choice(len(PRIMS), size=depth, replace=True)]
    def prog(g):
        for fn in fns:
            g = np.asarray(fn(g), int)
            if g.size == 0 or g.max() > 9 or g.min() < 0 or max(g.shape) > MAXD:
                raise ValueError
        return g
    return prog


# --------------------------------------------------------------------------- #
def gen_compositional_task(rng, k=4, depth=None, tries=40):
    depth = depth or int(rng.integers(1, 4))
    for _ in range(tries):
        prog = sample_program(rng, depth)
        pairs, ok = [], True
        for _ in range(k):
            inp = rand_input(rng)
            try:
                out = prog(inp)
            except Exception:
                ok = False; break
            if np.array_equal(out, inp) and rng.random() < 0.7:
                ok = False; break          # mostly drop trivial identity rules
            pairs.append({"input": inp.tolist(), "output": out.tolist()})
        if ok and len(pairs) == k and len({tuple(np.shape(p["output"])) for p in pairs}) <= 3:
            return {"pairs": pairs}
    return None


def harden_task(task, rng):
    """Input-side complexity augmentation that preserves the rule is hard to do
    generically; here we apply a shared color-split to BOTH input and output
    (a valid relabeling) + optional shared geometric transform — producing a novel
    but valid variant the base hasn't memorized."""
    perm = rng.permutation(10)
    flip, rot = int(rng.integers(0, 2)), int(rng.integers(0, 4))

    def t(grid):
        a = perm[np.asarray(grid, int)]
        if flip:
            a = a.T
        return np.rot90(a, rot)
    out = {"pairs": []}
    for p in task["pairs"]:
        out["pairs"].append({"input": t(p["input"]).tolist(), "output": t(p["output"]).tolist()})
    return out

"""SDG batch 1: compositional, context-dependent ARC task generators.

Unlike hard_gen.py (procedural op-chains — measured null), each concept here
requires *interpreting* the grid (legends, selectors, object properties,
interactions) before applying a transform. Outputs are correct by construction
(computed by the concept's transform on sampled inputs).

  python -m train.sdg_gen --out data/sdg/batch01 --per-concept 50 [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def _place(grid, h, w, rng, tries=200):
    """Random top-left for an h x w box not overlapping nonzero cells."""
    H, W = grid.shape
    for _ in range(tries):
        r = rng.integers(0, H - h + 1)
        c = rng.integers(0, W - w + 1)
        if np.all(grid[r:r + h, c:c + w] == 0):
            return r, c
    return None


def _objects(grid):
    """4-connected components of nonzero cells -> list of (color, cells)."""
    H, W = grid.shape
    seen = np.zeros_like(grid, dtype=bool)
    out = []
    for r in range(H):
        for c in range(W):
            if grid[r, c] and not seen[r, c]:
                col = grid[r, c]
                stack, cells = [(r, c)], []
                seen[r, c] = True
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and grid[ny, nx] == col and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                out.append((col, cells))
    return out


def _blob(rng, h, w, color, fill=0.75):
    """Random connected blob in an h x w box."""
    box = np.zeros((h, w), dtype=int)
    r, c = rng.integers(h), rng.integers(w)
    box[r, c] = color
    target = max(2, int(h * w * fill))
    cells = [(r, c)]
    while len(cells) < target:
        y, x = cells[rng.integers(len(cells))]
        dy, dx = ((1, 0), (-1, 0), (0, 1), (0, -1))[rng.integers(4)]
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w:
            if box[ny, nx] == 0:
                box[ny, nx] = color
                cells.append((ny, nx))
        if rng.random() < 0.02:
            break
    return box


# ---------------------------------------------------------------- concepts

def gen_key_legend(rng):
    """A 2-row legend in a corner maps colors; recolor main objects per legend."""
    H, W = int(rng.integers(12, 19)), int(rng.integers(12, 19))
    n_map = int(rng.integers(2, 4))
    colors = rng.permutation(np.arange(1, 10))[: 2 * n_map]
    src, dst = colors[:n_map], colors[n_map:]

    def sample():
        g = np.zeros((H, W), dtype=int)
        g[0, :2 * n_map:2] = src
        g[0, 1:2 * n_map:2] = dst
        g[1, :2 * n_map] = 5  # legend underline marker
        for _ in range(int(rng.integers(3, 6))):
            col = src[rng.integers(n_map)]
            h, w = int(rng.integers(2, 4)), int(rng.integers(2, 4))
            pos = _place(g[2:], h, w, rng)
            if pos:
                g[2 + pos[0]:2 + pos[0] + h, pos[1]:pos[1] + w] = np.where(
                    _blob(rng, h, w, col) > 0, col, 0)
        return g

    def transform(g):
        out = g.copy()
        for s, d in zip(src, dst):
            out[2:][g[2:] == s] = d
        return out

    return sample, transform


def gen_conditional_size(rng):
    """Objects with >= k cells keep color c_big; smaller ones are erased."""
    H, W = int(rng.integers(10, 17)), int(rng.integers(10, 17))
    k = int(rng.integers(4, 7))
    c_big = int(rng.integers(1, 10))

    def sample():
        g = np.zeros((H, W), dtype=int)
        for _ in range(int(rng.integers(4, 8))):
            col = int(rng.integers(1, 10))
            h, w = int(rng.integers(1, 4)), int(rng.integers(1, 4))
            pos = _place(g, h + 1, w + 1, rng)
            if pos:
                b = _blob(rng, h, w, col)
                g[pos[0]:pos[0] + h, pos[1]:pos[1] + w] = b
        return g

    def transform(g):
        out = np.zeros_like(g)
        for col, cells in _objects(g):
            if len(cells) >= k:
                for y, x in cells:
                    out[y, x] = c_big
        return out

    return sample, transform


def gen_gravity_obstacles(rng):
    """Loose cells fall; obstacle color 5 is static; cells stack."""
    H, W = int(rng.integers(9, 15)), int(rng.integers(9, 15))

    def sample():
        g = np.zeros((H, W), dtype=int)
        for _ in range(int(rng.integers(2, 5))):
            w = int(rng.integers(2, max(3, W // 2)))
            r = int(rng.integers(2, H - 1))
            c = int(rng.integers(0, W - w + 1))
            g[r, c:c + w] = 5
        for _ in range(int(rng.integers(4, 9))):
            r, c = int(rng.integers(0, H - 1)), int(rng.integers(0, W))
            if g[r, c] == 0:
                g[r, c] = int(rng.choice([1, 2, 3, 4, 6, 7, 8, 9]))
        return g

    def transform(g):
        out = np.where(g == 5, 5, 0)
        for c in range(g.shape[1]):
            floor = g.shape[0] - 1
            for r in range(g.shape[0] - 1, -1, -1):
                if g[r, c] == 5:
                    floor = r - 1
                elif g[r, c] != 0:
                    while floor >= 0 and out[floor, c] != 0:
                        floor -= 1
                    if floor >= 0:
                        out[floor, c] = g[r, c]
                        floor -= 1
        return out

    return sample, transform


def gen_count_to_size(rng):
    """Output = k x k block of the majority color, k = its object count."""
    H, W = int(rng.integers(10, 16)), int(rng.integers(10, 16))

    def sample():
        g = np.zeros((H, W), dtype=int)
        cols = rng.permutation(np.arange(1, 10))[:3]
        counts = rng.permutation([2, 3, 4])
        for col, n in zip(cols, counts):
            for _ in range(int(n)):
                pos = _place(g, 2, 2, rng)
                if pos:
                    g[pos[0], pos[1]] = col
                    g[pos[0] + (rng.integers(2)), pos[1] + 1] = col
        return g

    def transform(g):
        best, k = 0, 0
        for col in range(1, 10):
            n = sum(1 for c, _ in _objects(g) if c == col)
            if n > k:
                best, k = col, n
        return np.full((k, k), best, dtype=int)

    return sample, transform


def gen_frame_select(rng):
    """Several hollow frames; a corner marker names the frame color; output = that frame's interior."""
    H, W = int(rng.integers(14, 20)), int(rng.integers(14, 20))

    def sample():
        g = np.zeros((H, W), dtype=int)
        cols = rng.permutation(np.arange(1, 10))[:3]
        interiors = {}
        for col in cols:
            h, w = int(rng.integers(4, 6)), int(rng.integers(4, 6))
            pos = _place(g[1:], h, w, rng)
            if pos is None:
                continue
            r, c = pos[0] + 1, pos[1]
            g[r:r + h, c:c + w] = col
            inner = np.array(
                [[int(rng.choice([0, 1, 2, 3, 4, 6, 7, 8, 9]))
                  for _ in range(w - 2)] for _ in range(h - 2)])
            g[r + 1:r + h - 1, c + 1:c + w - 1] = inner
            interiors[int(col)] = inner
        pick = int(rng.choice([c for c in cols if int(c) in interiors]))
        g[0, 0] = pick
        return g

    def transform(g):
        pick = int(g[0, 0])
        best = None
        for col, cells in _objects(g):
            if col != pick:
                continue
            ys = [y for y, _ in cells]
            xs = [x for _, x in cells]
            h, w = max(ys) - min(ys) + 1, max(xs) - min(xs) + 1
            if h >= 3 and w >= 3 and len(cells) >= 2 * h + 2 * w - 4:
                best = (min(ys), min(xs), h, w)
        if best is None:
            return g.copy()
        r, c, h, w = best
        return g[r + 1:r + h - 1, c + 1:c + w - 1].copy()

    return sample, transform


def gen_ray_hit(rng):
    """Color-8 arrows shoot rays (rightward); first object hit turns color 8's row-parity color."""
    H, W = int(rng.integers(10, 16)), int(rng.integers(12, 18))
    hit_col = int(rng.choice([2, 3, 4, 6, 7, 9]))

    def sample():
        g = np.zeros((H, W), dtype=int)
        for _ in range(int(rng.integers(2, 4))):
            h, w = int(rng.integers(2, 4)), int(rng.integers(2, 3))
            pos = _place(g, h, w, rng)
            if pos and pos[1] > 3:
                g[pos[0]:pos[0] + h, pos[1]:pos[1] + w] = int(rng.choice([1, 5]))
        for _ in range(int(rng.integers(1, 3))):
            r = int(rng.integers(0, H))
            if g[r, 0] == 0:
                g[r, 0] = 8
        return g

    def transform(g):
        out = g.copy()
        H2, W2 = g.shape
        for r in range(H2):
            if g[r, 0] == 8:
                for c in range(1, W2):
                    if g[r, c] != 0:
                        col = g[r, c]
                        for oc, cells in _objects(g):
                            if oc == col and (r, c) in cells:
                                for y, x in cells:
                                    out[y, x] = hit_col
                                break
                        break
        return out

    return sample, transform


def gen_symmetry_repair(rng):
    """Left-right mirror grid with a damaged patch (set to 0) + a distractor dot; restore symmetry, drop distractor."""
    H, W = int(rng.integers(8, 13)), int(rng.integers(4, 7)) * 2

    def sample_pair():
        half = np.array([[int(rng.choice([0, 0, 1, 2, 3, 4]))
                          for _ in range(W // 2)] for _ in range(H)])
        full = np.concatenate([half, half[:, ::-1]], axis=1)
        g = full.copy()
        ph, pw = int(rng.integers(2, 4)), int(rng.integers(2, 4))
        r, c = int(rng.integers(0, H - ph)), int(rng.integers(0, W - pw))
        g[r:r + ph, c:c + pw] = 0
        dr, dc = int(rng.integers(0, H)), int(rng.integers(0, W))
        g[dr, dc] = 9
        return g, full

    return sample_pair, None  # pair generator (input, output sampled together)


def gen_parity_recolor(rng):
    """Recolor each object by centroid parity: (even,even)->a, (odd,odd)->b, mixed->c."""
    H, W = int(rng.integers(10, 16)), int(rng.integers(10, 16))
    a, b, c3 = [int(x) for x in rng.permutation(np.arange(1, 10))[:3]]

    def sample():
        g = np.zeros((H, W), dtype=int)
        for _ in range(int(rng.integers(4, 7))):
            h, w = int(rng.integers(1, 3)), int(rng.integers(1, 3))
            pos = _place(g, h + 1, w + 1, rng)
            if pos:
                g[pos[0]:pos[0] + h, pos[1]:pos[1] + w] = int(rng.integers(1, 10))
        return g

    def transform(g):
        out = g.copy()
        for col, cells in _objects(g):
            cy = int(round(sum(y for y, _ in cells) / len(cells)))
            cx = int(round(sum(x for _, x in cells) / len(cells)))
            new = a if (cy % 2 == 0 and cx % 2 == 0) else (
                b if (cy % 2 == 1 and cx % 2 == 1) else c3)
            for y, x in cells:
                out[y, x] = new
        return out

    return sample, transform


CONCEPTS = {
    "key_legend": gen_key_legend,
    "conditional_size": gen_conditional_size,
    "gravity_obstacles": gen_gravity_obstacles,
    "count_to_size": gen_count_to_size,
    "frame_select": gen_frame_select,
    "ray_hit": gen_ray_hit,
    "symmetry_repair": gen_symmetry_repair,
    "parity_recolor": gen_parity_recolor,
}


def make_task(concept_fn, rng, n_train=3):
    maker = concept_fn(rng)
    pairs = []
    for _ in range(n_train + 1):
        if maker[1] is None:  # paired sampler (input+output together)
            gi, go = maker[0]()
        else:
            gi = maker[0]()
            go = maker[1](gi)
        if go.size == 0 or go.shape[0] > 30 or go.shape[1] > 30:
            return None
        pairs.append({"input": gi.tolist(), "output": go.tolist()})
    ins = [json.dumps(p["input"]) for p in pairs]
    if len(set(ins)) < len(ins):  # duplicate inputs -> degenerate
        return None
    if all(p["input"] == p["output"] for p in pairs):  # identity task
        return None
    return {"train": pairs[:n_train], "test": [pairs[n_train]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sdg/batch01")
    ap.add_argument("--per-concept", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    n_ok = 0
    for name, fn in CONCEPTS.items():
        made = 0
        attempts = 0
        while made < args.per_concept and attempts < args.per_concept * 8:
            attempts += 1
            try:
                t = make_task(fn, rng)
            except Exception:
                continue
            if t is None:
                continue
            with open(os.path.join(args.out, f"sdg_{name}_{made:03d}.json"), "w") as f:
                json.dump(t, f)
            made += 1
            n_ok += 1
        print(f"{name}: {made}/{args.per_concept} (attempts {attempts})")
    print(f"total tasks written: {n_ok} -> {args.out}")


if __name__ == "__main__":
    main()

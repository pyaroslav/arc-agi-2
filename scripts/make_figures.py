"""Generate the paper's two figures from the actual candidate caches + the
sourced forum numbers. Outputs docs/figures/*.png with Wilson 95% CIs.
"""
import glob
import json
import math
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, _SRC)
from nvarc.selection import score_kgmon

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures")
os.makedirs(FIG, exist_ok=True)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def load_solutions(d="data/arc-agi-2/evaluation"):
    s = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        s[os.path.splitext(os.path.basename(f))[0]] = [x["output"] for x in json.load(open(f))["test"]]
    return s


def cache_stats(pkl, sols):
    dec = pickle.load(open(pkl, "rb"))
    n = oracle = top2 = 0
    for bk, pool in dec.items():
        tid, ti = bk.rsplit("_", 1)
        if tid not in sols:
            continue
        gold = sols[tid][int(ti)]
        n += 1
        oracle += any(np.array_equal(s["solution"], gold) for s in pool.values())
        ordered = score_kgmon({k: v for k, v in pool.items()})
        top2 += any(np.array_equal(g, gold) for g in ordered[:2])
    return n, oracle, top2


def err(k, n):
    p = k / n
    lo, hi = wilson(k, n)
    return [[p - lo], [hi - p]]


def fig1(sols):
    rows = []
    for label, pkl in [("shallow\n(aug8, p0.2, 2v)", "outputs/cache_aug8.pkl"),
                       ("deep\n(aug16, p0.1, 3v)", "outputs/cache_deep.pkl")]:
        if os.path.exists(pkl):
            rows.append((label,) + cache_stats(pkl, sols))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(rows)); w = 0.38
    orc = [r[2] / r[1] for r in rows]; p2 = [r[3] / r[1] for r in rows]
    orc_e = np.hstack([err(r[2], r[1]) for r in rows]); p2_e = np.hstack([err(r[3], r[1]) for r in rows])
    ax.bar(x - w/2, orc, w, yerr=orc_e, capsize=4, label="oracle (coverage ceiling)", color="#4C72B0")
    ax.bar(x + w/2, p2, w, yerr=p2_e, capsize=4, label="achieved pass@2 (score_kgmon)", color="#DD8452")
    for i, r in enumerate(rows):
        ax.text(i, max(orc[i], p2[i]) + 0.04, f"gap {orc[i]-p2[i]:+.1%}\nn={r[1]}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel("pass@2 (per test-output)"); ax.set_ylim(0, 0.55)
    ax.set_title("Fig 1. Coverage-bound: the gap is generation, not selection\n(95% Wilson CIs; differences are within noise at this n)")
    ax.legend(loc="upper right", fontsize=9); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_coverage.png"), dpi=130); plt.close(fig)
    print("wrote fig1_coverage.png", rows)


def fig2():
    # sourced from data/discussions: NVARC 30.5->27 (winner post), TRM 22->10.83,
    # TOPAS 36->11.67 (Bitterbot post); "ours" = our two submissions: shallow
    # (local per-task 22.8 -> LB 25.00) and deep (local 25.2 = shallow + paired
    # +2.4pp -> LB 19.17). Bars = local eval vs leaderboard.
    fams = ["LLM + TTT\n(NVARC)", "LLM + TTT\n(ours, shallow)", "LLM + TTT\n(ours, deep)",
            "Tiny Recursive\n(TRM)", "Deep-recursive\n(TOPAS)"]
    local = [30.5, 22.8, 25.2, 22.0, 36.0]; lb = [27.0, 25.0, 19.17, 10.83, 11.67]
    drop_rel = [(l - b) / l * 100 for l, b in zip(local, lb)]
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    x = np.arange(len(fams)); w = 0.38
    ax.bar(x - w/2, local, w, label="local public-eval", color="#55A868")
    ax.bar(x + w/2, lb, w, label="public leaderboard", color="#C44E52")
    for i in range(len(fams)):
        s = f"-{drop_rel[i]:.0f}% rel" if drop_rel[i] >= 0 else f"+{-drop_rel[i]:.0f}% rel"
        ax.text(i, max(local[i], lb[i]) + 1.2, s, ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(fams)
    ax.set_ylabel("pass@2 (%)"); ax.set_ylim(0, 45)
    ax.set_title("Fig 2. Eval->LB generalization gap by approach family\n"
                 "(LLM+TTT is far more LB-robust; forum-sourced + our v7 submission)")
    ax.legend(fontsize=9); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_family_gap.png"), dpi=130); plt.close(fig)
    print("wrote fig2_family_gap.png")


if __name__ == "__main__":
    sols = load_solutions()
    fig1(sols)
    fig2()

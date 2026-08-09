"""Compare two candidate caches (e.g. aug_n=8 vs 16) on their COMMON tasks:
oracle ceiling (coverage) + pass@2 (score_kgmon). Isolates the coverage lever.

  python scripts/compare_caches.py --a outputs/cache_aug8.pkl --b outputs/cache_aug16.pkl
"""
import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, _SRC)
from nvarc.selection import score_kgmon


def load_solutions(d):
    s = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        s[os.path.splitext(os.path.basename(f))[0]] = [x["output"] for x in json.load(open(f))["test"]]
    return s


def stats(decoded, sols, keys):
    oracle = top2 = n = 0
    for bk in keys:
        tid, ti = bk.rsplit("_", 1)
        correct = sols[tid][int(ti)]
        n += 1
        g = decoded[bk]
        if any(np.array_equal(s["solution"], correct) for s in g.values()):
            oracle += 1
        ordered = score_kgmon({k: v for k, v in g.items()})
        if any(np.array_equal(x, correct) for x in ordered[:2]):
            top2 += 1
    return n, oracle, top2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="outputs/cache_aug8.pkl")
    ap.add_argument("--b", default="outputs/cache_aug16.pkl")
    ap.add_argument("--la", default="aug8")
    ap.add_argument("--lb", default="aug16")
    ap.add_argument("--data", default="data/arc-agi-2/evaluation")
    args = ap.parse_args()
    A = pickle.load(open(args.a, "rb"))
    B = pickle.load(open(args.b, "rb"))
    sols = load_solutions(args.data)
    common = sorted(set(A) & set(B))
    if not common:
        print("no common base_keys between caches yet"); return
    na, oa, ta = stats(A, sols, common)
    nb, ob, tb = stats(B, sols, common)
    print(f"common test-outputs: {len(common)}")
    print(f"{'setting':8} {'oracle(coverage)':>18} {'pass@2':>10}")
    print("-" * 42)
    print(f"{args.la:8} {oa}/{na}={oa/na:>6.1%}{'':>5} {ta}/{na}={ta/na:>6.1%}")
    print(f"{args.lb:8} {ob}/{nb}={ob/nb:>6.1%}{'':>5} {tb}/{nb}={tb/nb:>6.1%}")
    print(f"\nΔ oracle (coverage): {(ob-oa)/na:+.1%}   Δ pass@2: {(tb-ta)/na:+.1%}")
    print("Interpretation: if Δoracle is large, more TTFT augmentation buys coverage (the lever).")


if __name__ == "__main__":
    main()

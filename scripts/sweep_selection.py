"""Roadmap item #1 (highest ROI, CPU-only): sweep FREE selection variants over the
cached candidate scores to find the best pass@2 selector without any GPU recompute.

Each cached candidate stores: beam_score (DFS NLL) + score_aug (list of m=8 per-aug
NLLs). We sweep count-weight x per-guess aggregation x across-guess reduction and
report pass@2 vs the NVARC baselines and the oracle ceiling.

  python scripts/sweep_selection.py [--pkl outputs/repro_decoded.pkl]
"""
import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, _SRC)
from nvarc.selection import score_sum, score_kgmon, score_full_probmul_3


def load_solutions(data_dir):
    sols = {}
    for f in sorted(glob.glob(os.path.join(data_dir, "*.json"))):
        tid = os.path.splitext(os.path.basename(f))[0]
        sols[tid] = [x["output"] for x in json.load(open(f))["test"]]
    return sols


def make_getter(count_weight, per_guess, across):
    def getter(group):
        cnt = count_weight * len(group)
        per = []
        for g in group:
            sa = g["score_aug"]
            per.append(sum(sa) if per_guess == "sum" else (sum(sa) / max(1, len(sa))))
        if across == "mean":
            aug = sum(per) / len(per)
        elif across == "min":
            aug = min(per)
        else:  # sum
            aug = sum(per)
        return cnt - aug                      # higher = better (NLL subtracted)
    return getter


def pass2(decoded, sols, selector):
    hit = n = 0
    for bk, guesses in decoded.items():
        tid, ti = bk.rsplit("_", 1)
        if tid not in sols:
            continue
        n += 1
        correct = sols[tid][int(ti)]
        ordered = selector({k: g for k, g in guesses.items()})
        if any(np.array_equal(g, correct) for g in ordered[:2]):
            hit += 1
    return hit, n


def oracle(decoded, sols):
    hit = n = 0
    for bk, guesses in decoded.items():
        tid, ti = bk.rsplit("_", 1)
        if tid not in sols:
            continue
        n += 1
        correct = sols[tid][int(ti)]
        if any(np.array_equal(g["solution"], correct) for g in guesses.values()):
            hit += 1
    return hit, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="outputs/repro_decoded.pkl")
    ap.add_argument("--data", default="data/arc-agi-2/evaluation")
    args = ap.parse_args()
    import pickle
    decoded = pickle.load(open(args.pkl, "rb"))
    sols = load_solutions(args.data)

    oh, on = oracle(decoded, sols)
    print(f"cache: {on} test-outputs | ORACLE ceiling: {oh}/{on} = {oh/on:.1%}\n")

    rows = []
    for cw, pg, ac in itertools.product([0, 1, 2], ["sum", "mean"], ["mean", "min", "sum"]):
        h, n = pass2(decoded, sols, lambda g, getter=make_getter(cw, pg, ac): score_sum(g, getter))
        rows.append((h / n, f"count*{cw} - {ac}({pg}_8)", h, n))
    # NVARC baselines
    for name, sel in [("NVARC score_kgmon", score_kgmon), ("NVARC score_full_probmul_3", score_full_probmul_3)]:
        h, n = pass2(decoded, sols, sel)
        rows.append((h / n, name, h, n))

    rows.sort(reverse=True)
    print(f"{'pass@2':>8}  selector")
    print("-" * 50)
    for acc, name, h, n in rows:
        tag = "  <-- NVARC" if name.startswith("NVARC") else ""
        print(f"{acc:7.1%}  {name:32s} ({h}/{n}){tag}")
    best = rows[0]
    print(f"\nbest free selector: {best[1]}  ->  {best[0]:.1%}  (oracle {oh/on:.1%})")
    if on < 40:
        print("NOTE: sample is small/noisy — re-run on the full 60-120 task cache before trusting deltas.")


if __name__ == "__main__":
    main()

"""Diagnose how much pass@2 we lose to SELECTION vs GENERATION.

Loads the decoded candidates saved by repro_nvarc.py and compares:
  - actual pass@2 under each selection algorithm (top-2)
  - oracle pass@2 (correct grid anywhere in the candidate pool) = the ceiling
  - rank of the correct candidate under score_kgmon (when present)

A big oracle-minus-actual gap => invest in selection (quick win).
A low oracle => invest in generation (TTFT/decoder/data).

  python scripts/analyze_selection.py [--pkl outputs/repro_decoded.pkl]
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
from nvarc.selection import score_kgmon, score_full_probmul_3


def load_solutions(data_dir):
    sols = {}
    for f in sorted(glob.glob(os.path.join(data_dir, "*.json"))):
        tid = os.path.splitext(os.path.basename(f))[0]
        sols[tid] = [x["output"] for x in json.load(open(f))["test"]]
    return sols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="outputs/repro_decoded.pkl")
    ap.add_argument("--data", default="data/arc-agi-2/evaluation")
    args = ap.parse_args()

    decoded = pickle.load(open(args.pkl, "rb"))
    sols = load_solutions(args.data)

    n = 0
    oracle = top1 = kgmon2 = probmul2 = 0
    ranks_when_present = []
    present_not_top2 = 0
    for bk, guesses in decoded.items():
        tid, ti = bk.rsplit("_", 1)
        if tid not in sols:
            continue
        correct = sols[tid][int(ti)]
        n += 1
        cand_solutions = [g["solution"] for g in guesses.values()]
        present = any(np.array_equal(c, correct) for c in cand_solutions)
        oracle += present
        # ordered unique solutions under each algo
        ord_kg = score_kgmon({k: g for k, g in guesses.items()})
        ord_pm = score_full_probmul_3({k: g for k, g in guesses.items()})
        eq = lambda lst, k: any(np.array_equal(g, correct) for g in lst[:k])
        top1 += eq(ord_kg, 1)
        kgmon2 += eq(ord_kg, 2)
        probmul2 += eq(ord_pm, 2)
        if present:
            rank = next((i for i, g in enumerate(ord_kg) if np.array_equal(g, correct)), None)
            ranks_when_present.append(rank)
            if rank is not None and rank >= 2:
                present_not_top2 += 1

    if n == 0:
        print("no scored outputs yet"); return
    print(f"scored test-outputs: {n}")
    print(f"  oracle (correct in pool)     : {oracle}/{n} = {oracle/n:.1%}   <- generation ceiling")
    print(f"  pass@1 (kgmon top-1)         : {top1}/{n} = {top1/n:.1%}")
    print(f"  pass@2 (score_kgmon)         : {kgmon2}/{n} = {kgmon2/n:.1%}   <- current")
    print(f"  pass@2 (score_full_probmul_3): {probmul2}/{n} = {probmul2/n:.1%}")
    print(f"  SELECTION GAP (oracle-kgmon) : {(oracle-kgmon2)}/{n} = {(oracle-kgmon2)/n:.1%}")
    print(f"  correct-present-but-not-top2 : {present_not_top2}")
    if ranks_when_present:
        rks = [r for r in ranks_when_present if r is not None]
        print(f"  rank of correct when present : min={min(rks)} median={int(np.median(rks))} max={max(rks)} "
              f"(0-indexed; #pooled solutions varies)")


if __name__ == "__main__":
    main()

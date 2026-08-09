"""Standalone symbolic-solver coverage on ARC-AGI-2 eval (CPU). Also reports
net-new vs the neural candidate cache if present."""
import glob
import json
import os
import sys

import numpy as np

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, _SRC)
from arc.symbolic import synthesize, predict


def main():
    data_dir = "data/arc-agi-2/evaluation"
    files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    n_out = solved = had_prog = 0
    symbolic_correct_keys = set()
    for f in files:
        tid = os.path.splitext(os.path.basename(f))[0]
        t = json.load(open(f))
        progs = synthesize(t["train"])
        for ti, test in enumerate(t["test"]):
            n_out += 1
            if progs:
                had_prog += 1
            preds = predict(t["train"], test["input"])[:2]   # pass@2
            if any(p == test["output"] for p in preds):
                solved += 1
                symbolic_correct_keys.add(f"{tid}_{ti}")
    print(f"symbolic solver on {n_out} eval test-outputs:")
    print(f"  any consistent program : {had_prog}/{n_out} = {had_prog/n_out:.1%}")
    print(f"  SOLVED (pass@2)         : {solved}/{n_out} = {solved/n_out:.1%}")

    # net-new vs neural cache
    pkl = "outputs/repro_decoded.pkl"
    if os.path.exists(pkl):
        import pickle
        dec = pickle.load(open(pkl, "rb"))
        sols = {}
        for f in files:
            tid = os.path.splitext(os.path.basename(f))[0]
            sols[tid] = [x["output"] for x in json.load(open(f))["test"]]
        neural_oracle = set()
        for bk, g in dec.items():
            tid, ti = bk.rsplit("_", 1)
            if tid in sols and any(np.array_equal(s["solution"], sols[tid][int(ti)]) for s in g.values()):
                neural_oracle.add(bk)
        net_new = symbolic_correct_keys - neural_oracle
        cached = {bk for bk in dec}
        sym_in_cached = {k for k in symbolic_correct_keys if k in cached}
        print(f"  (neural cache covers {len(cached)} outputs; neural oracle-correct={len(neural_oracle)})")
        print(f"  symbolic-correct among cached outputs : {len(sym_in_cached)}")
        print(f"  NET-NEW (symbolic right, neural missed): {len(net_new & cached)}")


if __name__ == "__main__":
    main()

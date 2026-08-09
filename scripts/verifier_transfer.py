"""Transfer test for the learned verifier: train on a (big) training-task cache,
evaluate selection pass@2 on a held-out EVAL cache — the deployment condition.

  python scripts/verifier_transfer.py \
      --train-pkl outputs/cache_train1000.pkl --train-data data/arc-agi-2/training \
      --test-pkl  outputs/cache_deep.pkl     --test-data  data/arc-agi-2/evaluation

Reports oracle / score_kgmon / verifier / rank-blend pass@2 on the test cache.
"""
import argparse
import os
import pickle
import sys

import numpy as np

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, _SRC)
from arc.verifier import make_dataset, _pass2_from_scores, new_model, FEATURE_ORDER
from nvarc.selection import score_kgmon


def load_solutions(data_dir):
    import glob, json
    sols = {}
    for f in sorted(glob.glob(os.path.join(data_dir, "*.json"))):
        tid = os.path.splitext(os.path.basename(f))[0]
        sols[tid] = [x["output"] for x in json.load(open(f))["test"]]
    return sols


def kgmon_scores(cache, groups):
    """score_kgmon rank per (base_key, solution grid), higher = better."""
    out = {}
    for bk, pool in cache.items():
        ordered = score_kgmon({k: v for k, v in pool.items()})  # best-first grids
        for rank, g in enumerate(ordered):
            out[(bk, tuple(map(tuple, g)))] = -rank
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-pkl", default="outputs/cache_train1000.pkl")
    ap.add_argument("--train-data", default="data/arc-agi-2/training")
    ap.add_argument("--test-pkl", default="outputs/cache_deep.pkl")
    ap.add_argument("--test-data", default="data/arc-agi-2/evaluation")
    args = ap.parse_args()

    Xtr, ytr, gtr, _ = make_dataset(pickle.load(open(args.train_pkl, "rb")),
                                    load_solutions(args.train_data))
    print(f"train {args.train_pkl}: {len(set(gtr))} outputs, {len(ytr)} candidates, "
          f"{int(ytr.sum())} positive")

    te_sols = load_solutions(args.test_data)
    te_cache = pickle.load(open(args.test_pkl, "rb"))
    Xte, yte, gte, candsols = make_dataset(te_cache, te_sols)
    oracle = sum(1 for bk in set(gte)
                 if any(s == te_sols[bk.rsplit('_', 1)[0]][int(bk.rsplit('_', 1)[1])]
                        for s, g in zip(candsols, gte) if g == bk))

    kg = kgmon_scores(te_cache, gte)
    kg_scores = np.array([kg.get((bk, tuple(map(tuple, s))), -999)
                          for bk, s in zip(gte, candsols)])
    kg_c, n = _pass2_from_scores(kg_scores, gte, candsols, te_sols)
    print(f"  test oracle               : {oracle}/{n} = {oracle/n:.1%}")
    print(f"  score_kgmon pass@2 (NVARC): {kg_c}/{n} = {kg_c/n:.1%}")

    m = new_model()
    m.fit(Xtr, ytr)
    v_scores = (m.predict_proba(Xte)[:, 1] if hasattr(m, "predict_proba")
                else m.decision_function(Xte))
    v_c, _ = _pass2_from_scores(v_scores, gte, candsols, te_sols)
    print(f"  VERIFIER (transfer)       : {v_c}/{n} = {v_c/n:.1%}   (Δ vs kgmon {(v_c-kg_c)/n:+.1%})")

    # rank-average blend within each output group
    blend = np.zeros(len(v_scores))
    for bk in set(gte):
        idx = [i for i, g in enumerate(gte) if g == bk]
        vr = np.argsort(np.argsort([v_scores[i] for i in idx]))
        kr = np.argsort(np.argsort([kg_scores[i] for i in idx]))
        for j, i in enumerate(idx):
            blend[i] = (vr[j] + kr[j]) / 2.0
    b_c, _ = _pass2_from_scores(blend, gte, candsols, te_sols)
    print(f"  BLEND (rank-avg w/ kgmon) : {b_c}/{n} = {b_c/n:.1%}   (Δ vs kgmon {(b_c-kg_c)/n:+.1%})")

    if hasattr(m, "feature_importances_"):
        imp = sorted(zip(FEATURE_ORDER, m.feature_importances_), key=lambda x: -x[1])[:6]
        print("  top features:", ", ".join(f"{k}={v:.2f}" for k, v in imp))


if __name__ == "__main__":
    main()

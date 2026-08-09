"""Train + group-cross-validate the learned verifier vs NVARC's score_kgmon.

  python scripts/train_verifier.py --pkl outputs/cache_deep.pkl

Reports oracle / score_kgmon pass@2 / verifier pass@2 (held-out via GroupKFold).
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
from arc.verifier import make_dataset, candidate_features, _pass2_from_scores, new_model, FEATURE_ORDER
from nvarc.selection import score_kgmon


def load_solutions(d):
    s = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        s[os.path.splitext(os.path.basename(f))[0]] = [x["output"] for x in json.load(open(f))["test"]]
    return s


def kgmon_scores(cache, groups_order):
    """score_kgmon score per (base_key, solution) aligned to the dataset order."""
    # build a lookup: (bk, sol_key) -> rank-based score (higher=better)
    score = {}
    for bk, pool in cache.items():
        ordered = score_kgmon({k: v for k, v in pool.items()})  # best-first list of grids
        for rank, g in enumerate(ordered):
            score[(bk, tuple(map(tuple, g)))] = -rank   # higher = better
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="outputs/cache_aug8.pkl")
    ap.add_argument("--data", default="data/arc-agi-2/evaluation")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    cache = pickle.load(open(args.pkl, "rb"))
    sols = load_solutions(args.data)
    X, y, groups, candsols = make_dataset(cache, sols)
    n_bk = len(set(groups))
    print(f"cache {args.pkl}: {n_bk} test-outputs, {len(y)} candidates, {int(y.sum())} positive")
    if y.sum() < 5:
        print("too few positives to train a verifier; need a bigger/deeper cache.");

    # oracle + baseline
    oracle = sum(1 for bk in set(groups)
                 if any(s == sols[bk.rsplit('_',1)[0]][int(bk.rsplit('_',1)[1])]
                        for s, g in zip(candsols, groups) if g == bk))
    kg = kgmon_scores(cache, groups)
    kg_scores = np.array([kg.get((bk, tuple(map(tuple, s))), -999) for bk, s in zip(groups, candsols)])
    kg_c, n = _pass2_from_scores(kg_scores, groups, candsols, sols)
    print(f"  oracle (correct in pool)  : {oracle}/{n} = {oracle/n:.1%}")
    print(f"  score_kgmon pass@2 (NVARC): {kg_c}/{n} = {kg_c/n:.1%}")

    # group CV verifier
    try:
        from sklearn.model_selection import GroupKFold
    except Exception:
        print("sklearn unavailable"); return
    if n_bk < args.folds or y.sum() < args.folds:
        print("not enough groups/positives for CV; skipping verifier."); return
    gkf = GroupKFold(n_splits=args.folds)
    oof = np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups):
        m = new_model()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1] if hasattr(m, "predict_proba") else m.decision_function(X[te])
    v_c, _ = _pass2_from_scores(oof, groups, candsols, sols)
    print(f"  VERIFIER pass@2 (held-out): {v_c}/{n} = {v_c/n:.1%}   (Δ vs kgmon {(v_c-kg_c)/n:+.1%})")
    # feature importances
    m = new_model(); m.fit(X, y)
    if hasattr(m, "feature_importances_"):
        imp = sorted(zip(FEATURE_ORDER, m.feature_importances_), key=lambda x: -x[1])[:6]
        print("  top features:", ", ".join(f"{k}={v:.2f}" for k, v in imp))


if __name__ == "__main__":
    main()

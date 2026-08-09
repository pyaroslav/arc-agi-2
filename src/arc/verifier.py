"""Learned candidate verifier/reranker (Lever A: close the pass@N -> pass@2 gap).

The 2025 winner reported 40% pass@10 but only 30.5% pass@2 — "the scorer missed
10%." NVARC's selector (`score_kgmon` = count - mean(aug-NLL)) is a fixed formula.
Here we learn a ranker over richer per-candidate features (consensus, beam/aug-NLL
statistics, shape priors) to pick the top-2 better. Trained on a labeled candidate
cache (public eval, where we know the gold), evaluated with group cross-validation
so the number is honest (held-out tasks).
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List
import numpy as np

FEATURE_ORDER = [
    "count", "count_frac", "beam_min", "beam_mean",
    "aug_mean", "aug_min", "aug_max", "aug_std",
    "h", "w", "area", "is_modal_shape", "n_cands", "count_rank", "augmean_rank",
]


def candidate_features(pool: Dict) -> List[tuple]:
    """pool = {subkey: sample}; sample has beam_score, score_aug(list), solution.
    Returns [(feature_dict, solution_grid)] per UNIQUE candidate solution."""
    groups: Dict[tuple, list] = {}
    for s in pool.values():
        groups.setdefault(tuple(map(tuple, s["solution"])), []).append(s)
    total = max(1, len(pool))
    shape_count = Counter()
    for key, samples in groups.items():
        sol = samples[0]["solution"]
        shape_count[(len(sol), len(sol[0]))] += len(samples)
    modal_shape = shape_count.most_common(1)[0][0] if shape_count else None

    rows = []
    for key, samples in groups.items():
        sol = np.asarray(samples[0]["solution"]).tolist()   # normalize np array -> list
        h, w = len(sol), len(sol[0])
        beam = [s["beam_score"] for s in samples]
        augs = [x for s in samples for x in s["score_aug"]] or [0.0]
        rows.append({
            "count": len(samples), "count_frac": len(samples) / total,
            "beam_min": float(min(beam)), "beam_mean": float(np.mean(beam)),
            "aug_mean": float(np.mean(augs)), "aug_min": float(min(augs)),
            "aug_max": float(max(augs)), "aug_std": float(np.std(augs)),
            "h": h, "w": w, "area": h * w,
            "is_modal_shape": 1.0 if (h, w) == modal_shape else 0.0,
            "n_cands": len(groups), "_sol": sol,
        })
    # within-pool ranks (1 = best): high count good, low aug_mean good
    for name, kf, reverse in [("count_rank", "count", True), ("augmean_rank", "aug_mean", False)]:
        order = sorted(rows, key=lambda r: r[kf], reverse=reverse)
        for i, r in enumerate(order):
            r[name] = i
    return [(r, r["_sol"]) for r in rows]


def make_dataset(cache: Dict, solutions: Dict):
    """Return X, y, groups(base_key), sols."""
    X, y, groups, sols = [], [], [], []
    for bk, pool in cache.items():
        tid, ti = bk.rsplit("_", 1)
        if tid not in solutions:
            continue
        gold = solutions[tid][int(ti)]
        for f, sol in candidate_features(pool):
            X.append([f[k] for k in FEATURE_ORDER])
            y.append(1 if sol == gold else 0)
            groups.append(bk)
            sols.append(sol)
    return np.asarray(X, dtype=float), np.asarray(y), groups, sols


def _pass2_from_scores(scores, groups, sols, solutions):
    """Top-2 per base_key by `scores`; return #correct test-outputs."""
    by_bk: Dict[str, list] = {}
    for sc, bk, sol in zip(scores, groups, sols):
        by_bk.setdefault(bk, []).append((sc, sol))
    correct = 0
    for bk, lst in by_bk.items():
        tid, ti = bk.rsplit("_", 1)
        gold = solutions[tid][int(ti)]
        top2 = [s for _, s in sorted(lst, key=lambda x: x[0], reverse=True)[:2]]
        correct += any(s == gold for s in top2)
    return correct, len(by_bk)


def new_model():
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                          subsample=0.8)
    except Exception:
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000, class_weight="balanced")

"""Pool candidates from independent local runs into one pool, then select.

Each cache is a {output_key: {sample_key: candidates}} pickle written by
scripts/repro_nvarc.py. Pools are merged per output (sample keys namespaced by
run) and re-scored with score_kgmon; reports pass@2 and oracle for each
combination, plus paired discordant counts vs the first run.

  python scripts/analyze_pooling.py A=outputs/cache_aug8.pkl \
      B=outputs/cache_aug16.pkl C=outputs/cache_ttt_ep2.pkl
"""
import glob
import itertools
import json
import os
import pickle
import sys
from math import comb

sys.path.insert(0, "src")
from nvarc.selection import score_kgmon


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    runs = dict(a.split("=", 1) for a in sys.argv[1:])
    sols = {}
    for f in sorted(glob.glob("data/arc-agi-2/evaluation/*.json")):
        sols[os.path.splitext(os.path.basename(f))[0]] = [x["output"] for x in json.load(open(f))["test"]]
    caches = {n: pickle.load(open(p, "rb")) for n, p in runs.items()}
    common = sorted(set.intersection(*[set(c) for c in caches.values()]))
    common = [k for k in common if k.rsplit("_", 1)[0] in sols]

    def score(names):
        p2, orc = {}, {}
        for bk in common:
            true = sols[bk.rsplit("_", 1)[0]][int(bk.rsplit("_", 1)[1])]
            merged = {f"{n}|{sk}": v for n in names for sk, v in caches[n][bk].items()}
            ordered = [[list(r) for r in g] for g in score_kgmon(merged)]
            p2[bk] = any(g == true for g in ordered[:2])
            orc[bk] = any(g == true for g in ordered)
        return p2, orc

    names = list(runs)
    base_p2, _ = score([names[0]])
    print(f"common outputs: {len(common)}")
    for r in range(1, len(names) + 1):
        for combo in itertools.combinations(names, r):
            p2, orc = score(list(combo))
            gain = sum(p2[k] and not base_p2[k] for k in common)
            loss = sum(base_p2[k] and not p2[k] for k in common)
            n = len(common)
            print(f"{'+'.join(combo):<10} pass@2 {sum(p2.values()):3d}/{n} = {sum(p2.values())/n:5.1%}"
                  f"  oracle {sum(orc.values())/n:5.1%}  vs {names[0]}: +{gain}/-{loss}"
                  f" (McNemar p={mcnemar_p(gain, loss):.2f})")


if __name__ == "__main__":
    main()

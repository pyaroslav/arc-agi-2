#!/bin/bash
# Paired 120-task A/B: BARC-FT model vs baseline 4B. Both resume their caches.
cd "$(dirname "$0")/.." || exit 1
echo "=== extending BARC-FT eval to 120 tasks ==="
python scripts/repro_nvarc.py --model outputs/sft_4b_barc_merged --limit 120 \
  --aug-n 8 --dfs-prob 0.2 --resume --pkl-out outputs/cache_4b_barc.pkl \
  >> outputs/eval_4b_barc.log 2>&1
echo "=== extending baseline eval to 120 tasks ==="
python scripts/repro_nvarc.py --model data/nvarc/qwen3_4b_sft --limit 120 \
  --aug-n 8 --dfs-prob 0.2 --resume --pkl-out outputs/cache_aug8.pkl \
  >> outputs/eval_base120.log 2>&1
echo "=== paired comparison ==="
python3 - << 'EOF'
import glob, json, os, pickle, sys
sys.path.insert(0, "src")
from nvarc.selection import score_kgmon

sols = {}
for f in sorted(glob.glob("data/arc-agi-2/evaluation/*.json")):
    sols[os.path.splitext(os.path.basename(f))[0]] = [x["output"] for x in json.load(open(f))["test"]]

def correct_map(path):
    res = {}
    for bk, pool in pickle.load(open(path, "rb")).items():
        tid, oi = bk.rsplit("_", 1)[0], int(bk.rsplit("_", 1)[1])
        if tid not in sols:
            continue
        true = sols[tid][oi]
        ordered = [[list(r) for r in g] for g in score_kgmon(dict(pool))]
        res[bk] = any(g == true for g in ordered[:2])
    return res

base = correct_map("outputs/cache_aug8.pkl")
barc = correct_map("outputs/cache_4b_barc.pkl")
common = sorted(set(base) & set(barc))
b = sum(base[k] for k in common)
f = sum(barc[k] for k in common)
dp = sum(1 for k in common if barc[k] and not base[k])
dn = sum(1 for k in common if base[k] and not barc[k])
print(f"common outputs: {len(common)}")
print(f"baseline pass@2 : {b}/{len(common)} = {b/len(common):.1%}")
print(f"BARC-FT  pass@2 : {f}/{len(common)} = {f/len(common):.1%}")
print(f"discordant: FT-only {dp}, base-only {dn}, net {dp-dn:+d}")
EOF
echo "AB_DONE"

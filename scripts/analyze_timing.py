"""Summarize the timing-probe kernel: how often the engine's time caps fire,
and whether capped tasks are the ones it fails.

  python scripts/analyze_timing.py <dir with kernel log + submission.json>
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

out = sys.argv[1]
log = open(glob.glob(os.path.join(out, "*.log"))[0]).read()
lines = [l.encode().decode("unicode_escape", errors="replace")
         for l in re.findall(r'"data":"(.*?)"\}', log)]
t = defaultdict(lambda: {"dfs": [], "dfs_cap": 0, "task_cap": False, "skipped": 0,
                         "batches": 0, "ttft": None, "total": None})
for l in lines:
    for m in re.finditer(r"\[TIMING\] (\S+) (.*)", l):
        key, rest = m.group(1), m.group(2)
        d = t[key]
        if rest.startswith("ttft_done"):
            d["ttft"] = float(re.search(r"ttft_done ([\d.]+)s", rest).group(1))
            d["batches"] = int(re.search(r"batches (\d+)", rest).group(1))
        elif rest.startswith("dfs"):
            d["dfs"].append(float(re.search(r"dfs ([\d.]+)s", rest).group(1)))
            d["dfs_cap"] += "DFS_CAP" in rest
        elif rest.startswith("task_cap"):
            d["task_cap"] = True
            d["skipped"] = int(re.search(r"skipped (\d+)/", rest).group(1))
        elif rest.startswith("total"):
            d["total"] = float(re.search(r"total ([\d.]+)s", rest).group(1))

sols = {}
for f in glob.glob("data/arc-agi-2/evaluation/*.json"):
    sols[os.path.splitext(os.path.basename(f))[0]] = [x["output"] for x in json.load(open(f))["test"]]
sub = json.load(open(os.path.join(out, "submission.json")))
def task_score(k):
    if k not in sub or k not in sols:
        return None
    outs = sols[k]
    hits = sum(1 for i, o in enumerate(outs)
               if i < len(sub[k]) and o in (sub[k][i].get("attempt_1"), sub[k][i].get("attempt_2")))
    return hits / len(outs)

done = {k: d for k, d in t.items() if d["total"] is not None}
n = len(done)
n_dfs = sum(len(d["dfs"]) for d in done.values())
n_dfs_cap = sum(d["dfs_cap"] for d in done.values())
capped = [k for k, d in done.items() if d["dfs_cap"] or d["task_cap"]]
clean = [k for k in done if k not in capped]
totals = sorted(d["total"] for d in done.values())
ttfts = sorted(d["ttft"] for d in done.values() if d["ttft"] is not None)
print(f"tasks finished: {n}/120 (of {len(t)} started)")
print(f"per-task total: median {totals[n//2]:.0f}s, p90 {totals[int(n*0.9)]:.0f}s, max {totals[-1]:.0f}s")
print(f"TTFT phase: median {ttfts[len(ttfts)//2]:.0f}s, max {ttfts[-1]:.0f}s")
print(f"DFS calls: {n_dfs}, hit the 540s cap: {n_dfs_cap} ({n_dfs_cap/max(1,n_dfs):.0%})")
n_tc = sum(d["task_cap"] for d in done.values())
sk = sum(d["skipped"] for d in done.values())
tb = sum(d["batches"] for d in done.values())
print(f"tasks hitting the 1200s task cap: {n_tc}, decoding batches skipped: {sk}/{tb}")
print(f"tasks touched by ANY cap: {len(capped)}/{n}")
def rate(keys):
    sc = [task_score(k) for k in keys if task_score(k) is not None]
    return (sum(sc) / len(sc), len(sc)) if sc else (float("nan"), 0)
r1, c1 = rate(capped); r0, c0 = rate(clean)
print(f"pass@2 on capped tasks: {r1:.1%} (n={c1})   on uncapped tasks: {r0:.1%} (n={c0})")
print("capped tasks:", ", ".join(sorted(capped)[:20]) + (" ..." if len(capped) > 20 else ""))

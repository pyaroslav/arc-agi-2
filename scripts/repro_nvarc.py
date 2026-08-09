"""Reproduce NVARC locally on the public ARC-AGI-2 eval set (single RTX 5090).

  python scripts/repro_nvarc.py --model data/nvarc/qwen3_4b_sft --limit 5 --aug-n 8

Trains a per-task LoRA (TTFT) on challenges ONLY (no answer leakage), decodes
with turbo_dfs, votes via augmentation NLL, then scores pass@2 against solutions.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, _SRC)

from nvarc.arc_loader import ArcDataset
from nvarc.selection import ArcDecoder, score_kgmon
from nvarc.solver_local import LocalSolver


def load_eval(data_dir, limit=None, keys=None):
    challenges, solutions = {}, {}
    for f in sorted(glob.glob(os.path.join(data_dir, "*.json"))):
        tid = os.path.splitext(os.path.basename(f))[0]
        t = json.load(open(f))
        challenges[tid] = {"train": t["train"], "test": [{"input": x["input"]} for x in t["test"]]}
        solutions[tid] = [x["output"] for x in t["test"]]
    all_keys = sorted(challenges)
    if keys:
        sel = [k for k in keys if k in challenges]
    else:
        sel = all_keys[:limit] if limit else all_keys
    return challenges, solutions, sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data/nvarc/qwen3_4b_sft")
    ap.add_argument("--data", default="data/arc-agi-2/evaluation")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--keys", nargs="*", default=None)
    ap.add_argument("--aug-n", type=int, default=8, help="TTFT augmentation count (16=full NVARC)")
    ap.add_argument("--eval-aug-n", type=int, default=2)
    ap.add_argument("--dfs-prob", type=float, default=0.2, help="DFS keep threshold (lower=more coverage)")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--per-task-budget", type=float, default=900.0)
    ap.add_argument("--resume", action="store_true", help="reuse the cache pkl; skip done tasks")
    ap.add_argument("--pkl-out", default="outputs/repro_decoded.pkl", help="candidate cache path")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)  # live progress when piped/backgrounded
    except Exception:
        pass
    challenges, solutions, keys = load_eval(args.data, args.limit, args.keys)
    print(f"reproducing on {len(keys)} tasks: {keys}")

    solve_data = ArcDataset(queries=challenges, is_orig=True)       # NO replies (no leakage)
    score_data = ArcDataset(queries=challenges, is_orig=True)
    score_data.replies = solutions

    solver = LocalSolver(args.model, dfs_prob=args.dfs_prob)
    print(f"model loaded | max_new_tokens={solver.max_new_tokens} | dfs_prob={args.dfs_prob}")

    import pickle
    os.makedirs("outputs", exist_ok=True)
    all_decoded = {}
    done_keys = []
    if args.resume and os.path.exists(args.pkl_out):
        all_decoded = pickle.load(open(args.pkl_out, "rb"))
        done = {bk.rsplit("_", 1)[0] for bk in all_decoded}
        done_keys = [k for k in keys if k in done]
        print(f"resume: {len(done_keys)} tasks already cached, will skip them")
    t_all = time.time()
    import torch
    skip = set(done_keys)
    for i, key in enumerate(keys):
        if key in skip:
            continue
        t0 = time.time()
        try:
            solver.train_task(solve_data.change_keys([key]), aug_n=args.aug_n, lr=args.lr)
            dr = solver.solve_task(key, solve_data, eval_aug_n=args.eval_aug_n,
                                   end_time=time.time() + args.per_task_budget)
        except Exception as e:
            torch.cuda.empty_cache()
            print(f"[{i+1}/{len(keys)}] {key}  FAILED: {type(e).__name__}: {e}")
            continue
        all_decoded.update(dr)
        done_keys.append(key)
        pickle.dump(all_decoded, open(args.pkl_out, "wb"))
        sol = solutions[key][0]
        got = any(np.array_equal(s["solution"], sol) for v in dr.values() for s in v.values())
        # running pass@2 over completed tasks
        d = ArcDecoder(score_data.split_multi_replies(), n_guesses=2)
        d.decoded_results = all_decoded
        sub = score_data.get_submission(d.run_selection_algo(score_kgmon))
        run_score = score_data.change_keys(done_keys, keep_flags=True).validate_submission(sub)
        print(f"[{i+1}/{len(keys)}] {key}  any_correct={got}  "
              f"running_pass@2={run_score:.1f}/{len(done_keys)}={run_score/len(done_keys):.1%}  "
              f"({time.time()-t0:.0f}s)")

    decoder = ArcDecoder(score_data.split_multi_replies(), n_guesses=2)
    decoder.decoded_results = all_decoded
    submission = score_data.get_submission(decoder.run_selection_algo(score_kgmon))
    score = score_data.change_keys(keys, keep_flags=True).validate_submission(submission)
    dt = time.time() - t_all
    print("\n==== NVARC reproduction ====")
    decoder.benchmark_selection_algos()
    print(f"  pass@2 (score_kgmon): {score:.2f} / {len(keys)} = {score/len(keys):.1%}")
    print(f"  wall {dt:.0f}s ({dt/len(keys):.0f}s/task) -> 240 tasks ~ {dt/len(keys)*240/3600:.1f}h/GPU")


if __name__ == "__main__":
    main()

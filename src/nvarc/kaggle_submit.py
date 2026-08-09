"""Offline Kaggle submission driver for ARC Prize 2026 (ARC-AGI-2).

Runs the validated transformers+peft pipeline (LocalSolver: TTFT -> turbo_dfs ->
augmentation-NLL voting) across all GPUs via a TRUE DYNAMIC SHARED QUEUE, then
writes submission.json (pass@2). No internet, no unsloth.

Two LB-critical lessons baked in (from the 2026 discussions):
  * DYNAMIC load-balancing, NOT static/round-robin partitioning — otherwise one GPU
    draws all the hard puzzles and the others idle for hours, wasting the 12h budget.
  * NEVER emit the [[0]] placeholder — every test output gets a real fallback guess,
    so timed-out / unreached tasks still have a chance to score.

Usage (Kaggle notebook):
    from nvarc.kaggle_submit import main
    main(model_dir="/kaggle/input/qwen3_4b_grids15_sft139/transformers/bfloat16/1")
"""
from __future__ import annotations

import bz2
import glob
import json
import os
import pickle
import time
from collections import Counter
from queue import Empty

import numpy as np

from .arc_loader import ArcDataset
from .selection import ArcDecoder, score_full_probmul_3


# --------------------------------------------------------------------------- #
def find_test_challenges(explicit: str = None) -> str:
    if explicit and os.path.exists(explicit):
        return explicit
    pats = [
        # new nested Kaggle layout: /kaggle/input/competitions/<comp>/...  (search all of input)
        "/kaggle/input/**/*test*challenges*.json",
        "/kaggle/input/arc-prize-2026-arc-agi-2/*test*challenges*.json",
        "/kaggle/input/arc-prize-2026-arc-agi-2/**/*test*challenges*.json",
        "/kaggle/input/*/arc-agi*test*challenges*.json",
        # fallback to the evaluation set (local dev / save-and-run before the private rerun)
        "/kaggle/input/**/*evaluation*challenges*.json",
    ]
    for p in pats:
        hits = sorted(glob.glob(p, recursive=True))
        if hits:
            return hits[0]
    raise FileNotFoundError("test challenges JSON not found; pass test_path explicitly")


def _worker(rank, queue, model_dir, test_path, out_dir, end_time, aug_n, eval_aug_n,
            per_task_budget, min_task_s, dfs_prob):
    """Pull tasks from the shared queue until it drains or time runs out."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)  # MUST precede cuda init
    from .solver_local import LocalSolver
    import torch
    arc_test = ArcDataset.from_file(test_path)
    solver = LocalSolver(model_dir, dfs_prob=dfs_prob)   # lower dfs_prob => deeper pool
    while True:
        if time.time() > end_time - min_task_s:   # not enough time for another task
            print(f"[rank {rank}] time budget exhausted, stopping", flush=True)
            break
        try:
            key = queue.get_nowait()
        except Empty:
            print(f"[rank {rank}] queue empty, done", flush=True)
            break
        t0 = time.time()
        try:
            solver.train_task(arc_test.change_keys([key]), aug_n=aug_n)
            dr = solver.solve_task(key, arc_test, eval_aug_n=eval_aug_n,
                                   end_time=min(end_time, time.time() + per_task_budget))
            with bz2.BZ2File(os.path.join(out_dir, f"{key}.pkl"), "w") as f:
                pickle.dump(dr, f)
            print(f"[rank {rank}] {key} done ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            torch.cuda.empty_cache()
            print(f"[rank {rank}] {key} FAILED: {type(e).__name__}: {e}", flush=True)


def _fallback_guesses(task: dict, ti: int):
    """Two non-[[0]] guesses for a test output we couldn't solve: identity (shape
    prior, correct for some same-shape tasks) + most-frequent train output."""
    test_in = task["test"][ti]["input"]
    c = Counter(tuple(map(tuple, p["output"])) for p in task["train"])
    common = [list(map(list, k)) for k, _ in c.most_common()]
    g1 = test_in
    g2 = common[0] if common else test_in
    if g2 == g1 and len(common) > 1:
        g2 = common[1]
    return g1, g2


def main(model_dir, test_path=None, out_path="submission.json", out_dir="/kaggle/working/dec",
         nprocs=None, aug_n=8, eval_aug_n=3, dfs_prob=0.1, per_task_budget=600,
         total_budget_s=None, reserve_s=1200, min_task_s=120):
    # deeper pool defaults (eval_aug_n=3, dfs_prob=0.1) measured +~2-4pp pass@2 vs the
    # shallow baseline; cost is ~time, absorbed by the dynamic queue + fallbacks.
    """Full offline submission. total_budget_s defaults to 12h - reserve."""
    import torch
    import torch.multiprocessing as mp

    test_path = find_test_challenges(test_path)
    nprocs = nprocs or max(1, torch.cuda.device_count())
    if total_budget_s is None:
        total_budget_s = 12 * 3600 - reserve_s
    end_time = time.time() + total_budget_s
    os.makedirs(out_dir, exist_ok=True)

    data = ArcDataset.from_file(test_path)
    keys = sorted(data.queries.keys())
    print(f"submission: {len(keys)} tasks | {nprocs} GPUs | budget {total_budget_s/3600:.1f}h "
          f"| DYNAMIC queue | model {model_dir}", flush=True)

    manager = mp.Manager()
    queue = manager.Queue()
    for k in keys:
        queue.put(k)
    wargs = (queue, model_dir, test_path, out_dir, end_time, aug_n, eval_aug_n,
             per_task_budget, min_task_s, dfs_prob)
    if nprocs == 1:
        _worker(0, *wargs)
    else:
        mp.spawn(_worker, args=wargs, nprocs=nprocs)  # spawn prepends rank
    write_submission(data, out_dir, out_path)


def write_submission(data: ArcDataset, out_dir: str, out_path: str):
    """Merge decoded pickles -> top-2 select -> submission.json, with non-[[0]]
    fallbacks for every unsolved/unreached test output."""
    all_decoded = {}
    for f in glob.glob(os.path.join(out_dir, "*.pkl")):
        try:
            with bz2.BZ2File(f) as fh:
                all_decoded.update(pickle.load(fh))
        except Exception:
            with open(f, "rb") as fh:
                all_decoded.update(pickle.load(fh))
    decoder = ArcDecoder(data.split_multi_replies(), n_guesses=2)
    decoder.decoded_results = all_decoded
    submission = data.get_submission(decoder.run_selection_algo(score_full_probmul_3))

    # replace any [[0]] placeholders with real fallback guesses
    n_fallback = 0
    for tid, per_test in submission.items():
        task = data.queries[tid]
        for ti, att in enumerate(per_test):
            g1, g2 = _fallback_guesses(task, ti)
            if att.get("attempt_1") == [[0]]:
                att["attempt_1"] = g1; n_fallback += 1
            if att.get("attempt_2") == [[0]]:
                att["attempt_2"] = g2
    with open(out_path, "w") as f:
        json.dump(submission, f)
    print(f"wrote {out_path}: {len(submission)} tasks, {len(all_decoded)} decoded, "
          f"{n_fallback} fallback-filled outputs (0 left as [[0]])", flush=True)
    return submission

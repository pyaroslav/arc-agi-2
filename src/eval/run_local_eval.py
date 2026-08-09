"""End-to-end local evaluation: TTFT -> generate -> vote -> pass@2.

Usage:
  python src/eval/run_local_eval.py --config configs/default.yaml --profile dev_small
  python src/eval/run_local_eval.py --config configs/default.yaml --no-ttft --limit 20
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import copy

import numpy as np
import yaml

# make `arc` / `eval` importable whether run as script or module
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from arc.model import load_base
from arc.ttft import train_task_lora, remove_lora
from arc.decode import generate_candidates
from arc.vote import select_pass2
from arc.data import load_task_dir, sample_augmentations
from eval.metrics import output_correct, score_dataset


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str, profile: str = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    prof = cfg.pop("dev_small", None)
    if profile == "dev_small" and prof:
        cfg = deep_merge(cfg, prof)
    return cfg


def solve_task(bundle, task, cfg, use_ttft: bool, seed: int):
    train_pairs = task["train"]
    n_demos = len(train_pairs)
    rng = np.random.default_rng(seed)

    if use_ttft:
        t = cfg["ttft"]
        bundle = train_task_lora(
            bundle, task, steps=t["steps"], lr=t["lr"], lora_r=t["lora_r"],
            lora_alpha=t["lora_alpha"], lora_dropout=t["lora_dropout"],
            target_modules=t["target_modules"], grad_checkpointing=t.get("grad_checkpointing", False),
            seed=seed,
        )

    d = cfg["decode"]
    n_gen = min(d["num_candidates"], 8)
    gen_augs = sample_augmentations(n_gen, rng, n_demos=n_demos, include_identity=True)
    samples_per_aug = max(1, d["num_candidates"] // n_gen)

    preds_per_output = []
    for test in task["test"]:
        cands = generate_candidates(
            bundle, train_pairs, test["input"], gen_augs=gen_augs,
            samples_per_aug=samples_per_aug, temperature=d["temperature"],
            top_p=d["top_p"], max_new_tokens=d["max_new_tokens"],
        )
        preds = select_pass2(bundle, train_pairs, test["input"], cands,
                             num_augmentations=cfg["vote"]["num_augmentations"], seed=seed,
                             fallback=train_pairs[0]["output"])
        preds_per_output.append(preds)

    if use_ttft:
        bundle = remove_lora(bundle)
    return bundle, preds_per_output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--no-ttft", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, args.profile)
    limit = args.limit if args.limit is not None else cfg["eval"].get("limit")
    use_ttft = not args.no_ttft

    print(f"loading base model: {cfg['model']['base_model']}")
    bundle = load_base(cfg["model"]["base_model"], dtype=cfg["model"]["dtype"],
                       attn_implementation=cfg["model"]["attn_implementation"])

    tasks = load_task_dir(cfg["eval"]["data_dir"])
    ids = list(tasks)[: limit] if limit else list(tasks)
    print(f"evaluating {len(ids)} tasks | ttft={use_ttft}")

    results = {}
    t_start = time.time()
    for i, tid in enumerate(ids):
        task = tasks[tid]
        t0 = time.time()
        bundle, preds = solve_task(bundle, task, cfg, use_ttft, seed=i)
        truths = [t["output"] for t in task["test"]]
        results[tid] = {"predictions": preds, "truths": truths}
        hit = sum(output_correct(p, t) for p, t in zip(preds, truths)) / max(1, len(truths))
        print(f"[{i+1}/{len(ids)}] {tid}  pass@2={hit:.2f}  ({time.time()-t0:.1f}s)")

    agg = score_dataset(results)
    dt = time.time() - t_start
    print("\n==== RESULTS ====")
    for k, v in agg.items():
        print(f"  {k}: {v}")
    print(f"  wall: {dt:.0f}s  ({dt/max(1,len(ids)):.1f}s/task)")
    print(f"  -> projected for 240 tasks: {dt/max(1,len(ids))*240/3600:.1f}h")


if __name__ == "__main__":
    main()

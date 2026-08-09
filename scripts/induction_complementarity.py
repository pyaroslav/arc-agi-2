"""Roadmap #6 GATING experiment: does the offline BARC induction model add
NET-NEW coverage on ARC-AGI-2 beyond the neural (NVARC) pipeline?

For N eval tasks: sample programs, execution-verify on train pairs, run survivors
on test inputs, take the top-2 most-frequent train-consistent predictions (pass@2).
Then compare against the neural candidate cache to count NET-NEW solves.

  python scripts/induction_complementarity.py --tasks 20 --samples 32

Decision: integrate the big bet only if NET-NEW >= ~2-3 / N.
"""
import argparse
import glob
import json
import os
import pickle
import sys
from collections import Counter

import numpy as np

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, _SRC)
from induction.prompt import build_messages
from induction.execute import parse_code, verify_and_predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data/barc_induction")
    ap.add_argument("--data", default="data/arc-agi-2/evaluation")
    ap.add_argument("--tasks", type=int, default=20)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--neural-cache", default="outputs/cache_aug8.pkl")
    ap.add_argument("--max-model-len", type=int, default=12000)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data, "*.json")))[: args.tasks]
    tasks = {os.path.splitext(os.path.basename(f))[0]: json.load(open(f)) for f in files}

    # build one prompt per (task, test_idx)
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, max_model_len=args.max_model_len, gpu_memory_utilization=0.80,
              enable_prefix_caching=True, dtype="bfloat16", max_num_seqs=args.samples,
              enforce_eager=True)   # eager + bounded seqs => stable on long prompts
    tok = llm.get_tokenizer()

    items, prompts = [], []
    for tid, t in tasks.items():
        for ti, test in enumerate(t["test"]):
            msgs = build_messages(t["train"], test["input"])
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            if len(tok.encode(text)) > 8000:   # BARC skips very long prompts
                continue
            items.append((tid, ti, t["train"], test["input"], test["output"]))
            prompts.append(text)

    sp = SamplingParams(temperature=0.8, top_p=1.0, max_tokens=1536, n=args.samples)
    print(f"generating {args.samples} programs/task for {len(prompts)} test-outputs (per-task, robust) ...", flush=True)

    # neural oracle set
    neural_oracle = set()
    if os.path.exists(args.neural_cache) and os.path.getsize(args.neural_cache) > 0:
        dec = pickle.load(open(args.neural_cache, "rb"))
        for tid, t in tasks.items():
            for ti, test in enumerate(t["test"]):
                bk = f"{tid}_{ti}"
                if bk in dec and any(np.array_equal(s["solution"], test["output"])
                                     for s in dec[bk].values()):
                    neural_oracle.add(bk)

    from induction.execute import run_program
    n = len(items)
    had_prog = ind_correct = net_new = 0
    diag = {"programs": 0, "parsed": 0, "ran_ok": 0, "partial_train": 0, "full_train": 0}
    solved_preds = {}   # bk -> verified test predictions (for later integration)
    for idx, (tid, ti, train, test_in, gold) in enumerate(items):
        out = llm.generate([prompts[idx]], sp, use_tqdm=False)[0]   # per-task => can't OOM on concurrency
        consistent = []
        for comp in out.outputs:
            diag["programs"] += 1
            code = parse_code(comp.text)
            if not code:
                continue
            diag["parsed"] += 1
            try:
                outs_all = run_program(code, [p["input"] for p in train] + [test_in])
            except Exception:
                continue
            train_outs = outs_all[: len(train)]
            if any(o is not None for o in train_outs):
                diag["ran_ok"] += 1
            matches = sum(o == train[j]["output"] for j, o in enumerate(train_outs))
            if matches >= 1:
                diag["partial_train"] += 1
            ok = matches == len(train)
            preds = outs_all[len(train):]
            if ok and preds[0] is not None:
                diag["full_train"] += 1
                consistent.append(tuple(map(tuple, preds[0])))
        if consistent:
            had_prog += 1
        cnt = Counter(consistent)
        top2 = [list(map(list, k)) for k, _ in cnt.most_common(2)]
        correct = any(p == gold for p in top2)
        bk = f"{tid}_{ti}"
        if correct:
            ind_correct += 1
            if bk not in neural_oracle:
                net_new += 1
        solved_preds[bk] = [list(map(list, k)) for k in consistent]
        pickle.dump(solved_preds, open("outputs/induction_preds.pkl", "wb"))  # incremental save
        print(f"  [{idx+1}/{n}] {bk}: progs={len(consistent):3d} consistent  induction_correct={correct}  "
              f"neural_had={bk in neural_oracle}", flush=True)

    print(f"\n==== induction complementarity ({n} test-outputs) ====")
    print(f"  PROGRAM DIAGNOSTICS: {diag['programs']} sampled | parsed {diag['parsed']} | "
          f"ran-ok {diag['ran_ok']} | match>=1 train {diag['partial_train']} | match-ALL train {diag['full_train']}")
    print(f"  >=1 train-consistent program : {had_prog}/{n} = {had_prog/n:.1%}")
    print(f"  induction pass@2 (standalone): {ind_correct}/{n} = {ind_correct/n:.1%}")
    print(f"  neural oracle covered        : {len(neural_oracle)}/{n}")
    print(f"  NET-NEW (induction solved, neural missed): {net_new}/{n}")
    print(f"  -> {'INTEGRATE' if net_new >= max(2, n//10) else 'DROP'} the induction big-bet")


if __name__ == "__main__":
    main()

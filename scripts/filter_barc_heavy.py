"""Stream BARC-HEAVY (GPT-4o-generated ARC tasks) and keep the OOD slice.

For each task: teacher-forced 4B loss on up-to-4 pairs; tasks with loss > --thresh
are saved as {"train": pairs} JSONs into --out (sft_data-compatible). A manifest
JSONL (idx, loss) makes the pass resumable (restart skips processed lines).

  python scripts/filter_barc_heavy.py --limit 100000   # ~5-6h on the 5090
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import torch

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, _SRC)
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from nvarc.arc_loader import QwenFormatter
from nvarc.solver_local import ArcTokenizer
from train.sft_data import make_example

REPO = "barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data/nvarc/qwen3_4b_sft")
    ap.add_argument("--data-file", default="data_100k.jsonl")
    ap.add_argument("--out", default="data/barc_heavy/ood")
    ap.add_argument("--manifest", default="data/barc_heavy/manifest.jsonl")
    ap.add_argument("--thresh", type=float, default=0.10)
    ap.add_argument("--limit", type=int, default=100000)
    ap.add_argument("--max-tokens", type=int, default=6000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(args.manifest), exist_ok=True)
    done = 0
    if os.path.exists(args.manifest):
        with open(args.manifest) as f:
            done = sum(1 for _ in f)
    print(f"resuming after {done} processed records", flush=True)

    tok = ArcTokenizer()
    fmt = QwenFormatter(tok)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()

    ds = load_dataset(REPO, data_files=args.data_file, split="train", streaming=True)
    it = itertools.islice(iter(ds), done, args.limit)
    mf = open(args.manifest, "a", buffering=1)  # line-buffered: a kill loses nothing
    t0, n_kept, n_seen = time.time(), 0, 0
    for idx, r in enumerate(it, start=done):
        n_seen += 1
        loss = -1.0
        try:
            exs = r["examples"]
            if len(exs) >= 3:
                pairs = [{"input": [[int(c) for c in row] for row in e[0]],
                          "output": [[int(c) for c in row] for row in e[1]]}
                         for e in exs[:4]]
                if not any(len(p[k]) > 30 or len(p[k][0]) > 30
                           for p in pairs for k in ("input", "output")):
                    ids, labels = make_example(tok, fmt, pairs)
                    if len(ids) <= args.max_tokens:
                        with torch.no_grad():
                            loss = model(
                                input_ids=torch.tensor([ids]).cuda(),
                                labels=torch.tensor([labels]).cuda()).loss.item()
                        if loss > args.thresh:
                            with open(os.path.join(args.out, f"bh_{idx:06d}.json"), "w") as f:
                                json.dump({"train": pairs}, f)
                            n_kept += 1
        except Exception:
            loss = -2.0
        mf.write(json.dumps({"i": idx, "loss": round(loss, 4)}) + "\n")
        if n_seen % 500 == 0:
            mf.flush()
            rate = n_seen / (time.time() - t0)
            print(f"[{idx + 1}] kept {n_kept}/{n_seen} ({n_kept / max(1, n_seen):.0%}) "
                  f"| {rate:.1f} rec/s | ETA {(args.limit - idx) / max(rate, 0.1) / 3600:.1f}h",
                  flush=True)
    mf.close()
    print(f"DONE: kept {n_kept} OOD tasks -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

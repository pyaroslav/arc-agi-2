"""Generate a corpus of harder compositional ARC tasks -> data/hardgen/ (flat
list-of-pairs JSON, consumable by train.sft_data)."""
import argparse
import json
import os
import sys

import numpy as np

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, _SRC)
from train.hard_gen import gen_compositional_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default="data/hardgen")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    made = 0
    for i in range(args.n * 3):
        if made >= args.n:
            break
        k = int(rng.integers(3, 6))
        t = gen_compositional_task(rng, k=k)
        if t:
            json.dump(t["pairs"], open(os.path.join(args.out, f"comp_{made:06d}.json"), "w"))
            made += 1
    print(f"generated {made} hard compositional tasks -> {args.out}")


if __name__ == "__main__":
    main()
